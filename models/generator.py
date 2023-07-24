import re
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
import torchvision
from models.networks import BaseNetwork, get_norm_layer
from models.flownet2_pytorch.networks.resample2d_package.resample2d import Resample2d


# The following class has been taken from https://github.com/NVlabs/SPADE and modified. 
# The use of this code is subject to the terms and conditions set forth by the original code's license.
class SPADE(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc):
        super().__init__()

        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))
        self.norm_nc = norm_nc
        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 256

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        # Make sure segmap has the same spatial size with input.
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta
        return out


# The following class has been taken from https://github.com/NVlabs/SPADE and modified. 
# The use of this code is subject to the terms and conditions set forth by the original code's license.
class SPADEResnetBlock(nn.Module):
    def __init__(self, semantic_nc, fin, fout, opt):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if 'spectral' in opt.norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        spade_config_str = opt.norm_G.replace('spectral', '')
        self.norm_0 = SPADE(spade_config_str, fin, semantic_nc)
        self.norm_1 = SPADE(spade_config_str, fmiddle, semantic_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(spade_config_str, fin, semantic_nc)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.actvn(self.norm_s(x, seg)))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


# The following class has been taken from https://github.com/clovaai/stargan-v2 and modified. 
# The use of this code is subject to the terms and conditions set forth by the original code's license.
class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


# The following class has been taken from https://github.com/clovaai/stargan-v2 and modified. 
# The use of this code is subject to the terms and conditions set forth by the original code's license.
class AdaINResnetBlock(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim, opt):
        super().__init__()
        self.opt = opt
        self.actv = nn.LeakyReLU(0.2)
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim):
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        # apply spectral norm if specified
        if 'spectral' in self.opt.norm_G:
            self.conv1 = spectral_norm(self.conv1)
            self.conv2 = spectral_norm(self.conv2)
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        return out


class FlowNetwork(nn.Module):
    def __init__(self, opt, activation, norm_layer, nl, base):
        super().__init__()
        self.opt = opt
        self.ngf = opt.ngf
        self.nl = nl
        # The flow application operator (warping function).
        self.resample = Resample2d()

        # Use average pool 2D to downsample predicted flow.
        self.downsample = nn.AvgPool2d(2, stride=2, count_include_pad=False)

        # Encoder first layer
        enc_first = [nn.ReflectionPad2d(opt.initial_kernel_size // 2),
                     norm_layer(nn.Conv2d(opt.output_nc+opt.input_nc,
                                          self.ngf,
                                          kernel_size=opt.initial_kernel_size,
                                          padding=0)),
                     activation]
        self.enc = [nn.Sequential(*enc_first)]

        # Encoder downsampling layers
        for i in range(self.nl):
            mult_in = base**i
            mult_out = base**(i+1)

            # Conditional encoders
            enc_down = [norm_layer(nn.Conv2d(self.ngf * mult_in,
                                             self.ngf * mult_out,
                                             kernel_size=opt.kernel_size,
                                             stride=2,
                                             padding=1)),
                        activation]
            self.enc.append(nn.Sequential(*enc_down))
        self.enc = nn.ModuleList(self.enc)

        # Residual part of decoder
        fin = (base**self.nl) * self.ngf
        fout = fin
        self.dec_res = []
        for i in range(self.nl):
            self.dec_res.append(SPADEResnetBlock(opt.input_nc * opt.n_frames_G, fin, fout, opt))
        self.dec_res = nn.ModuleList(self.dec_res)

        # Upsampling part of decoder.
        self.dec_up = []
        self.dec_main = []
        for i in range(self.nl):
            fin = (base**(self.nl-i)) * self.ngf

            # In case of PixelShuffle, let it do the filters amount reduction.
            fout = (base**(self.nl-i-1)) * self.ngf if self.opt.no_pixelshuffle else fin
            if self.opt.no_pixelshuffle:
                self.dec_up.append(nn.Upsample(scale_factor=2))
            else:
                self.dec_up.append(nn.PixelShuffle(upscale_factor=2))
            self.dec_main.append(SPADEResnetBlock(opt.input_nc * opt.n_frames_G, fin, fout, opt))

        self.dec_up = nn.ModuleList(self.dec_up)
        self.dec_main = nn.ModuleList(self.dec_main)
        self.dec_flow = [nn.ReflectionPad2d(3),
                         nn.Conv2d(self.ngf, 2,
                                   kernel_size=opt.initial_kernel_size,
                                   padding=0)]
        self.dec_flow = nn.Sequential(*self.dec_flow)
        self.dec_mask = [nn.ReflectionPad2d(3),
                         nn.Conv2d(self.ngf, 1,
                                   kernel_size=opt.initial_kernel_size,
                                   padding=0)]
        self.dec_mask = nn.Sequential(*self.dec_mask)

    def forward(self, input, ref_input):
        # Get dimensions sizes
        NN_ref, _, H, W = ref_input.size()
        N = input.size()[0]
        N_ref = NN_ref // N

        # Repeat the conditional input for all reference frames
        seg = input.repeat(1, N_ref, 1, 1).view(NN_ref, -1, H, W)

        # Encode
        feats = []
        feat = ref_input
        for i in range(self.nl + 1):
            feat = self.enc[i](feat)
            feats.append(feat)

        # Decode
        for i in range(self.nl):
            feat = self.dec_res[i](feat, seg)
        for i in range(self.nl):
            feat = self.dec_main[i](feat, seg)
            feat = self.dec_up[i](feat)

        # Compute flow layer
        flow = self.dec_flow(feat)
        mask = self.dec_mask(feat)
        mask = (torch.tanh(mask) + 1) / 2
        flow = flow * mask
        down_flow = flow

        # Apply flow on features to match them spatially with the desired pose.
        flow_feats = []
        for i in range(self.nl + 1):
            flow_feats.append(self.resample(feats[i], down_flow))

            # Downsample flow and reduce its magnitude.
            down_flow = self.downsample(down_flow) / 2.0
        return flow, flow_feats, mask

class FramesEncoder(nn.Module):
    def __init__(self, opt, activation, norm_layer, nl, base):
        super().__init__()
        self.ngf = opt.ngf
        self.nl = nl
        cond_enc_first = [nn.ReflectionPad2d(opt.initial_kernel_size // 2),
                          norm_layer(nn.Conv2d(opt.output_nc+opt.input_nc,
                                               self.ngf,
                                               kernel_size=opt.initial_kernel_size,
                                               padding=0)),
                          activation]
        self.cond_enc = [nn.Sequential(*cond_enc_first)]
        for i in range(self.nl):
            mult_in = base**i
            mult_out = base**(i+1)
            cond_enc_down = [norm_layer(nn.Conv2d(self.ngf * mult_in,
                                                  self.ngf * mult_out,
                                                  kernel_size=opt.kernel_size,
                                                  stride=2,
                                                  padding=1)),
                             activation]
            self.cond_enc.append(nn.Sequential(*cond_enc_down))
        self.cond_enc = nn.ModuleList(self.cond_enc)

    def forward(self, ref_input):
        # Encode
        feats = []
        x_cond_enc = ref_input
        for i in range(self.nl + 1):
            x_cond_enc = self.cond_enc[i](x_cond_enc)
            feats.append(x_cond_enc)
        return feats


class RenderingNetwork(nn.Module):
    def __init__(self, opt, activation, norm_layer, nl, base):
        super().__init__()
        self.opt = opt
        self.ngf = opt.ngf
        self.naf = opt.naf
        self.nl = nl

        # Encode
        cond_enc_first = [nn.ReflectionPad2d(opt.initial_kernel_size // 2),
                          norm_layer(nn.Conv2d(opt.input_nc * opt.n_frames_G, self.ngf,
                                               kernel_size=opt.initial_kernel_size,
                                               padding=0)),
                          activation]
        self.cond_enc = [nn.Sequential(*cond_enc_first)]
        for i in range(self.nl):
            mult_in = base**i
            mult_out = base**(i+1)
            cond_enc_down = [norm_layer(nn.Conv2d(self.ngf * mult_in,
                                                  self.ngf * mult_out,
                                                  kernel_size=opt.kernel_size,
                                                  stride=2,
                                                  padding=1)),
                             activation]
            self.cond_enc.append(nn.Sequential(*cond_enc_down))
        self.cond_enc = nn.ModuleList(self.cond_enc)

        # Decode
        self.cond_dec = []
        if not self.opt.no_audio_input:
            self.cond_dec_audio = []
        self.cond_dec_up = []

        for i in range(self.nl):
            fin = (base**(self.nl-i)) * opt.ngf
            fout = (base**(self.nl-i-1)) * opt.ngf if opt.no_pixelshuffle else fin
            self.cond_dec.append(SPADEResnetBlock(fin, fin, fout, opt))
            if not self.opt.no_audio_input:
                self.cond_dec_audio.append(AdaINResnetBlock(fout, fout, self.naf, opt))
            if self.opt.no_pixelshuffle:
                self.cond_dec_up.append(nn.Upsample(scale_factor=2))
            else:
                self.cond_dec_up.append(nn.PixelShuffle(upscale_factor=2))

        self.cond_dec.append(SPADEResnetBlock(opt.ngf, opt.ngf, opt.ngf, opt))
        if not self.opt.no_audio_input:
            self.cond_dec_audio.append(AdaINResnetBlock(opt.ngf, opt.ngf, self.naf, opt))
        self.cond_dec.append(SPADEResnetBlock(opt.output_nc, opt.ngf, opt.ngf, opt))
        self.cond_dec = nn.ModuleList(self.cond_dec)
        if not self.opt.no_audio_input:
            self.cond_dec_audio = nn.ModuleList(self.cond_dec_audio)
        self.cond_dec_up = nn.ModuleList(self.cond_dec_up)

        self.conv_img = [nn.ReflectionPad2d(3), nn.Conv2d(self.ngf, 3, kernel_size=opt.initial_kernel_size, padding=0)]
        self.conv_img = nn.Sequential(*self.conv_img)

    def forward(self, input, feat_maps, warped, audio_feats):
        # Encode
        x_cond_enc = input
        for i in range(self.nl + 1):
            x_cond_enc = self.cond_enc[i](x_cond_enc)
        x = x_cond_enc

        # Decode
        for i in range(self.nl):
            x = self.cond_dec[i](x, feat_maps[-i-1])
            if not self.opt.no_audio_input:
                x = self.cond_dec_audio[i](x, audio_feats)
            x = self.cond_dec_up[i](x)

        x = self.cond_dec[self.nl](x, feat_maps[0])
        if not self.opt.no_audio_input:
            x = self.cond_dec_audio[self.nl](x, audio_feats)
        x = self.cond_dec[self.nl+1](x, warped)

        imgs = self.conv_img(F.leaky_relu(x, 2e-1))
        imgs = torch.tanh(imgs)
        return imgs

class headGANGenerator(BaseNetwork):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.resample = Resample2d()

        # Activation functions
        activation = nn.ReLU()
        leaky_activation = nn.LeakyReLU(2e-1)

        # Non-adaptive normalization layer
        norm_layer = get_norm_layer(opt, opt.norm_G_noadapt)

        # Number of times to (up/down)-sample spatial dimensions.

        nl = round(math.log(opt.crop_size // opt.down_size, 2))

        # If pixelshuffle is used, quadruple the number of filters when
        # upsampling, else simply double them.
        base = 2 if self.opt.no_pixelshuffle else 4

        if not self.opt.no_flownetwork:
            self.flow_network = FlowNetwork(opt, activation, norm_layer, nl, base)
        else:
            self.frames_encoder = FramesEncoder(opt, activation, norm_layer, nl, base)
        self.rendering_network = RenderingNetwork(opt, activation, norm_layer, nl, base)

    def forward(self, input, ref_input, audio_feats):
        if not self.opt.no_flownetwork:
            # Get flow and warped features.
            flow, flow_feats, mask = self.flow_network(input, ref_input)
        else:
            flow = torch.zeros_like(ref_input[:,:2,:,:])
            mask = torch.zeros_like(ref_input[:,:1,:,:])
            flow_feats = self.frames_encoder(ref_input)
        feat_maps = flow_feats

        # Apply flows on reference frame(s)
        ref_rgb_input = ref_input[:,-self.opt.output_nc:,:,:]
        if not self.opt.no_flownetwork:
            warped = self.resample(ref_rgb_input, flow)
        else:
            warped = ref_rgb_input
        imgs = self.rendering_network(input, feat_maps, warped, audio_feats)

        if self.opt.isTrain:
            return imgs, feat_maps, warped, flow, mask
        else:
            return imgs, warped, flow
