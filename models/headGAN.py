import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import models.networks as networks
import models.losses as losses
from models.base_model import BaseModel
import util.util as util
import torchvision


class headGANModelD(BaseModel):
    def name(self):
        return 'headGANModelD'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.gpu_ids = opt.gpu_ids
        self.n_frames_D = opt.n_frames_D
        self.output_nc = opt.output_nc
        self.input_nc = opt.input_nc

        # Image discriminator
        netD_input_nc = self.input_nc + self.output_nc
        self.netD = networks.define_D(opt, netD_input_nc)

        # Mouth discriminator
        if not opt.no_mouth_D:
            if not self.opt.no_audio_input:
                netDm_input_nc = opt.naf + self.output_nc
            else:
                netDm_input_nc = self.output_nc
            self.netDm = networks.define_D(opt, netDm_input_nc)

        # load networks
        if (opt.continue_train or opt.load_pretrain):
            self.load_network(self.netD, 'D', opt.which_epoch, opt.load_pretrain)
            if not opt.no_mouth_D:
                self.load_network(self.netDm, 'Dm', opt.which_epoch, opt.load_pretrain)
            print('---------- Discriminators loaded -------------')
        else:
            print('---------- Discriminators initialized -------------')

        # set loss functions and optimizers
        self.old_lr = opt.lr
        self.criterionGAN = losses.GANLoss(opt.gan_mode, tensor=self.Tensor, opt=self.opt)
        self.criterionL1 = torch.nn.L1Loss()
        if not opt.no_vgg_loss:
            self.criterionVGG = losses.VGGLoss(self.opt.gpu_ids[0])
        if not opt.no_maskedL1_loss:
            self.criterionMaskedL1 = losses.MaskedL1Loss()

        self.loss_names = ['G_VGG', 'G_GAN', 'G_GAN_Feat', 'G_MaskedL1', 'D_real', 'D_generated']
        if not self.opt.no_flownetwork:
            self.loss_names += ['G_VGG_w', 'G_MaskedL1_w', 'G_L1_mask']
        if not opt.no_mouth_D:
            self.loss_names += ['Gm_GAN', 'Gm_GAN_Feat', 'Dm_real', 'Dm_generated']

        beta1, beta2 = opt.beta1, opt.beta2
        lr = opt.lr
        if opt.no_TTUR:
            D_lr = lr
        else:
            D_lr = lr * 2

        # initialize optimizers
        params = list(self.netD.parameters())
        if not opt.no_mouth_D:
            params += list(self.netDm.parameters())
        self.optimizer_D = torch.optim.Adam(params, lr=D_lr, betas=(beta1, beta2))

    def compute_D_losses(self, netD, real_A, real_B, generated_B):
        # Input
        if real_A is not None:
            real_AB = torch.cat((real_A, real_B), dim=1)
            generated_AB = torch.cat((real_A, generated_B), dim=1)
        else:
            real_AB = real_B
            generated_AB = generated_B
        # D losses
        pred_real = netD.forward(real_AB)
        pred_generated = netD.forward(generated_AB.detach())
        loss_D_real = self.criterionGAN(pred_real, True, for_discriminator=True)
        loss_D_generated = self.criterionGAN(pred_generated, False, for_discriminator=True)
        # G losses
        pred_generated = netD.forward(generated_AB)
        loss_G_GAN = self.criterionGAN(pred_generated, True, for_discriminator=False)
        loss_G_GAN_Feat = self.FM_loss(pred_real, pred_generated)
        return loss_D_real, loss_D_generated, loss_G_GAN, loss_G_GAN_Feat

    def FM_loss(self, pred_real, pred_generated):
        loss_G_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(min(len(pred_generated), self.opt.num_D)):
                for j in range(len(pred_generated[i])-1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                        self.criterionL1(pred_generated[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat
        else:
            loss_G_GAN_Feat = torch.zeros(self.bs, 1).cuda()
        return loss_G_GAN_Feat

    def forward(self, real_B, generated_B, warped_B, real_A, masks_B, masks, audio_feats, mouth_centers):
        lambda_feat = self.opt.lambda_feat
        lambda_vgg = self.opt.lambda_vgg
        lambda_maskedL1 = self.opt.lambda_maskedL1
        lambda_mask = self.opt.lambda_mask
        
        self.bs , _, self.height, self.width = real_B.size()

        # VGG loss
        loss_G_VGG = (self.criterionVGG(generated_B, real_B) * lambda_vgg) if not self.opt.no_vgg_loss else torch.zeros(self.bs, 1).cuda()
     
        # GAN and FM loss for Generator
        loss_D_real, loss_D_generated, loss_G_GAN, loss_G_GAN_Feat = self.compute_D_losses(self.netD, real_A, real_B, generated_B)
      
        loss_G_MaskedL1 = torch.zeros(self.bs, 1).cuda()
        if not self.opt.no_maskedL1_loss:
            loss_G_MaskedL1 = self.criterionMaskedL1(generated_B, real_B, real_A) * lambda_maskedL1

        loss_list = [loss_G_VGG, loss_G_GAN, loss_G_GAN_Feat, loss_G_MaskedL1, loss_D_real, loss_D_generated]
        
        # Warp Losses
        if not self.opt.no_flownetwork:
            loss_G_VGG_w = (self.criterionVGG(warped_B, real_B) * lambda_vgg) if not self.opt.no_vgg_loss else torch.zeros(self.bs, 1).cuda()
            loss_G_MaskedL1_w = torch.zeros(self.bs, 1).cuda()
            if not self.opt.no_maskedL1_loss:
                loss_G_MaskedL1_w = self.criterionMaskedL1(warped_B, real_B, real_A) * lambda_maskedL1
            loss_G_L1_mask = self.criterionL1(masks, masks_B.detach()) * lambda_mask
            loss_list += [loss_G_VGG_w, loss_G_MaskedL1_w, loss_G_L1_mask]

        # Mouth discriminator losses
        if not self.opt.no_mouth_D:
            # Extract mouth region around the center
            real_B_mouth, generated_B_mouth = util.get_ROI([real_B, generated_B], mouth_centers, self.opt)

            if not self.opt.no_audio_input:
                # Repeat audio features spatially for conditional input to mouth discriminator
                real_A_mouth = audio_feats[:, -self.opt.naf:].view(audio_feats.size(0), self.opt.naf, 1, 1)
                real_A_mouth = real_A_mouth.repeat(1, 1, real_B_mouth.size(2), real_B_mouth.size(3))
            else:
                real_A_mouth = None

            # Losses for mouth discriminator
            loss_Dm_real, loss_Dm_generated, loss_Gm_GAN, loss_Gm_GAN_Feat = self.compute_D_losses(self.netDm, real_A_mouth, real_B_mouth, generated_B_mouth)
            mouth_weight = 1
            loss_Gm_GAN *= mouth_weight
            loss_Gm_GAN_Feat *= mouth_weight
            loss_list += [loss_Gm_GAN, loss_Gm_GAN_Feat, loss_Dm_real, loss_Dm_generated]

        loss_list = [loss.unsqueeze(0) for loss in loss_list]
        return loss_list

    def save(self, label):
        self.save_network(self.netD, 'D', label, self.gpu_ids)
        if not self.opt.no_mouth_D:
            self.save_network(self.netDm, 'Dm', label, self.gpu_ids)

    def update_learning_rate(self, epoch):
        if self.opt.niter_decay > 0:
            lr = self.opt.lr * (1 - (epoch - self.opt.niter) / self.opt.niter_decay)
            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = lr
            print('Update learning rate for D: %f -> %f' % (self.old_lr, lr))
            self.old_lr = lr


class headGANModelG(BaseModel):
    def name(self):
        return 'headGANModelG'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.n_frames_G = opt.n_frames_G
        self.output_nc = opt.output_nc
        self.input_nc = opt.input_nc

        self.netG = networks.define_G(opt)

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            self.load_network(self.netG, 'G', opt.which_epoch, opt.load_pretrain)
            print('---------- Generator loaded -------------')
        else:
            print('---------- Generator initialized -------------')

        # Otimizer for G
        if self.isTrain:
            self.old_lr = opt.lr
            
            # initialize optimizer G
            paramsG = list(self.netG.parameters())
            beta1, beta2 = opt.beta1, opt.beta2
            lr = opt.lr
            if opt.no_TTUR:
                G_lr = lr
            else:
                G_lr = lr / 2
            self.optimizer_G = torch.optim.Adam(paramsG, lr=G_lr, betas=(beta1, beta2))

    def forward(self, input, ref_input, audio_feats):
        ret = self.netG.forward(input, ref_input, audio_feats)
        return ret

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)

    def update_learning_rate(self, epoch):
        if self.opt.niter_decay > 0:
            lr = self.opt.lr * (1 - (epoch - self.opt.niter) / self.opt.niter_decay)
            for param_group in self.optimizer_G.param_groups:
                param_group['lr'] = lr
            print('Update learning rate for G: %f -> %f' % (self.old_lr, lr))
            self.old_lr = lr


def create_model_G(opt):
    modelG = headGANModelG()
    modelG.initialize(opt)
    modelG = nn.DataParallel(modelG, device_ids=opt.gpu_ids)
    return modelG
