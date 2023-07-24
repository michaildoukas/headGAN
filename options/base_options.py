import sys
import argparse
import os
from util import util
import torch
import random
import numpy as np

class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # experiment specifics
        parser.add_argument('--name', type=str, default='headGAN', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='0,1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--results_dir', type=str, default='./results', help='saves results here.')
        parser.add_argument('--load_pretrain', type=str, default=None, help='Directory of model to load.')
        parser.add_argument('--model', type=str, default='headGAN', help='which model to use')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test')
        parser.add_argument('--target_name', type=str, default=None, help='If given, use only this target identity.')

        # input/output sizes
        parser.add_argument('--use_landmarks_input', action='store_true', help='Use facial landmark sketches instead of NMFC images as conditional input.')
        parser.add_argument('--resize', action='store_true', help='')
        parser.add_argument('--load_size', type=int, default=256, help='Scale images to this size. The final image will be cropped to --crop_size.')
        parser.add_argument('--crop_size', type=int, default=256, help='Crop to the width of crop_size (after initially scaling the images to load_size.)')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        parser.add_argument('--input_nc', type=int, default=3, help='# of input NMFC channels')
        parser.add_argument('--n_frames_G', type=int, default=3, help='number of frames to look in the past (T) + current -> T + 1')
        parser.add_argument('--batch_size', type=int, default=6, help='batch size')
        parser.add_argument('--n_frames_total', type=int, default=6, help='Starting number of frames to read for each sequence in the batch. Increases progressively.')
        parser.add_argument('--naf', type=int, default=300, help='number of audio features for each frame.')

        # for setting inputs
        parser.add_argument('--reference_frames_strategy', type=str, default='ref', help='[ref|previous]')
        parser.add_argument('--dataroot', type=str, default='datasets/voxceleb')
        parser.add_argument('--dataset_mode', type=str, default='video')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes sequences in sorted order for making batches, otherwise takes them randomly')
        parser.add_argument('--nThreads', default=8, type=int, help='# threads for loading data')

        # for generator
        parser.add_argument('--no_audio_input', action='store_true', help='')
        parser.add_argument('--no_pixelshuffle', action='store_true', help='Do not use PixelShuffle for upsampling.')
        parser.add_argument('--no_previousframesencoder', action='store_true', help='Do not condition synthesis of generator on previouly generated frames.')
        parser.add_argument('--no_flownetwork', action='store_true', help='')
        parser.add_argument('--netG', type=str, default='headGAN', help='selects model to use for netG.')
        parser.add_argument('--norm_G', type=str, default='spectralspadeinstance3x3', help='The type of adaptive normalization.')
        parser.add_argument('--norm_G_noadapt', type=str, default='spectralinstance', help='The type of non-adaptive normalization.')
        parser.add_argument('--ngf', type=int, default=32, help='# of gen filters in first conv layer')
        parser.add_argument('--down_size', type=int, choices=(8, 16, 32, 64), default=64, help="The size of the bottleneck spatial dimension when encoding-decoding.")
        parser.add_argument('--kernel_size', type=int, default=3, help='kernel size of encoder convolutions')
        parser.add_argument('--initial_kernel_size', type=int, default=7, help='kernel size of the first convolution of encoder')

        # initialization
        parser.add_argument('--init_type', type=str, default='xavier', help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_variance', type=float, default=0.02, help='variance of the initialization distribution')

        # visualization
        parser.add_argument('--display_winsize', type=int, default=400, help='display window size')
        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, unknown = parser.parse_known_args()
        opt = parser.parse_args()
        self.parser = parser
        return opt

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def option_file_path(self, opt, makedir=False):
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if makedir:
            util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt')
        return file_name

    def save_options(self, opt):
        file_name = self.option_file_path(opt, makedir=True)
        with open(file_name + '.txt', 'wt') as opt_file:
            for k, v in sorted(vars(opt).items()):
                comment = ''
                default = self.parser.get_default(k)
                if v != default:
                    comment = '\t[default: %s]' % str(default)
                opt_file.write('{:>25}: {:<30}{}\n'.format(str(k), str(v), comment))

    def parse(self, save=False):
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test
        opt.dataroot = opt.dataroot.replace('\ ', ' ')
        # Remove '_' from target_name.
        if opt.target_name:
            opt.target_name = opt.target_name.replace('_', '')
        self.print_options(opt)
        if opt.isTrain:
            self.save_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        self.opt = opt
        return self.opt
