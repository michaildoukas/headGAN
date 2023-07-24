import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import time
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from torch.autograd import Variable
from dataloader.data_loader import CreateDataLoader
from models.headGAN import headGANModelG 
from options.test_options import TestOptions
from util.visualizer import Visualizer
import util.util as util

opt = TestOptions().parse(save=False)

visualizer = Visualizer(opt)

modelG = headGANModelG()
modelG.initialize(opt)
modelG = nn.DataParallel(modelG, device_ids=opt.gpu_ids)
modelG.eval()

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)

print('Generating %d frames' % dataset_size)

save_dir = os.path.join(opt.results_dir, opt.name, opt.which_epoch, opt.phase)

for idx, data in enumerate(dataset):
    _, _, height, width = data['nmfc'].size()
    
    input_A = Variable(data['nmfc']).view(opt.batch_size, -1, opt.input_nc, height, width).cuda(opt.gpu_ids[0])
    image = Variable(data['image']).view(opt.batch_size, -1, opt.output_nc, height, width).cuda(opt.gpu_ids[0])

    if not opt.no_audio_input:
        audio_feats = Variable(data['audio_feats'][:, -opt.naf:]).cuda(opt.gpu_ids[0])
    else:
        audio_feats = None
    ref_input_A = Variable(data['ref_nmfc']).view(opt.batch_size, opt.input_nc, height, width).cuda(opt.gpu_ids[0])
    ref_input_B = Variable(data['ref_image']).view(opt.batch_size, opt.output_nc, height, width).cuda(opt.gpu_ids[0])
    img_path = data['paths']

    print('Processing NMFC image %s' % img_path[-1])
    
    input = input_A.view(opt.batch_size, -1, height, width)
    ref_input = torch.cat([ref_input_A, ref_input_B], dim=1)

    if opt.time_fwd_pass:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

    generated_B, warped_B, flow = modelG(input, ref_input, audio_feats)

    if opt.time_fwd_pass:
        end.record()
        # Wait for everything to finish running
        torch.cuda.synchronize()
        print('Forward pass time: %.6f' % start.elapsed_time(end))

    generated = util.tensor2im(generated_B.data[0])
    warped = util.tensor2im(warped_B.data[0])
    flow = util.tensor2flow(flow.data[0])
    nmfc = util.tensor2im(input_A[-1], normalize=False)
    image = util.tensor2im(image[-1])

    visual_list = [('image', image),
                   ('nmfc', nmfc),
                   ('generated', generated),
                   ('warped', warped),
                   ('flow', flow)]

    visuals = OrderedDict(visual_list)
    visualizer.save_images(save_dir, visuals, img_path[-1])
