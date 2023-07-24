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
from models.headGAN import headGANModelG, headGANModelD 
from options.train_options import TrainOptions
from util.visualizer import Visualizer
import util.util as util
import torchvision
from models.segmenter_pytorch.segmenter import Segmenter

torch.autograd.set_detect_anomaly(True)

opt = TrainOptions().parse()

visualizer = Visualizer(opt)
segmenter = Segmenter(opt.gpu_ids[0])

modelG = headGANModelG()
modelG.initialize(opt)
modelG = nn.DataParallel(modelG, device_ids=opt.gpu_ids)

modelD = headGANModelD()
modelD.initialize(opt)
modelD = nn.DataParallel(modelD, device_ids=opt.gpu_ids)

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
visualizer.vis_print('Number of identities in dataset %d' % data_loader.dataset.n_identities)
visualizer.vis_print('Number of sequences in dataset %d' % dataset_size)

iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')

if opt.continue_train:
    try:
        start_epoch, seen_seqs = np.loadtxt(iter_path , delimiter=',', dtype=int)
    except:
        start_epoch, seen_seqs = 1, 0

    if seen_seqs > 0:
        # initialize dataset again
        if opt.serial_batches:
            data_loader = CreateDataLoader(opt, seen_seqs)
            dataset = data_loader.load_data()
            dataset_size = len(data_loader)

    visualizer.vis_print('Resuming from epoch %d at iteration %d' % (start_epoch, seen_seqs))

    if start_epoch > opt.niter:
        modelG.module.update_learning_rate(start_epoch)
        modelD.module.update_learning_rate(start_epoch)
    if start_epoch >= opt.niter_start:
        data_loader.dataset.update_sequence_length((start_epoch - opt.niter_start + opt.niter_step) // opt.niter_step)
else:
    start_epoch, seen_seqs = 1, 0
    visualizer.vis_print('Initiating training.')

seen_seqs_start_time = None
n_steps_G = 0

for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    for idx, data in enumerate(dataset):
        # New batch of sequences
        if seen_seqs_start_time is None:
            seen_seqs_start_time = time.time()

        bs, n_frames_total, height, width = data['image'].size()
        n_frames_total = n_frames_total // opt.output_nc

        ref_input_A = Variable(data['ref_nmfc']).view(opt.batch_size, opt.input_nc, height, width).cuda(opt.gpu_ids[0])
        ref_input_B = Variable(data['ref_image']).view(opt.batch_size, opt.output_nc, height, width).cuda(opt.gpu_ids[0])

        ref_input_B_segmenter = Variable(data['ref_image_segmenter']).view(opt.batch_size, opt.output_nc, 512, 512).cuda(opt.gpu_ids[0])
        ref_masks_B = segmenter.get_masks(ref_input_B_segmenter, (height, width)).view(opt.batch_size, 1, height, width)

        # Go through sequences
        for i in range(0, n_frames_total-opt.n_frames_G+1):
            nmfc = Variable(data['nmfc'][:, i*opt.input_nc:(i+opt.n_frames_G)*opt.input_nc, ...])
            input_A = nmfc.view(opt.batch_size, opt.n_frames_G, opt.input_nc, height, width).cuda(opt.gpu_ids[0])

            image = Variable(data['image'][:, i*opt.output_nc:(i+opt.n_frames_G)*opt.output_nc, ...])
            input_B = image.view(opt.batch_size, opt.n_frames_G, opt.output_nc, height, width).cuda(opt.gpu_ids[0])

            image_segmenter = Variable(data['image_segmenter'][:, (i+opt.n_frames_G-1)*opt.output_nc:(i+opt.n_frames_G)*opt.output_nc, ...])
            image_segmenter = image_segmenter.view(opt.batch_size, opt.output_nc, 512, 512).cuda(opt.gpu_ids[0])

            masks_B = segmenter.get_masks(image_segmenter, (height, width)).view(opt.batch_size, 1, height, width)
            masks_union_B = segmenter.join_masks(masks_B, ref_masks_B)

            audio_feats = None
            if not opt.no_audio_input:
                audio_feats = Variable(data['audio_feats'][:, (i+opt.n_frames_G-1)*opt.naf:(i+opt.n_frames_G)*opt.naf])
                audio_feats = audio_feats.cuda(opt.gpu_ids[0])
                
            mouth_centers = Variable(data['mouth_centers'][:, i*2:(i+opt.n_frames_G)*2]).view(opt.batch_size, opt.n_frames_G, 2) if not opt.no_mouth_D else None

            input = input_A.view(opt.batch_size, -1, height, width)
            ref_input = torch.cat([ref_input_A, ref_input_B], dim=1)

            # Generator forward
            generated_B, feat_maps, warped, flows, masks = modelG(input, ref_input, audio_feats)

            real_A = input_A[:, opt.n_frames_G-1, :, :, :]
            real_B = input_B[:, opt.n_frames_G-1, :, :, :]

            mouth_centers = mouth_centers[:,opt.n_frames_G - 1,:] if not opt.no_mouth_D else None

            # Image (and Mouth) Discriminator forward
            losses = modelD(real_B, generated_B, warped, real_A, masks_union_B, masks, audio_feats, mouth_centers)

            losses = [torch.mean(loss) for loss in losses]
            loss_dict = dict(zip(modelD.module.loss_names, losses))

            # Losses
            loss_D = loss_dict['D_generated'] + loss_dict['D_real']
            loss_G = loss_dict['G_GAN'].clone()

            if not opt.no_ganFeat_loss:
                loss_G += loss_dict['G_GAN_Feat']

            if not opt.no_vgg_loss:
                loss_G += loss_dict['G_VGG']

            if not opt.no_maskedL1_loss:
                loss_G += loss_dict['G_MaskedL1']

            if not opt.no_flownetwork:
                if not opt.no_vgg_loss:
                    loss_G += loss_dict['G_VGG_w']

                if not opt.no_maskedL1_loss:
                    loss_G += loss_dict['G_MaskedL1_w']

                loss_G += loss_dict['G_L1_mask']

            if not opt.no_mouth_D:
                loss_G += loss_dict['Gm_GAN'] + loss_dict['Gm_GAN_Feat']
                loss_D += (loss_dict['Dm_generated'] + loss_dict['Dm_real']) * 0.5

            # Backward
            optimizer_G = modelG.module.optimizer_G
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()
            n_steps_G += 1
            if n_steps_G % opt.n_steps_update_D == 0:
                optimizer_D = modelD.module.optimizer_D
                optimizer_D.zero_grad()
                loss_D.backward()
                optimizer_D.step()

        # End of sequences
        seen_seqs += opt.batch_size

        # Print out errors
        if (seen_seqs / opt.batch_size) % opt.print_freq == 0:
            t = (time.time() - seen_seqs_start_time) / (opt.print_freq * opt.batch_size)
            seen_seqs_start_time = time.time()
            errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
            t_epoch = util.seconds_to_hours_mins(time.time() - epoch_start_time)
            visualizer.print_current_errors(epoch, seen_seqs, errors, t, t_epoch)
            visualizer.plot_current_errors(errors, seen_seqs)

        # Display output images
        if (seen_seqs / opt.batch_size) % opt.display_freq == 0:
            visual_dict = []
            for i in range(opt.batch_size):
                visual_dict += [('input_nmfc_image %d' % i, util.tensor2im(real_A[i, :opt.input_nc], normalize=False))]
                visual_dict += [('generated image %d' % i, util.tensor2im(generated_B[i])),
                                ('warped image %d' % i, util.tensor2im(warped[i])),
                                ('real image %d' % i, util.tensor2im(real_B[i]))]
                visual_dict += [('reference image %d' % i, util.tensor2im(ref_input_B[i]))]
                visual_dict += [('flow %d' % i, util.tensor2flow(flows[i]))]
                visual_dict += [('masks union %d' % i, util.tensor2im(masks_union_B[i], normalize=False))]
                visual_dict += [('masks %d' % i, util.tensor2im(masks[i], normalize=False))]

            visuals = OrderedDict(visual_dict)
            visualizer.display_current_results(visuals, epoch, seen_seqs)

        # Save latest model
        if (seen_seqs / opt.batch_size) % opt.save_latest_freq == 0:
            modelG.module.save('latest')
            modelD.module.save('latest')
            np.savetxt(iter_path, (epoch, seen_seqs), delimiter=',', fmt='%d')
            visualizer.vis_print('Saved the latest model (epoch %d, seen sequences %d)' % (epoch, seen_seqs))

        # Break when we have gone through the entire dataset.
        if seen_seqs >= dataset_size:
            break

    # End of epoch
    seen_seqs = 0
    visualizer.vis_print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    # Save model for this epoch, as latest
    modelG.module.save('latest')
    modelD.module.save('latest')
    np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')
    visualizer.vis_print('Saved the model at the end of epoch %d (as latest)' % (epoch))

    if epoch % opt.save_epoch_freq == 0:
        modelG.module.save(epoch)
        modelD.module.save(epoch)
        visualizer.vis_print('Saved the model at the end of epoch %d' % (epoch))

    # Linearly decay learning rate after certain iterations
    if epoch + 1 > opt.niter:
        modelG.module.update_learning_rate(epoch + 1)
        modelD.module.update_learning_rate(epoch + 1)

    # Grow training sequence length
    if epoch + 1 >= opt.niter_start:
        data_loader.dataset.update_sequence_length((epoch + 1 - opt.niter_start + opt.niter_step) // opt.niter_step)
