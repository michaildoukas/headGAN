import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from tqdm import tqdm
from facenet_pytorch import MTCNN, extract_face
from PIL import Image

from models.headGAN import headGANModelG 
from options.reenact_options import ReenactOptions
from helpers.audio_features.audioFeaturesExtractor import get_mid_features
from helpers.audio_features.deepspeechFeaturesExtractor import get_logits
from helpers.reconstruction import NMFCRenderer
from detect_faces import get_faces
from extract_audio_features import extract_audio_features
from util.util import *
from dataloader.base_dataset import get_params, get_transform

opt = ReenactOptions().parse(save=False)

if not opt.no_crop:
    detector = MTCNN(opt.cropped_image_size, opt.margin, post_process=False, device='cuda:' + str(opt.gpu_id))

renderer = NMFCRenderer(opt)

modelG = headGANModelG()
modelG.initialize(opt)
modelG = nn.DataParallel(modelG, device_ids=opt.gpu_ids)
modelG.eval()

if is_video_file(opt.driving_path):
    driving_frames, fps = read_mp4(opt.driving_path, opt.n_frames_G - 1)
else:
    print('%s is not a video. Exit' % opt.driving_path)
    exit(0)
        
if is_image_file(opt.reference_path):
    reference_image = read_image(opt.reference_path)
else:
    print('%s is not an image. Exit' % opt.reference_path)
    exit(0)

driving_name = os.path.splitext(os.path.basename(opt.driving_path))[0]
reference_name = os.path.splitext(os.path.basename(opt.reference_path))[0]
save_name = driving_name + '_' + reference_name
save_dir = os.path.join(opt.results_dir, opt.name, opt.which_epoch, 'reenact', save_name)
mkdir(save_dir)

# Detect faces
if not opt.no_crop:
    driving_stat, driving_frames = get_faces(detector, driving_frames, opt)
    if driving_stat:
        driving_frames = tensor2npimage(driving_frames)
    else:
        print('%s Face detection failed. Exit' % opt.driving_path)
        exit(0)
    reference_stat, reference_image = get_faces(detector, reference_image, opt)
    if reference_stat:
        reference_image = tensor2npimage(reference_image)
    else:
        print('%s Face detection failed. Exit' % opt.reference_path)
        exit(0)
else:
    reference_image = [cv2.resize(make_image_square(reference_image[0]), (opt.crop_size, opt.crop_size))]
    driving_frames = [cv2.resize(make_image_square(driving_frame), (opt.crop_size, opt.crop_size))
                      for driving_frame in driving_frames]
    
if not opt.no_audio_input:
    # Extract audio features
    audio_save_path = os.path.join(save_dir, 'audio.wav')
    audio_features = extract_audio_features(opt.driving_path, audio_save_path)

# Run face reconstruction for reference image
ref_success, ref_cam_params, ref_id_params, ref_exp_params, _ = renderer.reconstruct(reference_image)
if not ref_success:
    print('%s Face reconstruction failed. Exit' % opt.reference_path)
    exit(0)
reference_nmfc = renderer.computeNMFCs(ref_cam_params, ref_id_params, ref_exp_params, return_RGB=True)

# Run face reconstruction for driving video
success, cam_params, _, exp_params, _  = renderer.reconstruct(driving_frames)
if not success:
    print('%s Face reconstruction failed. Exit' % opt.driving_path)
    exit(0)

# Adapt driving camera parameters
cam_params = adapt_cam_params(cam_params, ref_cam_params, opt)

# Use the reference identity parameters
id_params = ref_id_params * len(exp_params)

# Render driving nmfcs
nmfcs = renderer.computeNMFCs(cam_params, id_params, exp_params, return_RGB=True)
renderer.clear()

height, width = reference_image[0].shape[:2]
params = get_params(opt, (width, height))
transform_nmfc = get_transform(opt, params, normalize=False)
transform_rgb = get_transform(opt, params)

ref_nmfc = transform_nmfc(Image.fromarray(reference_nmfc[0]))
ref_input_A = ref_nmfc.view(opt.batch_size, opt.input_nc, height, width).cuda(opt.gpu_ids[0])
ref_rgb = transform_rgb(Image.fromarray(reference_image[0]))
ref_input_B = ref_rgb.view(opt.batch_size, opt.output_nc, height, width).cuda(opt.gpu_ids[0])

driving_nmfc = torch.stack([transform_nmfc(Image.fromarray(nmfc)) for nmfc in nmfcs[:opt.n_frames_G]], dim=0)

print('Running generative network')
result_frames = []
with torch.no_grad():
    for i, nmfc in enumerate(tqdm(nmfcs[opt.n_frames_G-1:])):
        driving_nmfc = torch.cat([driving_nmfc[1:,:,:,:], transform_nmfc(Image.fromarray(nmfc)).unsqueeze(0)], dim=0)
        input_A = driving_nmfc.view(opt.batch_size, -1, opt.input_nc, height, width).cuda(opt.gpu_ids[0])

        if not opt.no_audio_input:
            audio_feats = torch.tensor(audio_features[i]).float().view(opt.batch_size, -1).cuda(opt.gpu_ids[0])
        else:
            audio_feats = None

        input = input_A.view(opt.batch_size, -1, height, width)
        ref_input = torch.cat([ref_input_A, ref_input_B], dim=1)

        generated, warped, _ = modelG(input, ref_input, audio_feats)

        generated = tensor2im(generated[0])
        result_list = [reference_image[0], driving_frames[i + opt.n_frames_G - 1], generated]

        mkdirs([os.path.join(save_dir, 'driving'), 
                os.path.join(save_dir, 'generated')])

        save_image(driving_frames[i + opt.n_frames_G - 1], os.path.join(save_dir, 'driving', str(i).zfill(6) + '.png'))
        save_image(generated, os.path.join(save_dir, 'generated', str(i).zfill(6) + '.png'))

        if opt.show_warped:
            warped = tensor2im(warped[0])
            result_list += [warped]
            mkdir(os.path.join(save_dir, 'warped'))
            save_image(warped, os.path.join(save_dir, 'warped', str(i).zfill(6) + '.png'))

        result_frame = np.concatenate(result_list, axis=1)
        result_frames.append(result_frame)

video_save_path = os.path.join(save_dir, 'video.mp4')
save_video(result_frames, video_save_path, fps)

if not opt.no_audio_input:
    # Add audio to generated video
    save_path = os.path.join(save_dir, 'video+audio.mp4')
    call = 'ffmpeg -y -i ' + video_save_path + ' -i ' + audio_save_path + ' -c:v copy -c:a aac ' + save_path + ' > /dev/null 2>&1'
    os.system(call)
    print("Video saved to %s" % save_path)
else:
    print("Video saved to %s" % video_save_path)