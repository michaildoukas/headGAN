import importlib
import os
import torch
import numpy as np
from PIL import Image
import torchvision
import cv2
from tqdm import tqdm

VID_EXTENSIONS = ['.mp4']

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.pgm', '.PGM', '.png', '.PNG', 
                  '.ppm', '.PPM', '.bmp', '.BMP', '.tiff', '.txt', '.json']

def is_video_file(filename):
    return any(filename.endswith(extension) for extension in VID_EXTENSIONS)

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def read_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return [image]

def read_mp4(mp4_path, n_replicate_first):
    reader = cv2.VideoCapture(mp4_path)
    fps = reader.get(cv2.CAP_PROP_FPS)
    images = []
    n_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    print('Reading %s' % mp4_path)
    for i in tqdm(range(n_frames)):
        _, image = reader.read()
        if image is None:
            break
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
    reader.release()
    if n_replicate_first > 0:
        pad = [images[0]] * n_replicate_first
        pad.extend(images)
        images = pad
    return images, fps

def seconds_to_hours_mins(t_sec):
    t_mins = t_sec //  60
    hours = t_mins // 60
    mins = t_mins - 60 * hours
    return hours, mins

def prepare_input(input_A, ref_input_A, ref_input_B):
    N, n_frames_G, channels, height, width = input_A.size()
    input = input_A.view(N, n_frames_G * channels, height, width)
    ref_input = torch.cat([ref_input_A, ref_input_B], dim=1)
    return input, ref_input

# get temporally subsampled frames for real/fake sequences
def get_skipped_frames(B_all, B, t_scales, n_frames_D):
    B_all = torch.cat([B_all.detach(), B], dim=1) if B_all is not None else B
    B_skipped = [None] * t_scales
    for s in range(t_scales):
        n_frames_Ds = n_frames_D ** s
        span = n_frames_Ds * (n_frames_D-1)
        if B_all.size()[1] > span:
            B_skipped[s] = B_all[:, -span-1::n_frames_Ds].contiguous()
    max_prev_frames = n_frames_D ** (t_scales-1) * (n_frames_D-1)
    if B_all.size()[1] > max_prev_frames:
        B_all = B_all[:, -max_prev_frames:]
    return B_all, B_skipped

def tensor2npimage(image_tensor):
    # Input tesnor in range [0, 255]
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2npimage(image_tensor[i]))
        return image_numpy
    if torch.is_tensor(image_tensor):
        image_numpy = np.transpose(image_tensor.cpu().float().numpy(), (1, 2, 0))
    image_numpy = np.clip(image_numpy, 0, 255)
    return image_numpy.astype(np.uint8)

def tensor2im(image_tensor, imtype=np.uint8, normalize=True):
    # Input tesnor in range [0, 1] or [-1, 1]
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy

    if isinstance(image_tensor, torch.autograd.Variable):
        image_tensor = image_tensor.data
    if len(image_tensor.size()) == 4:
        image_tensor = image_tensor[0]
    image_numpy = image_tensor.cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1:
        image_numpy = np.tile(image_numpy, (1,1,3))
    return image_numpy.astype(imtype)

def tensor2flow(output, imtype=np.uint8):
    if isinstance(output, torch.autograd.Variable):
        output = output.data
    if len(output.size()) == 5:
        output = output[0, -1]
    if len(output.size()) == 4:
        output = output[0]
    output = output.cpu().float().numpy()
    output = np.transpose(output, (1, 2, 0))
    #mag = np.max(np.sqrt(output[:,:,0]**2 + output[:,:,1]**2))
    #print(mag)
    hsv = np.zeros((output.shape[0], output.shape[1], 3), dtype=np.uint8)
    hsv[:, :, 0] = 255
    hsv[:, :, 1] = 255
    mag, ang = cv2.cartToPolar(output[..., 0], output[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def save_video(frames, save_path, fps):
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h), True)
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame)
    writer.release()

def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def fit_ROI_in_frame(center, opt):
    center_w, center_h = center[0], center[1]
    center_h = torch.tensor(opt.ROI_size // 2, dtype=torch.int32).cuda(opt.gpu_ids[0]) if center_h < opt.ROI_size // 2 else center_h
    center_w = torch.tensor(opt.ROI_size // 2, dtype=torch.int32).cuda(opt.gpu_ids[0]) if center_w < opt.ROI_size // 2 else center_w
    center_h = torch.tensor(opt.crop_size - opt.ROI_size // 2, dtype=torch.int32).cuda(opt.gpu_ids[0]) if center_h > opt.crop_size - opt.ROI_size // 2 else center_h
    center_w = torch.tensor(opt.crop_size - opt.ROI_size // 2, dtype=torch.int32).cuda(opt.gpu_ids[0]) if center_w > opt.crop_size - opt.ROI_size // 2 else center_w
    return (center_w, center_h)

def crop_ROI(img, center, ROI_size):
    return img[..., center[1] - ROI_size // 2:center[1] + ROI_size // 2,
                    center[0] - ROI_size // 2:center[0] + ROI_size // 2]

def get_ROI(tensors, centers, opt):
    real_B, fake_B = tensors
    # Extract region of interest around the center.
    real_B_ROI = []
    fake_B_ROI = []
    for t in range(centers.shape[0]):
        center = fit_ROI_in_frame(centers[t], opt)
        real_B_ROI.append(crop_ROI(real_B[t], center, opt.ROI_size))
        fake_B_ROI.append(crop_ROI(fake_B[t], center, opt.ROI_size))
    real_B_ROI = torch.stack(real_B_ROI, dim=0)
    fake_B_ROI = torch.stack(fake_B_ROI, dim=0)
    return real_B_ROI, fake_B_ROI

def smoothen_signal(S, window_size=15):
    left_p = window_size // 2
    right_p =  window_size // 2 if window_size % 2 == 1 else window_size // 2 - 1
    window = np.ones(int(window_size))/float(window_size) # kernel-filter
    S = np.array(S)
    # Padding
    left_padding = np.stack([S[0]] * left_p, axis=0)
    right_padding = np.stack([S[-1]] * right_p, axis=0)
    S_padded = np.concatenate([left_padding, S, right_padding])
    if len(S_padded.shape) == 1:
        S = np.convolve(S_padded, window, 'valid')
    else:
        for coord in range(S_padded.shape[1]):
            S[:, coord] = np.convolve(S_padded[:, coord], window, 'valid')
    return S

def adapt_cam_params(s_cam_params, t_cam_params, args):
    cam_params = s_cam_params
    if not args.no_scale_or_translation_adaptation:
        mean_S_target = np.mean([params[0] for params in t_cam_params])
        mean_S_source = np.mean([params[0] for params in s_cam_params])
        S = [params[0] * (mean_S_target / mean_S_source)
             for params in s_cam_params]
        # Smoothen scale
        S = smoothen_signal(S)
        # Normalised Translation for source and target.
        nT_target = [params[2] / params[0] for params in t_cam_params]
        nT_source = [params[2] / params[0] for params in s_cam_params]
        cam_params = [(s, params[1], s * t) \
                      for s, params, t in zip(S, s_cam_params, nT_source)]
        if not args.no_translation_adaptation:
            mean_nT_target = np.mean(nT_target, axis=0)
            mean_nT_source = np.mean(nT_source, axis=0)
            if args.standardize:
                std_nT_target = np.std(nT_target, axis=0)
                std_nT_source = np.std(nT_source, axis=0)
                nT = [(t - mean_nT_source) * std_nT_target / std_nT_source \
                     + mean_nT_target for t in nT_source]
            else:
                nT = [t - mean_nT_source + mean_nT_target
                      for t in nT_source]
            # Smoothen translation
            nT = smoothen_signal(nT)
            cam_params = [(s, params[1], s * t) \
                          for s, params, t in zip(S, s_cam_params, nT)]
    return cam_params

def make_image_square(image):
    h, w = image.shape[:2]
    d = abs(h - w)
    if h > w:
        image = image[d // 2: d // 2 + w, :, :]
    else:
        image = image[:, d // 2: d // 2 + h, :]
    return image