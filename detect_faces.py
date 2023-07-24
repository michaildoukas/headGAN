import os
import cv2
import numpy as np
import pandas as pd
import scipy.signal
from PIL import Image
import torch
import argparse
from facenet_pytorch import MTCNN, extract_face
import collections
from tqdm import tqdm
from util.util import *


def save_images(images, name, split, args):
    print('Saving images')
    lim = len(images) - len(images) % args.train_seq_length if split == 'train' else len(images)
    for i in tqdm(range(lim)):
        n_frame = "{:06d}".format(i)
        part = "_{:06d}".format((i) // args.train_seq_length) if split == 'train' else ""
        save_dir = os.path.join(args.dataset_path, split, 'images', name + part)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_image(images[i], os.path.join(save_dir, n_frame + '.png'))


def get_split_dict(csv_file, args):
    if csv_file and os.path.exists(csv_file):
        csv = pd.read_csv(csv_file)
        names = list(csv['filename'])
        names = [os.path.splitext(name)[0] for name in names]
        splits = list(csv['split'])
        split_dict = dict(zip(names, splits))
        return split_dict
    else:
        print('No metadata file provided. All samples will be saved in the %s split.' % args.split)
        return None


def get_vid_paths_dict(dir):
    # Returns dict: {vid_name: path, ...}
    if os.path.exists(dir) and is_video_file(dir):
        # If path to single .mp4 file was given directly.
        # If '_' in file name remove it, since it causes problems.
        vid_files = {os.path.splitext(os.path.basename(dir))[0].replace('_', '') : dir}
    else:
        vid_files = {}
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_video_file(fname):
                    path = os.path.join(root, fname)
                    name = os.path.splitext(fname)[0]
                    if name not in vid_files:
                        vid_files[name] = path
    return collections.OrderedDict(sorted(vid_files.items()))


def is_vid_path_processed(name, split, args):
    first_part = '_000000' if split == 'train' else ''
    path = os.path.join(args.dataset_path, split, 'images', name + first_part)
    return os.path.isdir(path)


def check_boxes(boxes, img_size, args):
    # Check if there are None boxes.
    for i in range(len(boxes)):
        if boxes[i] is None:
            boxes[i] = next((item for item in boxes[i+1:] if item is not None), boxes[i-1])
    if boxes[0] is None:
        print('Not enough boxes detected.')
        return False, [None]
    boxes = [box[0] for box in boxes]
    # Check if detected faces are very far from each other. Check distances between all boxes.
    maxim_dst = 0
    for i in range(len(boxes)-1):
        for j in range(len(boxes)-1):
            dst = max(abs(boxes[i] - boxes[j])) / img_size
            if dst > maxim_dst:
                maxim_dst = dst
    if maxim_dst > args.dst_threshold:
         print('L_inf distance between bounding boxes %.4f larger than threshold' % maxim_dst)
         return False, [None]
    # Get average box
    avg_box = np.median(boxes, axis=0)
    # Make boxes square.
    offset_w = avg_box[2] - avg_box[0]
    offset_h = avg_box[3] - avg_box[1]
    offset_dif = (offset_h - offset_w) / 2
    # width
    avg_box[0] = avg_box[2] - offset_w - offset_dif
    avg_box[2] = avg_box[2] + offset_dif
    # height - center a bit lower
    avg_box[3] = avg_box[3] + args.height_recentre * offset_h
    avg_box[1] = avg_box[3] - offset_h
    return True, avg_box


def get_faces(detector, images, args):
    ret_faces = []
    all_boxes = []
    avg_box = None
    all_imgs = []
    # Get bounding boxes
    print('Getting bounding boxes')
    for lb in tqdm(np.arange(0, len(images), args.mtcnn_batch_size)):
        imgs_pil = [Image.fromarray(image) for image in images[lb:lb+args.mtcnn_batch_size]]
        boxes, _, _ = detector.detect(imgs_pil, landmarks=True)
        all_boxes.extend(boxes)
        all_imgs.extend(imgs_pil)
    # Check if boxes are fine, do temporal smoothing, return average box.
    img_size = (all_imgs[0].size[0] + all_imgs[0].size[1]) / 2
    stat, avg_box = check_boxes(all_boxes, img_size, args)
    # Crop face regions.
    if stat:
        print('Extracting faces')
        for img in tqdm(all_imgs, total=len(all_imgs)):
            face = extract_face(img, avg_box, args.cropped_image_size, args.margin)
            ret_faces.append(face)
    return stat, ret_faces


def detect_and_save_faces(detector, name, file_path, split, args):
    if is_video_file(file_path):
        images, fps = read_mp4(file_path, args.n_replicate_first)
    else:
        images = read_image(file_path, args)
    if not args.no_crop:
        stat, face_images = get_faces(detector, images, args)
    else:
        stat, face_images = True, images
    if stat:
        save_images(tensor2npimage(face_images), name, split, args)
    return stat


def main():
    print('-------------- Face detector -------------- \n')
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--original_files_path', type=str, required=True,
                        help='Path to videos root directory')
    parser.add_argument('--dataset_path', type=str, default='datasets/voxceleb', 
                        help='Path to save dataset.')
    parser.add_argument('--gpu_id', type=str, default='0', 
                        help='Negative value to use CPU, or greater equal than zero for GPU id.')
    parser.add_argument('--metadata_path', type=str, default=None,
                        help='Path to metadata (train/test split information).')
    parser.add_argument('--no_crop', action='store_true', help='Save the frames without face detection and cropping.')
    parser.add_argument('--mtcnn_batch_size', default=1, type=int, help='The number of frames for face detection.')
    parser.add_argument('--cropped_image_size', default=256, type=int, help='The size of frames after cropping the face.')
    parser.add_argument('--margin', default=100, type=int, help='.')
    parser.add_argument('--dst_threshold', default=0.45, type=float, help='Max L_inf distance between any bounding boxes in a video. (normalised by image size: (h+w)/2)')
    parser.add_argument('--height_recentre', default=0.0, type=float, help='The amount of re-centring bounding boxes lower on the face.')
    parser.add_argument('--train_seq_length', default=50, type=int, help='The number of frames for each training sub-sequence.')
    parser.add_argument('--split', default='train', choices=['train', 'test'], type=str, help='The split for data [train|test]')
    parser.add_argument('--n_replicate_first', default=0, type=int, help='How many times to replicate and append the first frame to the beginning of the video.')

    args = parser.parse_args()

    # Figure out the device
    gpu_id = int(args.gpu_id)
    if gpu_id < 0:
        device = 'cpu'
    elif torch.cuda.is_available():
        if gpu_id >= torch.cuda.device_count():
            device = 'cuda:0'
        else:
            device = 'cuda:' + str(gpu_id)
    else:
        print('GPU device not available. Exit')
        exit(0)

    # Read metadata file to create data split
    split_dict = get_split_dict(args.metadata_path, args)

    # Store file paths in dictionary.
    files_paths_dict = get_vid_paths_dict(args.original_files_path)
    n_files = len(files_paths_dict)
    print('Number of files to process: %d \n' % n_files)

    # Initialize the MTCNN face  detector.
    detector = MTCNN(image_size=args.cropped_image_size, margin=args.margin, post_process=False, device=device)

    # Run detection
    n_completed = 0
    for name, path in files_paths_dict.items():
        n_completed += 1
        split = split_dict[name] if split_dict else args.split
        if not is_vid_path_processed(name, split, args):
            success = detect_and_save_faces(detector, name, path, split, args)
            if success:
                print('(%d/%d) %s (%s file) [SUCCESS]' % (n_completed, n_files, path, split))
            else:
                print('(%d/%d) %s (%s file) [FAILED]' % (n_completed, n_files, path, split))
        else:
            print('(%d/%d) %s (%s file) already processed!' % (n_completed, n_files, path, split))

if __name__ == "__main__":
    main()
