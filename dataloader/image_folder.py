import os
import random
from PIL import Image
import torch.utils.data as data
from util.util import *

def make_video_dataset(dir, target_name, max_seqs_per_identity=None):
    images = []
    if dir:
        # Gather sequences
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        fnames = sorted(os.walk(dir))
        for fname in sorted(fnames):
            paths = []
            root = fname[0]
            for f in sorted(fname[2]):
                names = os.path.basename(root).split('_')
                target = names[0]
                if is_image_file(f):
                    if (target_name is None or target_name == target):
                        paths.append(os.path.join(root, f))
            if len(paths) > 0:
                images.append(paths)
        # Find minimun sequence length and reduce all sequences to that.
        # Only for training, in order to be able to form batches.
        if max_seqs_per_identity is not None:
            min_len = float("inf")
            for img_dir in images:
                min_len = min(min_len, len(img_dir))
            for i in range(len(images)):
                images[i] = images[i][:min_len]
            # Keep max_seqs_per_identity for each identity
            trimmed_images = []
            prev_name = None
            temp_seqs = []
            for i in range(len(images)):
                folder = images[i][0].split('/')[-2].split('_')
                name = folder[0]
                if prev_name is None:
                    prev_name = name
                if name == prev_name:
                    temp_seqs.append(i)
                if name != prev_name or i == len(images)-1:
                    if len(temp_seqs) > max_seqs_per_identity:
                        identity_seqs = sorted(temp_seqs)[:max_seqs_per_identity]
                    else:
                        identity_seqs = sorted(temp_seqs)
                    trimmed_images.extend([images[j] for j in identity_seqs])
                    temp_seqs = [i]
                prev_name = name
            images = trimmed_images
    return images

def assert_valid_pairs(A_paths, B_paths):
    assert len(A_paths) > 0 and len(B_paths) > 0, 'No sequences found.'
    assert len(A_paths) == len(B_paths), 'Number of NMFC sequences different than RGB sequences.'
    for i in range(len(A_paths)):
        if abs(len(A_paths[i]) - len(B_paths[i])) <= 3:
            min_len = min(len(A_paths[i]), len(B_paths[i]))
            A_paths[i] = A_paths[i][:min_len]
            B_paths[i] = B_paths[i][:min_len]
        assert len(A_paths[i]) == len(B_paths[i]), 'Number of NMFC frames in sequence different than corresponding RGB frames.'
