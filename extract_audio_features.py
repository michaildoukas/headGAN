import os
import argparse
import numpy as np
import collections
import cv2
import pandas as pd
from shutil import copyfile
from helpers.audio_features.audioFeaturesExtractor import get_mid_features
from helpers.audio_features.deepspeechFeaturesExtractor import get_logits
from util.util import *


def save_audio_features(audio_features, name, split, args):
    save_results = True
    if split == 'train':
        n_parts = len(audio_features) // args.train_seq_length
        n_audio_features = n_parts  * args.train_seq_length
    else:
        n_audio_features = len(audio_features)
    for i in range(n_audio_features):
        n_frame = "{:06d}".format(i)
        part = "_{:06d}".format(i // args.train_seq_length) if split == 'train' else ""
        save_dir = os.path.join(args.dataset_path, split, 'audio_features', name + part)

        # Check if corresponding images directory exists before saving the audio features
        # If it doesn't, don't save results
        if os.path.exists(save_dir.replace('audio_features', 'images')):
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            np.savetxt(os.path.join(save_dir, n_frame + '.txt'), audio_features[i])
        else:
            save_results = False
            break

    # If test split, save audio .wav file as well
    if split == 'test' and save_results:
        save_dir = os.path.join(args.dataset_path, split, 'audio')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, name + '.wav')
        copyfile('temp_audio.wav', save_path)


def is_path_processed(name, split, args):
    first_part = '_000000' if split == 'train' else ''
    path = os.path.join(args.dataset_path, split, 'audio_features', name + first_part)
    return os.path.isdir(path)


def get_split_dict(csv_file, args):
    if csv_file and os.path.exists(csv_file):
        csv = pd.read_csv(csv_file)
        names = list(csv['filename'])
        names = [os.path.splitext(name)[0] for name in names]
        splits = list(csv['split'])
        split_dict = dict(zip(names, splits))
        return split_dict, set(val for val in split_dict.values())
    else:
        print('No metadata file found. All samples will be saved in the %s split.' % args.split)
        return None, set([args.split])


def get_video_paths_dict(dir):
    # Returns dict: {video_name: path, ...}
    if os.path.exists(dir) and is_video_file(dir):
        # If path to single .mp4 file was given directly.
        # If '_' in file name remove it.
        video_files = {os.path.splitext(os.path.basename(dir))[0].replace('_', '') : dir}
    else:
        video_files = {}
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_video_file(fname):
                    path = os.path.join(root, fname)
                    video_name = os.path.splitext(fname)[0]
                    if video_name not in video_files:
                        video_files[video_name] = path
    return collections.OrderedDict(sorted(video_files.items()))


def get_video_info(mp4_path):
    reader = cv2.VideoCapture(mp4_path)
    fps = reader.get(cv2.CAP_PROP_FPS)
    n_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    reader.release()
    return fps, n_frames


def fix_audio_features_size(audio_feats, n_frames):
    # Check that we have one feature vector per frame
    if len(audio_feats) < n_frames:
        a = audio_feats[-1:] * (n_frames - len(audio_feats))
        audio_feats.extend(audio_feats[-1:] * (n_frames - len(audio_feats)))
    if len(audio_feats) > n_frames:
        audio_feats = audio_feats[:n_frames]
    return audio_feats


def extract_audio_features(mp4_path, audio_save_path='temp_audio.wav'):
    print('Extracting audio features')
    # Get video frame rate.
    fps, n_frames = get_video_info(mp4_path)

    # Use ffmpeg to get audio data in .wav format.
    ffmpeg_call = 'ffmpeg -y -i ' + mp4_path.replace(' ', '\ ') + ' ' + audio_save_path + ' > /dev/null 2>&1'
    os.system(ffmpeg_call)

    # Extract lower level audio features
    audio_feats = get_mid_features(audio_save_path, 8/fps, 1/fps, 8/(fps * 16), 1/(fps * 16))
    audio_feats = fix_audio_features_size(audio_feats, n_frames)

    # Extract deepspeech character one-hot vectors (higher level features)
    deepspeech_feats = get_logits(audio_save_path, fps, window=8)

    if len(deepspeech_feats) == 1:
        # If no deepspeech features detected, replicate empty token.
        deepspeech_feats = deepspeech_feats * len(audio_feats)
    deepspeech_feats = fix_audio_features_size(deepspeech_feats, n_frames)

    # Concatenate features
    feats = [np.concatenate((af, df)) for af, df in zip(audio_feats, deepspeech_feats)]
    return feats


def main():
    print('---- Extract audio features from .mp4 files ---- \n')
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--original_files_path', type=str, required=True, 
                        help='Path to video root directory.')
    parser.add_argument('--dataset_path', type=str, default='datasets/voxceleb', 
                        help='Path to save dataset.')
    parser.add_argument('--metadata_path', type=str, default=None, 
                        help='Path to metadata (train/test split information).')
    parser.add_argument('--train_seq_length', default=50, type=int, help='The number of frames for each training sequence.')
    parser.add_argument('--split', default='train', type=str, help='The default split for data if no metadata file is provided. [train|test]')
    args = parser.parse_args()

    # Read metadata files to create data split
    split_dict, splits = get_split_dict(args.metadata_path, args)

    if not os.path.exists(args.dataset_path):
        os.makedirs(args.dataset_path)

    # Store video paths in dictionary.
    mp4_paths_dict = get_video_paths_dict(args.original_files_path)
    n_mp4s = len(mp4_paths_dict)
    print('Number of videos to process: %d \n' % n_mp4s)

    # Run audio feature extraction.
    n_completed = 0
    for name, mp4_path in mp4_paths_dict.items():
        n_completed += 1
        split = split_dict[name] if split_dict else args.split
        if not is_path_processed(name, split, args):
            # Extract features
            feats = extract_audio_features(mp4_path)

            # Save features
            save_audio_features(feats, name, split, args)
            
            os.remove('temp_audio.wav')
            print('(%d/%d) %s (%s file) [SUCCESS]' % (n_completed, n_mp4s, mp4_path, split))
        else:
            print('(%d/%d) %s (%s file) already processed!' % (n_completed, n_mp4s, mp4_path, split))

if __name__ == "__main__":
    main()
