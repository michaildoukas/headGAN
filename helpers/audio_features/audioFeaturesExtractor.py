# Notice: The following code has been taken from https://github.com/tyiannak/pyAudioAnalysis and modified. 
# The use of this code is subject to the terms and conditions set forth by the original code's license.

import os
import time
import glob
import numpy as np
import matplotlib.pyplot as plt
from helpers.audio_features.audioBasicIO import read_audio_file, stereo_to_mono
from helpers.audio_features.ShortTermFeatures import feature_extraction

def mid_feature_extraction(signal, sampling_rate, mid_window, mid_step, short_window, short_step):
    short_features, short_feature_names = feature_extraction(signal, sampling_rate, short_window, short_step)
    n_stats = 2
    n_feats = len(short_features)
    mid_window_ratio = int(round(mid_window / short_step))
    mt_step_ratio = int(round(mid_step / short_step))

    mid_features, mid_feature_names = [], []
    for i in range(n_stats * n_feats):
        mid_features.append([])
        mid_feature_names.append("")

    for i in range(n_feats):
        cur_position = 0
        num_short_features = len(short_features[i])
        mid_feature_names[i] = short_feature_names[i] + "_" + "mean"
        mid_feature_names[i + n_feats] = short_feature_names[i] + "_" + "std"

        while cur_position < num_short_features:
            end = cur_position + mid_window_ratio
            if end > num_short_features:
                end = num_short_features
            cur_st_feats = short_features[i][cur_position:end]

            mid_features[i].append(np.mean(cur_st_feats))
            mid_features[i + n_feats].append(np.std(cur_st_feats))
            cur_position += mt_step_ratio
    return np.array(mid_features), short_features, mid_feature_names

def get_mid_features(file_path, mid_window, mid_step, short_window, short_step):
    sampling_rate, signal = read_audio_file(file_path)
    signal = stereo_to_mono(signal)
    mid_features, _, _ = \
        mid_feature_extraction(signal, sampling_rate,
                               round(sampling_rate * mid_window),
                               round(sampling_rate * mid_step),
                               round(sampling_rate * short_window),
                               round(sampling_rate * short_step))
    return list(mid_features.T)
