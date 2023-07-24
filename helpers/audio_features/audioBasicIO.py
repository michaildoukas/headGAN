# Notice: The following code has been taken from https://github.com/tyiannak/pyAudioAnalysis and modified. 
# The use of this code is subject to the terms and conditions set forth by the original code's license.

import os
import glob
import aifc
import numpy
import shutil
import numpy as np
from pydub import AudioSegment


def read_audio_file(path):
    """
    This function returns a numpy array that stores the audio samples of a
    specified WAV of AIFF file
    """

    sampling_rate = 0
    signal = np.array([])
    extension = os.path.splitext(path)[1].lower()
    if extension in ['.aif', '.aiff']:
        sampling_rate, signal = read_aif(path)
    elif extension in [".mp3", ".wav", ".au", ".ogg"]:
        sampling_rate, signal = read_audio_generic(path)
    else:
        print("Error: unknown file type {extension}")

    if signal.ndim == 2 and signal.shape[1] == 1:
        signal = signal.flatten()

    return sampling_rate, signal


def read_aif(path):
    """
    Read audio file with .aif extension
    """
    sampling_rate = -1
    signal = np.array([])
    try:
        with aifc.open(path, 'r') as s:
            nframes = s.getnframes()
            strsig = s.readframes(nframes)
            signal = numpy.fromstring(strsig, numpy.short).byteswap()
            sampling_rate = s.getframerate()
    except:
        print("Error: read aif file. (DECODING FAILED)")
    return sampling_rate, signal


def read_audio_generic(path):
    """
    Function to read audio files with the following extensions
    [".mp3", ".wav", ".au", ".ogg"]
    """
    sampling_rate = -1
    signal = np.array([])
    try:
        audiofile = AudioSegment.from_file(path)
        data = np.array([])
        if audiofile.sample_width == 2:
            data = numpy.fromstring(audiofile._data, numpy.int16)
        elif audiofile.sample_width == 4:
            data = numpy.fromstring(audiofile._data, numpy.int32)

        if data.size > 0:
            sampling_rate = audiofile.frame_rate
            temp_signal = []
            for chn in list(range(audiofile.channels)):
                temp_signal.append(data[chn::audiofile.channels])
            signal = numpy.array(temp_signal).T
    except:
        print("Error: file not found or other I/O error. (DECODING FAILED)")
    return sampling_rate, signal


def stereo_to_mono(signal):
    """
    This function converts the input signal
    (stored in a numpy array) to MONO (if it is STEREO)
    """

    if signal.ndim == 2:
        if signal.shape[1] == 1:
            signal = signal.flatten()
        else:
            if signal.shape[1] == 2:
                signal = (signal[:, 1] / 2) + (signal[:, 0] / 2)
    return signal
