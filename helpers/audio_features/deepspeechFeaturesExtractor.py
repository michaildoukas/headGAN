import numpy as np
import os
import sys
import shlex
import subprocess
import wget
import wave
from deepspeech import Model

try:
    from shhlex import quote
except ImportError:
    from pipes import quote

def metadata_to_string(metadata):
    return ''.join(token.text for token in metadata.tokens)

def binary_search_interval(lst, left, right, key):
    idx = (left + right) // 2
    if key >= lst[idx][1] and key <= lst[idx][2]:
        return lst[idx][0]
    elif key < lst[idx][1]:
        return binary_search_interval(lst, left, idx, key)
    else:
        return binary_search_interval(lst, idx+1, right, key)

def create_window(lst, win):
    # Pad first and last item.
    front = lst[:1] * (win//2)
    back = lst[-1:] * (win//2-1) if (win % 2 == 0) else lst[-1:] * (win//2)
    front.extend(lst)
    front.extend(back)
    window_lst = []
    for i in range(len(lst)):
        w = front[i:i+win]
        window_lst.append(w)
    return window_lst

def token_window_to_onehot(lst):
    data = []
    for token in lst:
        order = ord(token)
        # 27 labels: 'a' -> 0, ..., 'z' --> 25, others --> 26
        if order > 96 and order < 123:
            # if alphabet char
            int_token = ord(token) - 97
        else:
            int_token = 26
        data.append(int_token)
    data = np.array(data)
    shape = (data.size, 27)
    onehot = np.zeros(shape)
    rows = np.arange(data.size)
    onehot[rows, data] = 1

    # Concatenate one-hot embeddings for window.
    onehot_vector = onehot.flatten()
    return onehot_vector


def convert_samplerate(audio_path, desired_sample_rate):
    sox_cmd = 'sox {} --type raw --bits 16 --channels 1 --rate {} --encoding signed-integer --endian little --compression 0.0 --no-dither - '.format(quote(audio_path), desired_sample_rate)
    try:
        output = subprocess.check_output(shlex.split(sox_cmd), stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError('SoX returned non-zero status: {}'.format(e.stderr))
    except OSError as e:
        raise OSError(e.errno, 'SoX not found, use {}hz files or install it: {}'.format(desired_sample_rate, e.strerror))

    return desired_sample_rate, np.frombuffer(output, np.int16)

def open_audio(audio_path, desired_sample_rate):
    fin = wave.open(audio_path, 'rb')
    fs_orig = fin.getframerate()
    if fs_orig != desired_sample_rate:
        print('Warning: original sample rate ({}) is different than {}hz. Resampling might produce erratic speech recognition.'.format(fs_orig, desired_sample_rate), file=sys.stderr)
        fs_new, audio = convert_samplerate(audio_path, desired_sample_rate)
    else:
        audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)

    audio_length = fin.getnframes() * (1/fs_orig)
    fin.close()
    return audio, audio_length

def get_logits(audio_path, fps, window):
    model_path='files/deepspeech-0.8.1-models.pbmm'
    scorer_path='files/deepspeech-0.8.1-models.scorer'

    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))

    if not os.path.isfile(model_path):
        print('DeepSpeech model file not found. Downloading...\n')
        wget.download('https://github.com/mozilla/DeepSpeech/releases/download/v0.8.1/deepspeech-0.8.1-models.pbmm',
                      model_path)
    if not os.path.isfile(scorer_path):
        print('DeepSpeech scorer file not found. Downloading...\n')
        wget.download('https://github.com/mozilla/DeepSpeech/releases/download/v0.8.1/deepspeech-0.8.1-models.scorer',
                      scorer_path)

    ds = Model(model_path)

    desired_sample_rate = ds.sampleRate()
    ds.enableExternalScorer(scorer_path)
    audio, audio_length = open_audio(audio_path, desired_sample_rate)

    # Get tokens from deepspeech network
    transcripts = ds.sttWithMetadata(audio, 1).transcripts[0]
    print('\nText: ' + metadata_to_string(transcripts))
    tokens = transcripts.tokens
    if len(tokens) == 0:
        # If no speech detected, create a single frame with ' ' token.
        data = np.array([26] * window)
        shape = (data.size, 27)
        onehot = np.zeros(shape)
        rows = np.arange(data.size)
        onehot[rows, data] = 1

        # Concatenate one-hot embeddings for window.
        onehot_vector = onehot.flatten()
        return [onehot_vector]

    # Create results list: [(token, start_time, end_time), ...]
    results = [(' ', 0.0, tokens[0].start_time)]
    for i in range(len(tokens)):
        token = tokens[i]
        next_token = tokens[i+1] if i < len(tokens) - 1 else None
        text = token.text
        start_time = token.start_time
        end_time = next_token.start_time if next_token else audio_length
        results.append((text, start_time, end_time))

    # Get per frame token - align each frame with a token.
    frame_tokens = []
    n_frames = round(fps * audio_length)
    offset_n_frames = 5
    for n in range(n_frames):
        t = (n + offset_n_frames) * audio_length / n_frames
        if t < audio_length:
            token = binary_search_interval(results, 0, len(results), t)
        else:
            token = results[-1][0]
        frame_tokens.append(token)

    # Create windows of tokens around each frame.
    window_tokens = create_window(frame_tokens, window)

    # One-hot encoding of tokens.
    deepspeech_feats = []
    for window_token in window_tokens:
        deepspeech_feats.append(token_window_to_onehot(window_token))
    return deepspeech_feats
