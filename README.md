# HeadGAN: One-shot Neural Head Synthesis and Editing

This repository contains the official PyTorch implementation of HeadGAN. Our method performs one-shot full head reenactment by transferring the facial expressions and head pose from a driving video to a reference facial image. Please note that this is a stripped down version of the code for the purposes of experimentation and comparison with other methods.

> **HeadGAN: One-shot Neural Head Synthesis and Editing**<br>
> [Michail Christos Doukas](https://michaildoukas.github.io), [Stefanos Zafeiriou](), [Viktoriia Sharmanska]()<br>
> In IEEE/CVF International Conference on Computer Vision (ICCV), 2021<br>
>
> Paper: https://arxiv.org/pdf/2012.08261.pdf<br>
> Video demo: https://www.youtube.com/watch?v=Xo9IW3cMGTg<br>
> Project Page: https://michaildoukas.github.io/HeadGAN<br>

## Installation

First, install ```sox``` for audio features extraction:
```
sudo apt-get install sox
```

Create and activate conda environment:
```
conda env create -f env.yml
conda activate headGAN
```

For 3D face reconstruction you might need to install Vulkan:
```
sudo apt install libvulkan1 vulkan-utils
```

Download the pre-trained model weights and other essential files (i.e. 3DMMs, weights of models for data pre-preprocessing, etc.):
```
python scripts/download_files.py
```

Please note that the pre-trained model weights are stored under the ```checkpoints/``` directory, and all other essential files under ```files/```.

## Usage

![](assets/cross-reenactment.gif)

In order to reenact a reference image with a driving video, run the following:
```
python reenact.py --reference_path <reference_path> --driving_path <driving_path>
```
**Important**: 

- In case you want to test the model on the task of reconstruction (self-reenactment), please use the ```--no_scale_or_translation_adaptation``` argument.

- In case the input image and video are already cropped around the face you can use the ```--no_crop``` argument to omit the face detection pre-processing step.

- Please note that the driving video file must contain an audio stream.

The generated samples are stored under the ```results/``` directory.

## Train your own model

### Proccess the video database

1) Given a directory of video files, run face detection and save images of cropped faces. Here, ```<videos_dir>``` is the input directory of .mp4 videos, ```<dataset_save_path>``` is the path to save the processed dataset and ```<split>``` is either ```train``` or ```test``` (default: ```train```).

```
python detect_faces.py --original_files_path <videos_dir> --dataset_path <dataset_save_path> --split <split>
```

2) Then, run 3D face reconstruction:
```
python reconstruct.py --dataset_path <dataset_save_path>
```

3) Compute facial landmarks:
```
python detect_landmarks.py --dataset_path <dataset_save_path>
```

4) In case you want to train a full HeadGAN model with audio feature inputs, run audio feature extraction on video data:
```
python extract_audio_features.py --original_files_path <videos_dir> --dataset_path <dataset_save_path> --split <split>
```

### Train a model from scratch

In order to train a new model using the processed dataset in ```<dataset_save_path>```, run:
```
python train.py --name <model_name> --dataroot <dataset_save_path> --batch_size <batch_size> --gpu_ids <gpu_ids>
```
Use the ```--no_audio_input``` argument in case you want to train a variation of HeadGAN that does not condition synthesis on audio features.

### Test reconstruction

Given that you have created a test data split in the ```<dataset_save_path>```, you can test reconstruction (self-reenactment) on the test set by running the following:
```
python test.py --name <model_name> --dataroot <dataset_save_path> --gpu_ids <gpu_ids>
```
During inference by default ```batch_size=1```, 

## Citation

If you find this code useful, please cite our paper.

```
@InProceedings{Doukas_2021_ICCV,
    author    = {Doukas, Michail Christos and Zafeiriou, Stefanos and Sharmanska, Viktoriia},
    title     = {HeadGAN: One-Shot Neural Head Synthesis and Editing},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {14398-14407}
}
```

## Notes
This code borrows from [vid2vid](https://github.com/NVIDIA/vid2vid) and [SPADE](https://github.com/NVlabs/SPADE).