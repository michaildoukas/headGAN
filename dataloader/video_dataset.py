import os
import random
import torch
import numpy as np
from PIL import Image
from dataloader.base_dataset import BaseDataset, get_params, get_transform, get_transform_segmenter, get_video_parameters
from dataloader.image_folder import make_video_dataset, assert_valid_pairs
from dataloader.landmarks_to_sketch import create_landmarks_sketch

class videoDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        input_type = 'landmarks70' if opt.use_landmarks_input else 'nmfcs'
        max_seqs_per_identity = self.opt.max_seqs_per_identity if opt.isTrain else None

        # Get dataset directories.
        if not self.opt.no_audio_input:
            self.dir_audio_feats = os.path.join(opt.dataroot, self.opt.phase, 'audio_features')
        self.dir_nmfc = os.path.join(opt.dataroot, self.opt.phase, input_type)
        self.dir_image = os.path.join(opt.dataroot, self.opt.phase, 'images')
        self.dir_nmfc_ref = os.path.join(opt.dataroot, self.opt.phase, input_type + '_fs')
        self.dir_image_ref = os.path.join(opt.dataroot, self.opt.phase, 'images_fs')

        # Collect image paths.
        if not self.opt.no_audio_input:
            self.audio_feats_paths = make_video_dataset(self.dir_audio_feats, opt.target_name, max_seqs_per_identity)
        self.nmfc_paths = make_video_dataset(self.dir_nmfc, opt.target_name, max_seqs_per_identity)
        self.image_paths = make_video_dataset(self.dir_image, opt.target_name, max_seqs_per_identity)
        self.nmfc_ref_paths = make_video_dataset(self.dir_nmfc_ref, opt.target_name)
        self.image_ref_paths = make_video_dataset(self.dir_image_ref, opt.target_name)

        # Make sure paths are okay.
        if not self.opt.no_audio_input:
            assert_valid_pairs(self.audio_feats_paths, self.image_paths)
        assert_valid_pairs(self.nmfc_paths, self.image_paths)
        assert_valid_pairs(self.nmfc_ref_paths, self.image_ref_paths)

        self.dir_landmark = os.path.join(opt.dataroot, self.opt.phase, 'landmarks70')
        self.landmark_paths = make_video_dataset(self.dir_landmark, opt.target_name, max_seqs_per_identity)
        assert_valid_pairs(self.landmark_paths, self.image_paths)

        self.n_of_seqs = len(self.nmfc_paths)
        self.seq_len_max = max([len(A) for A in self.nmfc_paths])
        self.init_frame_index(self.nmfc_paths)
        self.create_identities_dict()

    def __getitem__(self, index):
        # Get sequence paths.
        seq_idx = self.update_frame_index(self.nmfc_paths, index)
        if not self.opt.no_audio_input:
            audio_feats_paths = self.audio_feats_paths[seq_idx]
        nmfc_paths = self.nmfc_paths[seq_idx]
        image_paths = self.image_paths[seq_idx]
        landmark_paths = self.landmark_paths[seq_idx]

        nmfc_len = len(nmfc_paths)

        # Get identity number
        identity_num = self.identities_dict[self.get_identity_name(nmfc_paths[0])]

        # Get reference frames paths.
        if self.opt.isTrain and self.opt.reference_frames_strategy  == 'previous':
            # Do not use the reference image directories.
            # Instead get the paths of previous sequence with the same identity.
            ref_seq_idx = self.get_ref_seq_idx(seq_idx, identity_num)
            ref_nmfc_paths = self.nmfc_paths[ref_seq_idx]
            ref_image_paths = self.image_paths[ref_seq_idx]
        else:
            # Use the reference image directories.
            ref_nmfc_paths = self.nmfc_ref_paths[identity_num]
            ref_image_paths = self.image_ref_paths[identity_num]

        # Get parameters and transforms.
        n_frames_total, start_idx = get_video_parameters(self.opt, self.n_frames_total, nmfc_len, self.frame_idx)
        first_image = Image.open(image_paths[0]).convert('RGB')
        params = get_params(self.opt, first_image.size)
        transform_nmfc = get_transform(self.opt, params, normalize=False) # do not normalize nmfc values
        transform_image = get_transform(self.opt, params)
        transform_image_segmenter = get_transform_segmenter()
        change_seq = False if self.opt.isTrain else self.change_seq

        # Read data
        paths = []
        audio_feats = image = nmfc = 0
        image_segmenter = 0
        mouth_centers = 0
        for i in range(n_frames_total):
            if not self.opt.no_audio_input:
                # Audio features
                audio_feats_path = audio_feats_paths[start_idx + i]
                audio_feats_i = self.get_audio_feats(audio_feats_path)
                audio_feats = audio_feats_i if i == 0 else torch.cat([audio_feats, audio_feats_i], dim=0)
            # NMFCs
            nmfc_path = nmfc_paths[start_idx + i]
            if self.opt.use_landmarks_input:
                nmfc_i = create_landmarks_sketch(nmfc_path, first_image.size, transform_nmfc)
            else:
                nmfc_i = self.get_image(nmfc_path, transform_nmfc)
            nmfc = nmfc_i if i == 0 else torch.cat([nmfc, nmfc_i], dim=0)
            # Images
            image_path = image_paths[start_idx + i]
            image_i = self.get_image(image_path, transform_image)
            image = image_i if i == 0 else torch.cat([image, image_i], dim=0)
            if self.opt.isTrain:
                # Read images using data transform for foreground segmenter network.
                image_segmenter_i = self.get_image(image_path, transform_image_segmenter)
                image_segmenter = image_segmenter_i if i == 0 else torch.cat([image_segmenter, image_segmenter_i], dim=0)
            # Mouth centers
            if self.opt.isTrain and not self.opt.no_mouth_D:
                landmark_path = landmark_paths[start_idx + i]
                mouth_centers_i = self.get_mouth_center(landmark_path)
                mouth_centers = mouth_centers_i if i == 0 else torch.cat([mouth_centers, mouth_centers_i], dim=0)
            # Paths
            paths.append(nmfc_path)

        ref_nmfc = ref_image = 0
        ref_image_segmenter = 0
        if self.opt.isTrain:
            # Sample one frame from the directory with reference frames.
            ref_idx = random.sample(range(0, len(ref_nmfc_paths)), 1)[0]
        else:
            # During test, use the middle frame from the directory with reference frames.
            ref_idx = len(ref_nmfc_paths) // 2

        # Read reference frame and corresponding NMFC.
        ref_nmfc_path = ref_nmfc_paths[ref_idx]
        if self.opt.use_landmarks_input:
            ref_nmfc = create_landmarks_sketch(ref_nmfc_path, first_image_image.size, transform_nmfc)
        else:
            ref_nmfc = self.get_image(ref_nmfc_path, transform_nmfc)
        ref_image_path = ref_image_paths[ref_idx]
        ref_image = self.get_image(ref_image_path, transform_image)
        if self.opt.isTrain:
            # Read reference frame using data transform for foreground segmenter network.
            ref_image_segmenter = self.get_image(ref_image_path, transform_image_segmenter)

        return_list = {'audio_feats': audio_feats, 'nmfc': nmfc, 'image': image, 'image_segmenter': image_segmenter,
                       'ref_image': ref_image, 'ref_nmfc': ref_nmfc, 'mouth_centers': mouth_centers,
                       'paths': paths, 'change_seq': change_seq, 'ref_image_segmenter': ref_image_segmenter}
        return return_list

    def get_ref_seq_idx(self, seq_idx, identity_num):
        # Assuming that each identity has at least 2 sequences in the dataset.
        ref_seq_idx = seq_idx - 1 if seq_idx > 0 and self.sequences_ids[seq_idx-1] == identity_num else seq_idx + 1
        return min(len(self.nmfc_paths)-1, ref_seq_idx)

    def create_identities_dict(self):
        self.identities_dict = {}
        self.sequences_ids = []
        id_cnt = 0
        for path in self.nmfc_paths:
            name = self.get_identity_name(path[0])
            if name not in self.identities_dict:
                self.identities_dict[name] = id_cnt
                id_cnt += 1
            self.sequences_ids.append(self.identities_dict[name])
        self.n_identities = id_cnt

    def get_identity_name(self, A_path):
        identity_name = os.path.basename(os.path.dirname(A_path))
        identity_name = identity_name[:-7] if self.opt.isTrain else identity_name
        return identity_name

    def get_image(self, A_path, transformA, convert_rgb=True):
        A_img = Image.open(A_path)
        if convert_rgb:
            A_img = A_img.convert('RGB')
        A_scaled = transformA(A_img)
        return A_scaled

    def get_audio_feats(self, audio_feats_path):
        audio_feats = np.loadtxt(audio_feats_path)
        assert audio_feats.shape[0] == self.opt.naf, '%s does not have %d audio features' % (audio_feats_path, self.opt.naf)
        return torch.tensor(audio_feats).float()

    def get_mouth_center(self, A_path):
        keypoints = np.loadtxt(A_path, delimiter=' ')
        # Mouth landmarks
        pts = keypoints[48:, :].astype(np.int32) 
        mouth_center = np.median(pts, axis=0)
        mouth_center = mouth_center.astype(np.int32)
        return torch.tensor(mouth_center)

    def __len__(self):
        if self.opt.isTrain:
            return len(self.nmfc_paths)
        else:
            return sum(self.n_frames_in_sequence)

    def name(self):
        return 'video'