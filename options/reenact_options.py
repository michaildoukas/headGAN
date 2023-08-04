from .base_options import BaseOptions

class ReenactOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        # Basic options
        parser.add_argument('--reference_path', type=str, default='assets/reference.png',
                            help='The path to reference image')
        parser.add_argument('--driving_path', type=str, default='assets/driving.mp4',
                            help='The path to driving .mp4 video')
        parser.add_argument('--no_crop', action='store_true', 
                            help='If set, do not perform face detection and cropping')
        parser.add_argument('--show_warped', action='store_true', 
                            help='If set, add the warped image to the results')
        # Face detection options
        parser.add_argument('--gpu_id', type=int, default=0, 
                            help='The gpu id for face detector and face reconstruction modules')
        parser.add_argument('--mtcnn_batch_size', default=1, type=int, 
                            help='The number of frames for face detection')
        parser.add_argument('--cropped_image_size', default=256, type=int, 
                            help='The size of frames after cropping the face')
        parser.add_argument('--margin', default=100, type=int, 
                            help='The margin around the face bounding box')
        parser.add_argument('--dst_threshold', default=0.35, type=float, 
                            help='Max L_inf distance between any bounding boxes in a video. (normalised by image size: (h+w)/2)')
        parser.add_argument('--height_recentre', default=0.0, type=float, 
                            help='The amount of re-centring bounding boxes lower on the face')
        # Reenactment options
        parser.add_argument('--no_scale_or_translation_adaptation', action='store_true',
                            help='Do not perform scale or translation adaptation using statistics from driving video')
        parser.add_argument('--no_translation_adaptation', action='store_true',
                            help='Do not perform translation adaptation using statistics from driving video')
        parser.add_argument('--standardize', action='store_true',
                            help='Perform adaptation using also std from driving videos')
        # Default reenact arguments
        parser.set_defaults(batch_size=1)
        self.isTrain = False
        return parser