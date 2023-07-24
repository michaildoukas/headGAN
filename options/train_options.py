from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        # for data
        parser.add_argument('--max_seqs_per_identity', type=int, default=1000, help='How many short sequences (of 50 frames) to use per identity')
        # show, save frequencies
        parser.add_argument('--print_freq', type=int, default=1, help='frequency of showing training results on console')
        parser.add_argument('--display_freq', type=int, default=1, help='frequency of showing training results on screen')
        parser.add_argument('--save_latest_freq', type=int, default=100, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        # for display
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        parser.add_argument('--tf_log', action='store_true', help='if specified, use tensorboard logging. Requires tensorflow installed')
        # for training
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--niter', type=int, default=6, help='# of iter at starting learning rate.')
        parser.add_argument('--niter_decay', type=int, default=4, help='# of iter to linearly decay learning rate to zero.')
        parser.add_argument('--niter_start', type=int, default=3, help='in which epoch do we start doubling the training sequences length')
        parser.add_argument('--niter_step', type=int, default=1, help='every how many epochs we double the training sequences length')
        parser.add_argument('--optimizer', type=str, default='adam')
        parser.add_argument('--beta1', type=float, default=0.0, help='momentum term of adam')
        parser.add_argument('--beta2', type=float, default=0.9, help='momentum term of adam')
        parser.add_argument('--no_TTUR', action='store_true', default=True, help='Do not use TTUR training scheme')
        # the default values for beta1 and beta2 differ by TTUR option
        opt, _ = parser.parse_known_args()
        if opt.no_TTUR:
            parser.set_defaults(beta1=0.5, beta2=0.999)
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        # for discriminators and losses
        parser.add_argument('--n_steps_update_D', type=int, default=4, help='')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        parser.add_argument('--lambda_vgg', type=float, default=10.0, help='weight for vgg loss')
        parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
        parser.add_argument('--lambda_maskedL1', type=float, default=50.0, help='')
        parser.add_argument('--lambda_mask', type=float, default=10.0, help='')
        parser.add_argument('--no_vgg_loss', action='store_true', help='if specified, do *not* use VGG feature matching loss')
        parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
        parser.add_argument('--no_maskedL1_loss', action='store_true', help='')
        parser.add_argument('--gan_mode', type=str, default='hinge', help='(ls|original|hinge)')
        parser.add_argument('--norm_D', type=str, default='spectralinstance', help='instance normalization or batch normalization')
        parser.add_argument('--netD', type=str, default='multiscale', help='(n_layers|multiscale|image)')
        parser.add_argument('--netD_subarch', type=str, default='n_layer', help='architecture of each discriminator')
        parser.add_argument('--num_D', type=int, default=2, help='number of discriminators to be used in multiscale')
        parser.add_argument('--n_layers_D', type=int, default=4, help='# layers in each discriminator')
        parser.add_argument('--n_frames_D', type=int, default=3, help='number of frames to feed into temporal discriminator')
        parser.add_argument('--no_mouth_D', action='store_true', help='if true, do not use mouth discriminator')
        parser.add_argument('--ROI_size', type=int, default=64, help='The size of the mouth area.')
        self.isTrain = True
        return parser
