from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        # Out directory.
        parser.add_argument('--time_fwd_pass', action='store_true', 
                            help='Show the forward pass time for synthesizing each frame.')
        # Default test arguments.
        parser.set_defaults(serial_batches=True)
        parser.set_defaults(batch_size=1)
        parser.set_defaults(nThreads=0)
        parser.set_defaults(phase='test')
        self.isTrain = False
        return parser
