import os
import ntpath
import time
import collections
from . import util
from . import html
import scipy.misc
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x

class Visualizer():
    def __init__(self, opt):
        self.opt = opt
        self.tf_log = opt.isTrain and opt.tf_log
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        if self.tf_log:
            import tensorflow as tf
            self.tf = tf
            self.log_dir = os.path.join(opt.checkpoints_dir, opt.name, 'logs')
            self.writer = tf.compat.v1.summary.FileWriter(self.log_dir)

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            util.mkdirs([self.web_dir, self.img_dir])
        if opt.isTrain:
            self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training Loss (%s) ================\n' % now)

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, step):
        if self.use_html: # save images to a html file
            for label, image_numpy in visuals.items():
                if isinstance(image_numpy, list):
                    for i in range(len(image_numpy)):
                        img_path = os.path.join(self.img_dir, 'epoch%.3d_%s_%d.jpg' % (epoch, label, i))
                        util.save_image(image_numpy[i], img_path)
                else:
                    img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.jpg' % (epoch, label))
                    util.save_image(image_numpy, img_path)

            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=10)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims = []
                txts = []
                links = []

                for label, image_numpy in visuals.items():
                    if isinstance(image_numpy, list):
                        for i in range(len(image_numpy)):
                            img_path = 'epoch%.3d_%s_%d.jpg' % (n, label, i)
                            ims.append(img_path)
                            txts.append(label+str(i))
                            links.append(img_path)
                    else:
                        img_path = 'epoch%.3d_%s.jpg' % (n, label)
                        ims.append(img_path)
                        txts.append(label)
                        links.append(img_path)

                len_ims = len(ims)
                ims_per_batch = int(len_ims / self.opt.batch_size)
                n_prev = self.opt.n_frames_G-1 if not self.opt.no_previousframesencoder else 0
                n_frames_print_second_row = 5
                for i in range(self.opt.batch_size):
                    first_row_ims = ims[ims_per_batch*i:ims_per_batch*i+3+n_prev]
                    first_row_txts = txts[ims_per_batch*i:ims_per_batch*i+3+n_prev]
                    first_row_links = links[ims_per_batch*i:ims_per_batch*i+3 +n_prev]
                    webpage.add_images(first_row_ims, first_row_txts, first_row_links, width=self.win_size)
                    second_row_ims = ims[ims_per_batch*i+3+n_prev:ims_per_batch*i+3+n_prev+n_frames_print_second_row]
                    second_row_txts = txts[ims_per_batch*i+3+n_prev:ims_per_batch*i+3+n_prev+n_frames_print_second_row]
                    second_row_links = links[ims_per_batch*i+3+n_prev:ims_per_batch*i+3+n_prev+n_frames_print_second_row]
                    webpage.add_images(second_row_ims, second_row_txts, second_row_links, width=self.win_size)
            webpage.save()

    # errors: dictionary of error labels and values
    def plot_current_errors(self, errors, step):
        errors_sorted = collections.OrderedDict(sorted(errors.items()))
        if self.tf_log:
            for tag, value in errors_sorted.items():
                summary = self.tf.compat.v1.Summary(value=[self.tf.compat.v1.Summary.Value(tag=tag, simple_value=value)])
                self.writer.add_summary(summary, step)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t, t_epoch):
        errors_sorted = collections.OrderedDict(sorted(errors.items()))
        message = 'Epoch: %d, sequences seen: %d, total epoch time: %d hrs %d mins (secs per sequence: %.3f) \n' % (epoch, i, t_epoch[0], t_epoch[1], t)
        for k, v in errors_sorted.items():
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    # save image to the disk
    def save_images(self, image_dir, visuals, image_path, webpage=None):
        dirname = os.path.basename(os.path.dirname(image_path[0]))
        image_dir = os.path.join(image_dir, dirname)
        util.mkdir(image_dir)
        name = os.path.basename(image_path[0])
        name = os.path.splitext(name)[0]

        if webpage is not None:
            webpage.add_header(name)
            ims, txts, links = [], [], []

        for label, image_numpy in visuals.items():
            util.mkdir(os.path.join(image_dir, label))
            image_name = '%s.%s' % (name, 'png')
            save_path = os.path.join(image_dir, label, image_name)
            util.save_image(image_numpy, save_path)

            if webpage is not None:
                ims.append(image_name)
                txts.append(label)
                links.append(image_name)
        if webpage is not None:
            webpage.add_images(ims, txts, links, width=self.win_size)

    def vis_print(self, message):
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)
