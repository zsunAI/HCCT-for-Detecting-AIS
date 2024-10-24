import argparse
import os
import SimpleITK as sitk


class Options():

    """This class defines options used during both training and test time."""

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):

        # basic parameters
        parser.add_argument('--lr_dir', type=str, default='/home/hydu/CT_SR/processCT/newdata/data3/LR_nearest/')
        parser.add_argument('--hr_dir', type=str, default='/home/hydu/CT_SR/processCT/newdata/data3/HR_nearest/')
        parser.add_argument('--preload1', type=str, default=None)
        parser.add_argument('--preload2', type=str, default=None)
        parser.add_argument('--gpu_ids', type=str, default='0,1,2,3', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--workers', default=8, type=int, help='number of data loading workers')
        parser.add_argument('--save_dir', type=str, default='./checkpoints')
        parser.add_argument('--save_model_epochs', type=int, default=20)
        parser.add_argument('--batch_size', type=int, default=2, help='batch size')
        parser.add_argument('--in_channels', default=1, type=int, help='Channels of the input')
        parser.add_argument('--out_channels', default=1, type=int, help='Channels of the output')

        # training parameters
        parser.add_argument('--epochs', default=500, help='Number of epochs')
        parser.add_argument('--lr', default=0.0001, help='Learning rate')

        # Inference
        # This is just a trick to make the predict script working
        parser.add_argument('--result', default=None, help='Keep this empty and go to predict_single_image script')
        parser.add_argument('--weights', default=None, help='Keep this empty and go to predict_single_image script')

        self.initialized = True
        return parser

    def parse(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        opt = parser.parse_args()
        # set gpu ids
        if opt.gpu_ids != '-1':
            os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
        return opt



