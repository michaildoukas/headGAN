# Notice: This code has been taken from https://github.com/zllrunning/face-parsing.PyTorch#Demo and modified.

from .model import BiSeNet
import torch

class Segmenter():
    def __init__(self, gpu_id=0):
        self.net = BiSeNet(n_classes=19)
        self.net.cuda(gpu_id)
        model_path='files/79999_iter.pth'
        self.net.load_state_dict(torch.load(model_path))
        self.net.eval()

    def get_masks(self, imgs, size):
        with torch.no_grad():
            out = self.net(imgs)[0]
            mask = torch.argmax(out, dim=1)
            mask[mask > 0] = 1
            # Add channel dimesion and make double
            mask = mask.unsqueeze(1).double()
            mask = torch.nn.functional.interpolate(mask, size=size, mode='bilinear')
            return mask

    def join_masks(self, masks, ref_masks):
        mask_union = torch.clamp(masks + ref_masks, 0.0, 1.0)
        return mask_union
