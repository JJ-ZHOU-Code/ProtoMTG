import torch
from torchmetrics.functional import multiscale_structural_similarity_index_measure as mmssim
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.functional import peak_signal_noise_ratio as psnr

class GenMeter(object):
    def __init__(self):
        self.ssim = 0
        self.mssim = 0
        self.psnr = 0
        self.mse = 0
        
        self.avgnum = 0

    @torch.no_grad()
    def update(self, pred, gt):
        img_real = gt
        img_fake = pred
        
        # img_real, img_fake = (img_real+1)/2, (img_fake+1)/2  # [-1, 1] -> [0, 1]
        img_real, img_fake = img_real.clamp(0, 1), img_fake.clamp(0, 1)
        
        self.ssim += ssim(img_real, img_fake)
        self.psnr += psnr(img_real, img_fake, data_range=1.)
        self.mssim += mmssim(img_real, img_fake, normalize='relu')
        self.mse += torch.mean(torch.square(img_real-img_fake)) 
        
        self.avgnum += 1
        
        # for img_real, img_fake in zip(real, results):
        #     img_real, img_fake = (img_real+1)/2, (img_fake+1)/2  # [-1, 1] -> [0, 1]
        #     img_real, img_fake = img_real.clamp(0, 1), img_fake.clamp(0, 1)

        #     self.ssim += ssim(img_real[None], img_fake[None])
        #     self.psnr += psnr(img_real[None], img_fake[None])
        #     self.mssim += mmssim(img_real[None], img_fake[None], normalize='relu')
        #     self.mse += torch.mean(torch.square(img_real-img_fake)) 
        
    def reset(self):
        self.ssim = 0
        self.mssim = 0
        self.psnr = 0
        self.mse = 0
            
    def get_score(self, logger=print, verbose=True):
        eval_result = dict()
        eval_result['ssim'] = self.ssim / self.avgnum
        eval_result['psnr'] = self.psnr / self.avgnum
        eval_result['mssim'] = self.mssim / self.avgnum
        eval_result['mse'] = self.mse / self.avgnum
        if verbose:
            logger(f'ssim: {eval_result["ssim"].item()}')
            logger(f'psnr: {eval_result["psnr"].item()}')
            logger(f'mssim: {eval_result["mssim"].item()}')
            logger(f'mse: {eval_result["mse"].item()}')
        return eval_result