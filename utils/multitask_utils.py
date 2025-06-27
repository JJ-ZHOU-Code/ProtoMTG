import os
import copy
import torch
import torch.nn.functional as F

from torchvision import transforms
from torch.utils.data import DataLoader


def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

""" 
    Loss functions 
"""
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss
# smp.losses.DiceLoss(mode='binary')  # 'multiclass' | 'binary'

def get_loss(task=None):
    """ Return loss function for a specific task """

    if task == 'semseg':
        from losses.loss_functions import SoftMaxwithLoss
        criterion = SoftMaxwithLoss()
    
    elif task.startswith('semseg'):
        from losses.loss_functions import BalancedCrossEntropyLoss, SoftBCEWithLogitsLoss
        criterion = DiceLoss(mode='binary')
        # SoftBCEWithLogitsLoss() | BalancedCrossEntropyLoss() | DiceLoss(mode='binary') | FocalLoss(mode='binary')
        
    elif task.startswith('gen'):
        criterion = torch.nn.L1Loss()
        
    elif task.startswith('gan'):
        from losses.gan_loss import GANLoss
        criterion = GANLoss(gan_mode = 'vanilla')
        
    else:
        raise NotImplementedError('Undefined Loss: Choose a task among '
                                  'edge, semseg, human_parts, sal, depth, or normals')

    return criterion


def get_criterion(task_kwargs):
    from losses.loss_schemes import MultiTaskLoss
    loss_ft = torch.nn.ModuleDict({task: get_loss(task) for task in task_kwargs['loss_weights'].keys()})
    loss_weights = task_kwargs['loss_weights']
    return MultiTaskLoss(task_kwargs['loss_weights'].keys(), loss_ft, loss_weights)

def get_criterion_new(loss_kwargs):
    from losses.loss_schemes import MultiTaskLoss
    loss_ft = torch.nn.ModuleDict({task: get_loss(losstype) for task, losstype in loss_kwargs['loss_type'].items()})
    loss_weights = loss_kwargs['loss_weights']
    return MultiTaskLoss(loss_kwargs['loss_weights'].keys(), loss_ft, loss_weights)

def get_output(output, task):
    # output = output.permute(0, 2, 3, 1)
    if task in {'semseg'}:
        _, output = torch.max(output, dim=1)
    elif task.startswith('semseg'):
        prob_mask = output.sigmoid()
        output = (prob_mask > 0.5).float()

    elif task.startswith('gen'):
        pass
    
    else:
        raise ValueError('Select one of the valid tasks')

    return output

def get_loss_meters(task_kwargs):
    """ Return dictionary with loss meters to monitor training """
    tasks = task_kwargs['task_names']

    losses = {task: AverageMeter('Loss %s' %(task), ':.4e') for task in tasks}
    losses['total'] = AverageMeter('Loss Total', ':.4e')
    return losses

class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, logger):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'