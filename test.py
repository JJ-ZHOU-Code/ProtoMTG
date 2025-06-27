import os
import shutil

import torch
import torch.nn as nn
import torch.utils.data
# import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import argparse
import re

from helpers import makedir
import push
import prune
import save
from preprocess import mean, std, preprocess_input_function

parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0') # python3 main.py -gpuid=0,1,2,3
parser.add_argument('-resume_path', type=str, default='')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
print(os.environ['CUDA_VISIBLE_DEVICES'])

# book keeping namings and code
from config.settings import *

base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)

# model_dir = '/home/test/zjj/code/Proto-Multitask/saved_models/proto/' + base_architecture + '/' + task_kwargs['task_names'][0] +'/'+ experiment_run + '/'
model_dir = base_architecture + experiment_run + '/'
args.resume_path = os.path.join(model_dir, 'ckp/best.pt')

img_dir = os.path.join(model_dir, 'img')
makedir(img_dir)
weight_matrix_filename = 'outputL_weights'
prototype_img_filename_prefix = 'prototype-img'
prototype_self_act_filename_prefix = 'prototype-self-act'
proto_bound_boxes_filename_prefix = 'bb'

# load the data
from dataset import BasicDataset
test_dataset = BasicDataset(path_root=test_dir,task_kwargs=task_kwargs,data_strengthen=False,)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=0, pin_memory=False)

# construct the model
import models.proto.model as model

ppnet = model.construct_PMTG(base_architecture=base_architecture,
                            task_kwargs=task_kwargs,
                            prototype_kwargs=prototype_kwargs,
                            pretrained=True, img_size=img_size,
                            prototype_shape=prototype_kwargs['shape'],
                            prototype_activation_function=prototype_activation_function,
                            add_on_layers_type=add_on_layers_type, 
                            proto_combine_mode = proto_combine_mode, 
                            decoder_channels=decoder_channels,
                            weighted=True)
ppnet = ppnet.cuda()
ppnet_multi = torch.nn.DataParallel(ppnet)

# Get criterion
print('Get loss')
from utils.multitask_utils import get_criterion, get_output
criterion = get_criterion(task_kwargs)
criterion.cuda()
print(criterion)

# Resume from checkpoint
if os.path.exists(args.resume_path):
    print('Restart from checkpoint {}'.format(args.resume_path))
    checkpoint = torch.load(args.resume_path, map_location='cpu')
    ppnet_multi.load_state_dict(checkpoint['model'],strict=False)
    start_epoch = checkpoint['epoch']
    print(start_epoch)
    best_result = checkpoint['eval_res']

else:
    print('No checkpoint file at {}'.format(args.resume_path))
    start_epoch = 0

# weighting of different training losses
# number of training epochs, number of warm epochs, push start epoch, push epochs
from utils.multitask_utils import get_loss_meters, ProgressMeter

def evaluation(net, val_loader):
    from evaluation.evaluate_utils import PerformanceMeter
    performance_meter = PerformanceMeter(task_kwargs)
    net.eval()
    js_loss_dict = {}
    for name in task_kwargs['task_names'] + ['share']:
        js_loss_dict[name] = []
    for idx, batch in enumerate(val_loader):
        images = batch['image'].cuda(non_blocking=True)
        targets = {task: batch[task].cuda(non_blocking=True) for task in task_kwargs['task_names']}
        min_distances, pred, proto_activations_taskdict, weights_dict, combine_proto_activations_taskdict = net(images, stage='two', return_weights=True)
        
        names = task_kwargs['task_names'] + ['share']
        performance_meter.update({t: get_output(pred[t], t) for t in task_kwargs['task_names']}, 
                                 {t: targets[t] for t in task_kwargs['task_names']})
    return

# train the model
best_result = -1

with torch.no_grad():
    eval_res = evaluation(net=ppnet_multi, val_loader=test_loader)


