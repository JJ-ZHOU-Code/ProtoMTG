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

from tqdm import tqdm

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

model_dir = './saved_models/' + base_architecture + '/' + experiment_run + '/'

makedir(model_dir)
ckp_dir = os.path.join(model_dir, 'ckp')
makedir(ckp_dir)
shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'config/settings.py'.format(datasetname)), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'models/proto/model.py'), dst=model_dir)


img_dir = os.path.join(model_dir, 'img')
makedir(img_dir)
weight_matrix_filename = 'outputL_weights'
prototype_img_filename_prefix = 'prototype-img'
prototype_self_act_filename_prefix = 'prototype-self-act'
proto_bound_boxes_filename_prefix = 'bb'

# load the data
from dataset import BasicDataset

train_dataset = BasicDataset(path_root=data_path, task_kwargs=task_kwargs, data_strengthen=True)
val_dataset = BasicDataset(path_root=data_path, task_kwargs=task_kwargs, data_strengthen=False)
test_dataset = BasicDataset(path_root=data_path, task_kwargs=task_kwargs, data_strengthen=False)

train_loader = torch.utils.data.DataLoader(train_dir, batch_size=train_batch_size, shuffle=True,
                            num_workers=4, pin_memory=False)
val_loader = torch.utils.data.DataLoader(val_dir, batch_size=train_push_batch_size, shuffle=False,
                            num_workers=4, pin_memory=False)
test_loader = torch.utils.data.DataLoader(test_dir, batch_size=test_batch_size, shuffle=False,
                            num_workers=4, pin_memory=False)
# we should look into distributed sampler more carefully at torch.utils.data.distributed.DistributedSampler(train_dataset)
print('training set size: {0}'.format(len(train_loader.dataset)))
print('test set size: {0}'.format(len(test_loader.dataset)))
print('batch size: {0}'.format(train_batch_size))

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
                            decoder_channels = decoder_channels,
                            weighted=True,
                            skip=True)

from utils.train_utils import count_params
count_params(ppnet, verbose=True)

ppnet = ppnet.cuda()
ppnet_multi = torch.nn.DataParallel(ppnet)

# # define optimizer
joint_optimizer = torch.optim.SGD(ppnet_multi.parameters(), lr=2e-2, momentum=0.9, weight_decay=1e-4)
joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=500, gamma=0.1)
warm_optimizer = torch.optim.SGD(ppnet_multi.parameters(), lr=5e-3, momentum=0.9, weight_decay=1e-4)

import logging
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)
# Resume from checkpoint
if os.path.exists(args.resume_path):
    print('Restart from checkpoint {}'.format(args.resume_path))
    checkpoint = torch.load(args.resume_path, map_location='cpu')
    warm_optimizer.load_state_dict(checkpoint['warm_optimizer'])
    joint_optimizer.load_state_dict(checkpoint['joint_optimizer'])
    joint_lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    ppnet_multi.load_state_dict(checkpoint['model'], strict=False)
    start_epoch = checkpoint['epoch']
    best_result = checkpoint['eval_res']
    logger.addHandler(logging.FileHandler(os.path.join(model_dir, 'train.log'), 'a+'))

else:
    print('No checkpoint file at {}'.format(args.resume_path))
    start_epoch = 0
    logger.addHandler(logging.FileHandler(os.path.join(model_dir, 'train.log'), 'w'))


# Get criterion
print('Get loss')
from utils.multitask_utils import get_criterion, get_output
criterion = get_criterion(task_kwargs)
criterion.cuda()

# number of training epochs, number of warm epochs, push start epoch, push epochs
from utils.multitask_utils import get_loss_meters, ProgressMeter
from push import push_prototypes
from utils.train_utils import train_vanilla, evaluation, warm_only, joint, stage_decoder

# train the model
logger.info('start training')
best_result = -1
cur_stg = 'one'
for epoch in range(start_epoch, num_train_epochs):

    logger.info('epoch: \t{0}'.format(epoch))
    if epoch < num_stage1_epochs:
        joint(ppnet_multi)
        # joint_lr_scheduler.step()
        # logger.info(joint_optimizer.state_dict()['param_groups'][0]['lr'])
        train_vanilla(stage=cur_stg, epoch=epoch, net=ppnet_multi, criterion=criterion, train_loader=train_loader, \
            optimizer=warm_optimizer, task_kwargs=task_kwargs, prototype_kwargs=prototype_kwargs, coefs=coefs, logger=logger)
    else: 
        if epoch == num_stage1_epochs:
            ppnet_multi.module.change_stage()
        cur_stg = 'two'
        logger.info('cur_stage: \t{0}'.format(cur_stg))
        ## update lr #########
        old_lr = joint_optimizer.state_dict()['param_groups'][0]['lr']
        joint_lr_scheduler.step()
        lr = joint_optimizer.state_dict()['param_groups'][0]['lr']
        print('learning rate %.7f -> %.7f' % (old_lr, lr))
        #######################
        # stage_decoder(ppnet_multi)
        train_vanilla(stage=cur_stg, epoch=epoch, net=ppnet_multi, criterion=criterion, train_loader=train_loader, \
            optimizer=joint_optimizer, task_kwargs=task_kwargs, prototype_kwargs=prototype_kwargs, coefs=coefs, logger=logger)

    
    if epoch % eval_per_epoch == 0:
        with torch.no_grad():
            eval_res = evaluation(stage=cur_stg, net=ppnet_multi, val_loader=val_loader, task_kwargs=task_kwargs, logger=logger)
        
    if epoch >= push_start and epoch in push_epochs:   
        push.push_prototypes(
            val_loader, # pytorch dataloader (must be unnormalized in [0,1])
            prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
            task_names=task_kwargs['task_names'],
            prototype_kwargs=prototype_kwargs,
            class_specific=False,
            preprocess_input_function=preprocess_input_function, # normalize if needed
            prototype_layer_stride=1,
            root_dir_for_saving_prototypes=img_dir, # if not None, prototypes will be saved here
            epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
            prototype_img_filename_prefix=prototype_img_filename_prefix,
            prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
            proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
            save_prototype_class_identity=False,
            log=logger.info)
    
    if epoch % eval_per_epoch == 0:
        with torch.no_grad():
            eval_res = evaluation(stage=cur_stg, net=ppnet_multi, val_loader=val_loader, task_kwargs=task_kwargs, logger=logger)
        # Checkpoint
        if epoch > 100 and epoch % 50: 
            print('Checkpoint ...')
            torch.save({'warm_optimizer': warm_optimizer.state_dict(), 'joint_optimizer': joint_optimizer.state_dict(), \
                'lr_scheduler': joint_lr_scheduler.state_dict(), 'model': ppnet_multi.state_dict(), \
                'epoch': epoch, 'eval_res': eval_res}, os.path.join(ckp_dir, 'epoch_{}.pt'.format(epoch)))

        cur_res = 0
        for name in task_kwargs['task_names']:
            cur_res += eval_res[name]['ssim']
        cur_res /= len(task_kwargs['task_names'])        
        if cur_res > best_result:
            best_result = cur_res
            best_epoch = epoch
            logger.info('\nbest_epoch: {}, best_result: {}\n'.format(best_epoch, best_result))
            torch.save({'warm_optimizer': warm_optimizer.state_dict(), 'joint_optimizer': joint_optimizer.state_dict(), \
            'lr_scheduler': joint_lr_scheduler.state_dict(), 'model': ppnet_multi.state_dict(), \
            'epoch': epoch, 'eval_res': eval_res, 'best_res':best_result}, os.path.join(ckp_dir, 'best.pt'))
        

# logclose()

