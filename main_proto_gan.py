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
# import train_and_test as tnt
import save
# from log import create_logger
from preprocess import mean, std, preprocess_input_function

import sys
sys.path.append('.')

parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='3') # python3 main.py -gpuid=0,1,2,3
parser.add_argument('-resume_path', type=str, default='')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
print(os.environ['CUDA_VISIBLE_DEVICES'])

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

train_dataset = BasicDataset(path_root=os.path.join(data_path, 'train'), task_kwargs=task_kwargs, phase='train', data_strengthen=True)
push_dataset = BasicDataset(path_root=os.path.join(data_path, 'train'), task_kwargs=task_kwargs, phase='push', data_strengthen=False)
val_dataset = BasicDataset(path_root=os.path.join(data_path, 'train'), task_kwargs=task_kwargs, phase='val', data_strengthen=False)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True,
                            num_workers=4, pin_memory=False)
push_loader = torch.utils.data.DataLoader(push_dataset, batch_size=test_batch_size, shuffle=False,
                            num_workers=4, pin_memory=False)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=train_push_batch_size, shuffle=False,
                            num_workers=4, pin_memory=False)
# we should look into distributed sampler more carefully at torch.utils.data.distributed.DistributedSampler(train_dataset)
print('training set size: {0}'.format(len(train_loader.dataset)))
print('push set size: {0}'.format(len(push_loader.dataset)))
print('validation set size: {0}'.format(len(val_loader.dataset)))
print('batch size: {0}'.format(train_batch_size))

# construct the model
import models.proto.model as model
ppnet = model.construct_PPNet(base_architecture=base_architecture,
                            task_kwargs=task_kwargs,
                            prototype_kwargs=prototype_kwargs,
                            pretrained=True, img_size=img_size,
                            prototype_shape=prototype_kwargs['shape'],
                            prototype_activation_function=prototype_activation_function,
                            add_on_layers_type=add_on_layers_type, 
                            proto_combine_mode = proto_combine_mode,
                            decoder_channels = decoder_channels,
                            weighted=True, skip=False, use_res=False)

from utils.train_utils import count_params
count_params(ppnet)

from models.discriminator import Multi_Discriminator
netD = Multi_Discriminator(task_kwargs)

ppnet = ppnet.cuda()
ppnet_multi = torch.nn.DataParallel(ppnet)
netD = netD.cuda()
netD_multi = torch.nn.DataParallel(netD)

# # define optimizer
# from settings_gen_stages import joint_optimizer_lrs, joint_lr_step_size
# optimizer = torch.optim.SGD(ppnet.parameters(), lr=1e-4, weight_decay=1e-3)
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=joint_lr_step_size, gamma=0.1)
# define optimizer
# from settings_gan import joint_optimizer_lrs, joint_lr_step_size
joint_optimizer_specs = \
[{'params': ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3}, # bias are now also being regularized
 {'params': ppnet.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
 {'params': ppnet.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors'], 'weight_decay': 1e-3},
 {'params': ppnet.selayers.parameters(), 'lr': joint_optimizer_lrs['selayers'], 'weight_decay': 1e-3},
 {'params': ppnet.decoders.parameters(), 'lr': joint_optimizer_lrs['decoders'], 'weight_decay': 1e-3},
 {'params': ppnet.warm_decoder.parameters(), 'lr': joint_optimizer_lrs['warm_decoder'], 'weight_decay': 1e-3},
]
joint_optimizer = torch.optim.SGD(joint_optimizer_specs)
joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=joint_lr_step_size, gamma=0.5)

# from settings_gan import warm_optimizer_lrs
warm_optimizer_specs = \
[
 {'params': ppnet.add_on_layers.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
 {'params': ppnet.prototype_vectors, 'lr': warm_optimizer_lrs['prototype_vectors'],'weight_decay': 1e-3},
 {'params': ppnet.selayers.parameters(), 'lr': warm_optimizer_lrs['selayers'], 'weight_decay': 1e-3},
 {'params': ppnet.decoders.parameters(), 'lr': warm_optimizer_lrs['decoders'], 'weight_decay': 1e-3},
 {'params': ppnet.warm_decoder.parameters(), 'lr': warm_optimizer_lrs['warm_decoder'], 'weight_decay': 1e-3},
]
warm_optimizer = torch.optim.SGD(warm_optimizer_specs)

last_layer_optimizer_specs = [{'params': ppnet.decoders.parameters(), 'lr': warm_optimizer_lrs['decoders'], 'weight_decay': 1e-3}]
last_layer_optimizer = torch.optim.SGD(last_layer_optimizer_specs)

# from settings_gan import d_lr
optimizer_D = torch.optim.SGD(netD.parameters(), lr=d_lr)


####
# joint_optimizer = torch.optim.SGD(ppnet_multi.parameters(), lr=1e-1, momentum=0.9, weight_decay=1e-4)
# joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=500, gamma=0.1)

# warm_optimizer = torch.optim.SGD(ppnet_multi.parameters(), lr=2e-2, momentum=0.9, weight_decay=1e-4)
# optimizer_D = torch.optim.SGD(netD.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)

joint_optimizer = torch.optim.Adam(ppnet_multi.parameters(), lr=2e-4, betas=(0.5, 0.999))
joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=joint_lr_step_size, gamma=0.1)

warm_optimizer = torch.optim.Adam(ppnet_multi.parameters(), lr=2e-4, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.999))

#####

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
    optimizer_D.load_state_dict(checkpoint['optimizer_D'])
    ppnet_multi.load_state_dict(checkpoint['model'], strict=False)
    netD.load_state_dict(checkpoint['netD'])
    start_epoch = checkpoint['epoch']
    best_result = checkpoint['eval_res']
    logger.addHandler(logging.FileHandler(os.path.join(model_dir, 'train.log'), 'a+'))

else:
    print('No checkpoint file at {}'.format(args.resume_path))
    start_epoch = 0
    logger.addHandler(logging.FileHandler(os.path.join(model_dir, 'train.log'), 'w'))

# number of training epochs, number of warm epochs, push start epoch, push epochs
# from settings_gan import num_train_epochs, num_warm_epochs, push_start, push_epochs, coefs
from utils.multitask_utils import get_loss_meters, ProgressMeter
from tqdm import tqdm

from utils.train_utils import compute_kld,compute_sep, warm_only, joint, stage_two, stage_one, set_requires_grad


# Get criterion
print('Get loss')
from utils.multitask_utils import get_criterion, get_output, get_criterion_new
from utils.train_utils import compute_div 

criterionL1 = get_criterion_new(task_kwargs['loss']['l1'])
criterionL1.cuda()

criterionGAN = get_criterion_new(task_kwargs['loss']['gan'])
criterionGAN.cuda()
# from losses.gan_loss import GANLoss
# criterionGAN = GANLoss(gan_mode = 'vanilla')

def train_vanilla(stage, epoch, net, netD, criterionL1, criterionGAN, train_loader, optimizer_G, optimizer_D, task_kwargs, kld_loss_weight, logger):
    
    loss_name_D = ['loss_D_fake', 'loss_D_real']
    loss_name_G = ['loss_G_GAN', 'loss_G_L1']
    losses_D = {name: get_loss_meters(task_kwargs) for name in loss_name_D}
    losses_G = {name: get_loss_meters(task_kwargs) for name in loss_name_G}
    progress_D = {name: ProgressMeter(len(train_loader),[v for v in losses_D[name].values()], prefix="Epoch: [{}]".format(epoch)) \
                        for name in loss_name_D}
    progress_G = {name: ProgressMeter(len(train_loader),[v for v in losses_G[name].values()], prefix="Epoch: [{}]".format(epoch)) \
                        for name in loss_name_G}
    
    lossdict_D, lossdict_G = {}, {}
    
    net.train()
    for idx, batch in enumerate(train_loader):
        images = batch['image'].cuda(non_blocking=True)
        targets = {task: batch[task].cuda(non_blocking=True) for task in task_kwargs['task_names']}

        min_distances, pred, proto_activations_taskdict = net(images, stage)
        
        # Discriminator
        # if cur_stg == 'two':
        set_requires_grad(netD, True)
        optimizer_D.zero_grad() 
        pred_fake = netD(images, pred, n_detach=True)
        F_dict = {task: False for task in task_kwargs['task_names']}
        loss_D_fake = criterionGAN(pred_fake, F_dict) # {task: }
        lossdict_D['loss_D_fake'] = loss_D_fake 
        
        pred_real = netD(images, targets)
        T_dict = {task: True for task in task_kwargs['task_names']}
        loss_D_real = criterionGAN(pred_real, T_dict) # {task: }
        lossdict_D['loss_D_real'] = loss_D_real 
        
        loss_D = (loss_D_fake['total'] + loss_D_real['total']) * 0.5
        
        loss_D.backward()
        optimizer_D.step() 
        
        for name in loss_name_D:
            for k, v in lossdict_D[name].items():
                losses_D[name][k].update(v.item())             

        # Generator
        set_requires_grad(netD, False)
        optimizer_G.zero_grad()
        # activation loss
        act_cost = torch.mean(min_distances).item()
        # diversity loss
        if prototype_kwargs['num_shared']:
            names = task_kwargs['task_names'] + ['share']
        else:
            names = task_kwargs['task_names']
        div_loss = compute_div(names, proto_activations_taskdict)
        
        # L1 loss
        loss_G_L1 = criterionL1(pred, targets)
        lossdict_G['loss_G_L1'] = loss_G_L1
        # GAN loss
        pred_fake = netD(images, pred)
        T_dict = {task: True for task in task_kwargs['task_names']}
        loss_G_GAN = criterionGAN(pred_fake, T_dict)
        lossdict_G['loss_G_GAN'] = loss_G_GAN
        # combine loss and calculate gradients
        loss_G = loss_G_GAN['total']*coefs['gan'] + loss_G_L1['total']*coefs['task'] + act_cost*coefs['clst'] + div_loss*coefs['sep']
        lossdict_G['loss_G'] = loss_G
        loss_G.backward()
        
        optimizer_G.step()
        
        
        for name in loss_name_G:
            for k, v in lossdict_G[name].items():
                losses_G[name][k].update(v.item())
        
        if idx % 25 == 0:
            for name in loss_name_G:
                logger.info(str(name))
                progress_G[name].display(idx, logger)

            # if cur_stg == 'two':
            for name in loss_name_D:
                logger.info(str(name))
                progress_D[name].display(idx, logger)
            logger.info('loss_D: {0:.4f}, loss_D_fake: {1:.4f}, loss_D_real:{2:.4f}'.format(\
                loss_D, loss_D_fake['total'], loss_D_real['total']))  
            logger.info('loss_G: {0:.4f}, loss_G_GAN: {1:.4f}, loss_G_L1:{2:.4f}, sep loss: {3:.4f}, cluster loss: {4:.4f}'.format(\
                    loss_G, loss_G_GAN['total']*coefs['gan'], loss_G_L1['total']*coefs['task'], div_loss*coefs['sep'], act_cost*coefs['clst']))

from utils.multitask_utils import get_three_channels
def evaluation(stage, net, val_loader, task_kwargs, logger):
    logger.info('evaluating.......')
    from evaluation.evaluate_utils import PerformanceMeter
    performance_meter = PerformanceMeter(task_kwargs)
    net.eval()
    for idx, batch in enumerate(tqdm(val_loader)):
        images = batch['image'].cuda(non_blocking=True)
        targets = {task: batch[task].cuda(non_blocking=True) for task in task_kwargs['task_names']}
        min_distances, pred, proto_activations_taskdict = net(images, stage)

        performance_meter.update({t: get_output(pred[t], t) for t in task_kwargs['task_names']}, 
                                 {t: targets[t] for t in task_kwargs['task_names']})

    train_img_dir = os.path.join(img_dir, 'res_epoch_{}'.format(epoch))
    pred = {t: get_three_channels(pred[t]) for t in task_kwargs['task_names']}
    eval_results = performance_meter.get_score(logger=logger.info, verbose = True)
    performance_meter.reset()

    return eval_results

# train the model
# from settings_gan import eval_per_epoch, num_stage1_epochs
logger.info('start training')
best_result = -1
from push import push_prototypes

# optimizer = torch.optim.SGD(ppnet_multi.parameters(), lr=3e-3, weight_decay=1e-3)

kld_loss_weight = coefs['kld']
cur_stg = 'one'

stage_one(ppnet_multi)
for epoch in range(start_epoch, num_train_epochs):

    logger.info('epoch: \t{0}'.format(epoch))
    #logger.info('lr: {}'.format(joint_optimizer.state_dict()['param_groups'][0]['lr']))
    if epoch < num_warm_epochs:
        warm_only(ppnet_multi)
        train_vanilla(stage=cur_stg, epoch=epoch, net=ppnet_multi, netD=netD_multi, criterionL1=criterionL1, criterionGAN=criterionGAN, train_loader=train_loader, \
            optimizer_G=warm_optimizer, optimizer_D=optimizer_D, task_kwargs=task_kwargs,kld_loss_weight=kld_loss_weight, logger=logger)
    elif epoch < num_stage1_epochs:
        # kld_loss_weight = coefs['kld'] * 0.0001
        joint(ppnet_multi)
        # joint_lr_scheduler.step()
        # logger.info(joint_optimizer.state_dict()['param_groups'][0]['lr'])
        train_vanilla(stage=cur_stg, epoch=epoch, net=ppnet_multi, netD=netD_multi, criterionL1=criterionL1, criterionGAN=criterionGAN, train_loader=train_loader, \
            optimizer_G=joint_optimizer, optimizer_D=optimizer_D, task_kwargs=task_kwargs, kld_loss_weight=kld_loss_weight, logger=logger)
    else: 
        if epoch == num_stage1_epochs:
            stage_two(ppnet_multi)
            ppnet_multi.module.change_stage()
        cur_stg = 'two'
        logger.info('cur_stage: \t{0}'.format(cur_stg))

        ## update lr #########
        old_lr = joint_optimizer.state_dict()['param_groups'][0]['lr']
        joint_lr_scheduler.step()
        lr = joint_optimizer.state_dict()['param_groups'][0]['lr']
        print('learning rate %.7f -> %.7f' % (old_lr, lr))
        #######################
        
        train_vanilla(stage=cur_stg, epoch=epoch, net=ppnet_multi, netD=netD_multi, criterionL1=criterionL1, criterionGAN=criterionGAN, train_loader=train_loader, \
            optimizer_G=joint_optimizer, optimizer_D=optimizer_D, task_kwargs=task_kwargs, kld_loss_weight=kld_loss_weight, logger=logger)

    
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
        if epoch % 20 == 0:
            print('Checkpoint ...')
            torch.save({'model': ppnet_multi.state_dict(), 'netD':netD_multi.state_dict(), 'lr_scheduler': joint_lr_scheduler.state_dict(), \
            'warm_optimizer': warm_optimizer.state_dict(), 'joint_optimizer': joint_optimizer.state_dict(),'optimizer_D':optimizer_D.state_dict(), \
            'epoch': epoch, 'eval_res': eval_res}, os.path.join(ckp_dir, 'epoch_{}.pt'.format(epoch)))

        cur_res = 0
        for name in task_kwargs['task_names']:
            cur_res += eval_res[name]['ssim']
        cur_res /= len(task_kwargs['task_names'])        
        if cur_res > best_result:
            best_result = cur_res
            best_epoch = epoch
            logger.info('\nbest_epoch: {}, best_result: {}\n'.format(best_epoch, best_result))
            torch.save({'model': ppnet_multi.state_dict(), 'netD':netD_multi.state_dict(), 'lr_scheduler': joint_lr_scheduler.state_dict(), \
        'warm_optimizer': warm_optimizer.state_dict(), 'joint_optimizer': joint_optimizer.state_dict(),'optimizer_D':optimizer_D.state_dict(), \
        'epoch': epoch, 'eval_res': eval_res, 'best_res':best_result}, os.path.join(ckp_dir, 'best.pt'))
    
    torch.save({'model': ppnet_multi.state_dict(), 'netD':netD_multi.state_dict(), 'lr_scheduler': joint_lr_scheduler.state_dict(), \
        'warm_optimizer': warm_optimizer.state_dict(), 'joint_optimizer': joint_optimizer.state_dict(),'optimizer_D':optimizer_D.state_dict(), \
        'epoch': epoch, 'eval_res': eval_res}, os.path.join(ckp_dir, 'last.pt'))


# logclose()

