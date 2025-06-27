import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import copy
import time

from tqdm import tqdm

from receptive_field import compute_rf_prototype
from helpers import makedir, find_high_activation_crop

# push each prototype to the nearest patch in the training set
def push_prototypes(dataloader, # pytorch dataloader (must be unnormalized in [0,1])
                    prototype_network_parallel, # pytorch network with prototype_vectors
                    task_names,
                    prototype_kwargs,
                    class_specific=True,
                    preprocess_input_function=None, # normalize if needed
                    prototype_layer_stride=1,
                    root_dir_for_saving_prototypes=None, # if not None, prototypes will be saved here
                    epoch_number=None, # if not provided, prototypes saved previously will be overwritten
                    prototype_img_filename_prefix=None,
                    prototype_self_act_filename_prefix=None,
                    proto_bound_boxes_filename_prefix=None,
                    save_prototype_class_identity=True, # which class the prototype image comes from
                    log=print,
                    prototype_activation_function_in_numpy=None):

    prototype_network_parallel.eval()
    log('\tpush')

    start = time.time()
    prototype_shape = prototype_network_parallel.module.prototype_shape # 15x128x1x1
    n_prototypes = prototype_network_parallel.module.num_prototypes # 15
    # saves the closest distance seen so far
    global_min_proto_dist = np.full(n_prototypes, np.inf)  # [inf,inf,inf,....]
    # saves the patch representation that gives the current smallest distance
    global_min_fmap_patches = np.zeros(
        [n_prototypes,
         prototype_shape[1],
         prototype_shape[2],
         prototype_shape[3]])

    '''
    proto_rf_boxes and proto_bound_boxes column:
    0: image index in the entire dataset
    1: height start index
    2: height end index
    3: width start index
    4: width end index
    5: (optional) class identity
    '''
    if save_prototype_class_identity:
        proto_rf_boxes = np.full(shape=[n_prototypes, 6],
                                    fill_value=-1)
        proto_bound_boxes = np.full(shape=[n_prototypes, 6],
                                            fill_value=-1)
    else:
        proto_rf_boxes = np.full(shape=[n_prototypes, 5],
                                    fill_value=-1)
        proto_bound_boxes = np.full(shape=[n_prototypes, 5],
                                            fill_value=-1)

    if root_dir_for_saving_prototypes != None:
        if epoch_number != None:
            proto_epoch_dir = os.path.join(root_dir_for_saving_prototypes,
                                           'epoch-'+str(epoch_number))
            makedir(proto_epoch_dir)
        else:
            proto_epoch_dir = root_dir_for_saving_prototypes
    else:
        proto_epoch_dir = None

    search_batch_size = dataloader.batch_size

    num_classes = prototype_network_parallel.module.num_classes

    for push_iter, search_batch in enumerate(tqdm(dataloader)):# (search_batch_input, search_y)
        '''
        start_index_of_search keeps track of the index of the image
        assigned to serve as prototype
        '''
        search_batch_input = search_batch['image']
        start_index_of_search_batch = push_iter * search_batch_size

        update_prototypes_on_batch(task_names,
                                   prototype_kwargs,
                                   search_batch_input,
                                   start_index_of_search_batch,
                                   prototype_network_parallel,
                                   global_min_proto_dist,
                                   global_min_fmap_patches,
                                   proto_rf_boxes,
                                   proto_bound_boxes,
                                   class_specific=class_specific,
                                   search_y=search_batch,
                                   num_classes=num_classes,
                                   preprocess_input_function=preprocess_input_function,
                                   prototype_layer_stride=prototype_layer_stride,
                                   dir_for_saving_prototypes=proto_epoch_dir,
                                   prototype_img_filename_prefix=prototype_img_filename_prefix,
                                   prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                                   prototype_activation_function_in_numpy=prototype_activation_function_in_numpy)

    if proto_epoch_dir != None and proto_bound_boxes_filename_prefix != None:
        np.save(os.path.join(proto_epoch_dir, proto_bound_boxes_filename_prefix + '-receptive_field' + str(epoch_number) + '.npy'),
                proto_rf_boxes) 
        # proto_rf_boxes: num_prototype x 6 记录了每个prototype对应的最好原型信息 [img_index,start_height,end_height,start_width,end_width, class]

        np.save(os.path.join(proto_epoch_dir, proto_bound_boxes_filename_prefix + str(epoch_number) + '.npy'),
                proto_bound_boxes)
        # proto_bound_boxes: num_prototype x 6 记录了每个prototype对应的最好原型distance激活信息 [img_index,start_height,end_height,start_width,end_width, class]

    log('\tExecuting push ...')
    prototype_update = np.reshape(global_min_fmap_patches,
                                  tuple(prototype_shape))
    prototype_network_parallel.module.prototype_vectors.data.copy_(torch.tensor(prototype_update, dtype=torch.float32).cuda())
    # prototype_network_parallel.cuda()
    
    # de-duplicate prototypes
    _, unique_index = np.unique(prototype_update, axis=0, return_index=True)
    duplicate_idx = [i for i in range(prototype_network_parallel.module.num_prototypes) if i not in unique_index]
    log(f'Removing {len(duplicate_idx)} duplicate prototypes.')
    prototype_network_parallel.module.prune_prototypes(duplicate_idx)
    os.makedirs(proto_epoch_dir, exist_ok=True)
    import json
    with open(os.path.join(proto_epoch_dir, 'unique_prototypes.json'), 'w') as fp:
        json.dump([int(i) for i in sorted(unique_index)], fp)
    
    end = time.time()
    log('\tpush time: \t{0}'.format(end -  start))

# update each prototype for current search batch
def update_prototypes_on_batch(task_names,
                               prototype_kwargs,
                               search_batch_input,
                               start_index_of_search_batch,
                               prototype_network_parallel,
                               global_min_proto_dist, # this will be updated
                               global_min_fmap_patches, # this will be updated
                               proto_rf_boxes, # this will be updated
                               proto_bound_boxes, # this will be updated
                               class_specific=True,
                               search_y=None, # required if class_specific == True
                               num_classes=None, # required if class_specific == True
                               preprocess_input_function=None,
                               prototype_layer_stride=1,
                               dir_for_saving_prototypes=None,
                               prototype_img_filename_prefix=None,
                               prototype_self_act_filename_prefix=None,
                               prototype_activation_function_in_numpy=None):

    prototype_network_parallel.eval()

    if preprocess_input_function is not None:
        # print('preprocessing input for pushing ...')
        # search_batch = copy.deepcopy(search_batch_input)
        search_batch = preprocess_input_function(search_batch_input)

    else:
        search_batch = search_batch_input

    with torch.no_grad():
        search_batch = search_batch.cuda()
        # this computation currently is not parallelized
        protoL_input_torch, proto_dist_torch = prototype_network_parallel.module.push_forward(search_batch)
        # protoL_input_torch: N x 128 x 7 x 7    # proto_dist_torch: N x num_prototype x 7 x 7
    protoL_input_ = np.copy(protoL_input_torch.detach().cpu().numpy())
    proto_dist_ = np.copy(proto_dist_torch.detach().cpu().numpy())

    del protoL_input_torch, proto_dist_torch

    if class_specific:
        class_to_img_index_dict = {key: [] for key in range(num_classes)} # 记录每个类对应的数据index
        # img_y is the image's integer label
        for img_index, img_y in enumerate(search_y):
            img_label = img_y.item()
            class_to_img_index_dict[img_label].append(img_index)

    prototype_shape = prototype_network_parallel.module.prototype_shape
    n_prototypes = prototype_shape[0]
    proto_h = prototype_shape[2]
    proto_w = prototype_shape[3]
    max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

    for j in range(n_prototypes):
        #if n_prototypes_per_class != None:
        if class_specific:
            # target_class is the class of the class_specific prototype # 当前prototype属于哪一个类
            target_class = torch.argmax(prototype_network_parallel.module.prototype_class_identity[j]).item()
            # if there is not images of the target_class from this batch
            # we go on to the next prototype
            if len(class_to_img_index_dict[target_class]) == 0:
                continue
            proto_dist_j = proto_dist_[class_to_img_index_dict[target_class]][:,j,:,:]
            # proto_dist_[class_to_img_index_dict[target_class]] 取出batch中对应某些个数据的distance结果
            # proto_dist_j 用来取出第j个prototype属于的类的img的distance的对应的到第j个原型的距离信息
        else:
            # if it is not class specific, then we will search through
            # every example
            proto_dist_j = proto_dist_[:,j,:,:]

        batch_min_proto_dist_j = np.amin(proto_dist_j)
        if batch_min_proto_dist_j < global_min_proto_dist[j]:
            batch_argmin_proto_dist_j = \
                list(np.unravel_index(np.argmin(proto_dist_j, axis=None),
                                      proto_dist_j.shape))  # 获得第j个prototype的最小值在数组中的位置
            if class_specific:
                '''
                change the argmin index from the index among
                images of the target class to the index in the entire search
                batch
                '''  # [0]位置是batch中数据的index
                batch_argmin_proto_dist_j[0] = class_to_img_index_dict[target_class][batch_argmin_proto_dist_j[0]]

            # retrieve the corresponding feature map patch
            img_index_in_batch = batch_argmin_proto_dist_j[0]
            fmap_height_start_index = batch_argmin_proto_dist_j[1] * prototype_layer_stride
            fmap_height_end_index = fmap_height_start_index + proto_h
            fmap_width_start_index = batch_argmin_proto_dist_j[2] * prototype_layer_stride
            fmap_width_end_index = fmap_width_start_index + proto_w
            # 上面的信息获取了索引为index的数据，且确定 patch 是位于特征图的哪一个位置（左上角坐标和左下角坐标）
            batch_min_fmap_patch_j = protoL_input_[img_index_in_batch,
                                                   :,
                                                   fmap_height_start_index:fmap_height_end_index,
                                                   fmap_width_start_index:fmap_width_end_index]
            # 取出了该index数据经过卷积后对应的feature_map，再根据上面定位的patch信息将patch取出来
            global_min_proto_dist[j] = batch_min_proto_dist_j
            global_min_fmap_patches[j] = batch_min_fmap_patch_j # 128 x 1 x 1 #将这个最近的patch更新为prototype表示的1x1卷积
            
            # get the receptive field boundary of the image patch
            # that generates the representation
            protoL_rf_info = prototype_network_parallel.module.proto_layer_rf_info
            rf_prototype_j = compute_rf_prototype(search_batch.size(2), batch_argmin_proto_dist_j, protoL_rf_info)
            # 上面计算了所在patch对应原图中的位置
            # get the whole image
            original_img_j = search_batch_input[rf_prototype_j[0]]  # 获取到对应index的input图像
            original_img_j = original_img_j.numpy()
            original_img_j = np.transpose(original_img_j, (1, 2, 0))
            original_img_size = original_img_j.shape[0]
            
            # crop out the receptive field 获取到patch对应到原图中的区域
            rf_img_j = original_img_j[rf_prototype_j[1]:rf_prototype_j[2],
                                      rf_prototype_j[3]:rf_prototype_j[4], :]  # 得到原图中patch的区域
            
            # save the prototype receptive field information # [img_index, rf_start_height_index, rf_end_height_index, rf_start_width_index, rf_end_width_index]
            proto_rf_boxes[j, 0] = rf_prototype_j[0] + start_index_of_search_batch
            proto_rf_boxes[j, 1] = rf_prototype_j[1]
            proto_rf_boxes[j, 2] = rf_prototype_j[2]
            proto_rf_boxes[j, 3] = rf_prototype_j[3]
            proto_rf_boxes[j, 4] = rf_prototype_j[4]
            if proto_rf_boxes.shape[1] == 6 and search_y is not None:
                proto_rf_boxes[j, 5] = search_y[rf_prototype_j[0]].item() # 第五个位置记录该prototype属于的类别信息
            # proto_rf_boxes: num_prototype x 6 记录了每个prototype对应的最好原型信息 [img_index,start_height,end_height,start_width,end_width, class]
            # find the highly activated region of the original image
            proto_dist_img_j = proto_dist_[img_index_in_batch, j, :, :] # 获得对应的距离信息，下面再接一个激活（距离越小，经过激活值越大）
            if prototype_network_parallel.module.prototype_activation_function == 'log':
                proto_act_img_j = np.log((proto_dist_img_j + 1) / (proto_dist_img_j + prototype_network_parallel.module.epsilon))
            elif prototype_network_parallel.module.prototype_activation_function == 'linear':
                proto_act_img_j = max_dist - proto_dist_img_j
            else:
                proto_act_img_j = prototype_activation_function_in_numpy(proto_dist_img_j)
            upsampled_act_img_j = cv2.resize(proto_act_img_j, dsize=(original_img_size, original_img_size),
                                             interpolation=cv2.INTER_CUBIC)  # 直接将激活后的distance，resize到原图
            proto_bound_j = find_high_activation_crop(upsampled_act_img_j) # lower_y, upper_y+1, lower_x, upper_x+1
            # crop out the image patch with high activation as prototype image 
            proto_img_j = original_img_j[proto_bound_j[0]:proto_bound_j[1],
                                         proto_bound_j[2]:proto_bound_j[3], :]  # 得到激活图

            # save the prototype boundary (rectangular boundary of highly activated region)
            proto_bound_boxes[j, 0] = proto_rf_boxes[j, 0]
            proto_bound_boxes[j, 1] = proto_bound_j[0]
            proto_bound_boxes[j, 2] = proto_bound_j[1]
            proto_bound_boxes[j, 3] = proto_bound_j[2]
            proto_bound_boxes[j, 4] = proto_bound_j[3]
            if proto_bound_boxes.shape[1] == 6 and search_y is not None:
                proto_bound_boxes[j, 5] = search_y[rf_prototype_j[0]].item()
            # proto_bound_boxes: num_prototype x 6 记录了每个prototype对应的最好原型distance激活信息 [img_index,start_height,end_height,start_width,end_width, class]
            
            
            task_prefix = ''
            if j < len(task_names)*prototype_kwargs['num_per_task']:
                task_name=task_names[j // prototype_kwargs['num_per_task']]
                task_prefix = '-task[{}]_'.format(task_name)
                task_j = j % prototype_kwargs['num_per_task']
            else: 
                task_prefix = '-share_'
                task_j = j % prototype_kwargs['num_per_task']            
            
            if dir_for_saving_prototypes is not None:
                if prototype_self_act_filename_prefix is not None:
                    # save the numpy array of the prototype self activation
                    np.save(os.path.join(dir_for_saving_prototypes,
                                         prototype_self_act_filename_prefix + task_prefix + str(task_j) + '.npy'),
                            proto_act_img_j) # 保存proto_act_img_j: distance经过激活的值 7x7
                if prototype_img_filename_prefix is not None:
                    # save the whole image containing the prototype as png
                    plt.imsave(os.path.join(dir_for_saving_prototypes,
                                            prototype_img_filename_prefix + '-original' + task_prefix + str(task_j) + '.png'),
                               original_img_j,
                               vmin=0.0,
                               vmax=1.0)
                    # overlay (upsampled) self activation on original image and save the result
                    rescaled_act_img_j = upsampled_act_img_j - np.amin(upsampled_act_img_j)
                    rescaled_act_img_j = rescaled_act_img_j / np.amax(rescaled_act_img_j)
                    heatmap = cv2.applyColorMap(np.uint8(255*rescaled_act_img_j), cv2.COLORMAP_JET)
                    heatmap = np.float32(heatmap) / 255
                    heatmap = heatmap[...,::-1]
                    overlayed_original_img_j = 0.5 * original_img_j + 0.3 * heatmap
                    plt.imsave(os.path.join(dir_for_saving_prototypes,
                                            prototype_img_filename_prefix + '-original_with_self_act' + task_prefix + str(task_j) + '.png'),
                               overlayed_original_img_j,
                               vmin=0.0,
                               vmax=1.0)
                    
                    # if different from the original (whole) image, save the prototype receptive field as png
                    if rf_img_j.shape[0] != original_img_size or rf_img_j.shape[1] != original_img_size:
                        plt.imsave(os.path.join(dir_for_saving_prototypes,
                                                prototype_img_filename_prefix + '-receptive_field' + task_prefix + str(task_j) + '.png'),
                                   rf_img_j,
                                   vmin=0.0,
                                   vmax=1.0)
                        overlayed_rf_img_j = overlayed_original_img_j[rf_prototype_j[1]:rf_prototype_j[2],
                                                                      rf_prototype_j[3]:rf_prototype_j[4]]
                        plt.imsave(os.path.join(dir_for_saving_prototypes,
                                                prototype_img_filename_prefix + '-receptive_field_with_self_act' + task_prefix + str(task_j) + '.png'),
                                   overlayed_rf_img_j,
                                   vmin=0.0,
                                   vmax=1.0)
                    
                    # save the prototype image (highly activated region of the whole image)
                    plt.imsave(os.path.join(dir_for_saving_prototypes,
                                            prototype_img_filename_prefix + task_prefix + str(task_j) + '.png'),
                               proto_img_j,
                               vmin=0.0,
                               vmax=1.0)
                if j < len(task_names)*prototype_kwargs['num_per_task']:
                    # mask save
                    original_mask_j = search_y[task_name][rf_prototype_j[0]]
                    original_mask_j = np.transpose(original_mask_j.to('cpu').numpy(), (1, 2, 0))
                    rf_mask_j = original_mask_j[rf_prototype_j[1]:rf_prototype_j[2],
                                            rf_prototype_j[3]:rf_prototype_j[4], :]
                    if task_name.startswith('semseg'):
                        plt.imsave(os.path.join(dir_for_saving_prototypes, 'prototype-mask-origin-{}.png'.format(task_prefix + str(task_j))), original_mask_j[:,:,0]*100)
                        plt.imsave(os.path.join(dir_for_saving_prototypes, 'prototype-mask-receptive_field-{}.png'.format(task_prefix + str(task_j))), rf_mask_j[:,:,0]*100)
                    elif task_name.startswith('gen'):
                        plt.imsave(os.path.join(dir_for_saving_prototypes, 'prototype-target-origin-{}.png'.format(task_prefix + str(task_j))), original_mask_j)
                        plt.imsave(os.path.join(dir_for_saving_prototypes, 'prototype-target-receptive_field-{}.png'.format(task_prefix + str(task_j))), rf_mask_j)
                    else: 
                        raise Exception('wrong taskname!')
                    
                else: 
                    for task_name in task_names:
                        # mask save
                        original_mask_j = search_y[task_name][rf_prototype_j[0]]
                        original_mask_j = np.transpose(original_mask_j.to('cpu').numpy(), (1, 2, 0))
                        rf_mask_j = original_mask_j[rf_prototype_j[1]:rf_prototype_j[2],
                                                rf_prototype_j[3]:rf_prototype_j[4], :]
                        if task_name.startswith('semseg'):
                            plt.imsave(os.path.join(dir_for_saving_prototypes, 'prototype-mask-origin-{}.png'.format(task_prefix + str(task_j) + '-task[{}]'.format(task_name))), original_mask_j[:,:,0]*100)
                            plt.imsave(os.path.join(dir_for_saving_prototypes, 'prototype-mask-receptive_field-{}.png'.format(task_prefix + str(task_j) + '-task[{}]'.format(task_name))), rf_mask_j[:,:,0]*100)
                        elif task_name.startswith('gen'):
                            plt.imsave(os.path.join(dir_for_saving_prototypes, 'prototype-mask-origin-{}.png'.format(task_prefix + str(task_j) + '-task[{}]'.format(task_name))), original_mask_j)
                            plt.imsave(os.path.join(dir_for_saving_prototypes, 'prototype-mask-receptive_field-{}.png'.format(task_prefix + str(task_j) + '-task[{}]'.format(task_name))), rf_mask_j)
                        else: 
                            raise Exception('wrong taskname!')

                
    if class_specific:
        del class_to_img_index_dict
