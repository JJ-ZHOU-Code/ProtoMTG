import torch
from tqdm import tqdm 
from utils.multitask_utils import get_loss_meters, ProgressMeter, get_output

def train_vanilla(stage, epoch, net, criterion, train_loader, optimizer, task_kwargs, prototype_kwargs, coefs, logger):
    
    losses = get_loss_meters(task_kwargs)
    progress = ProgressMeter(len(train_loader),
        [v for v in losses.values()], prefix="Epoch: [{}]".format(epoch))
    
    net.train()
    for idx, batch in enumerate(train_loader):
        images = batch['image'].cuda(non_blocking=True)
        targets = {task: batch[task].cuda(non_blocking=True) for task in task_kwargs['task_names']}

        min_distances, pred, proto_activations_taskdict = net(images, stage)
        # cluster loss
        act_cost = torch.mean(min_distances).item()
        # kld loss
        if prototype_kwargs['num_shared']:
            names = task_kwargs['task_names'] + ['share']
        else:
            names = task_kwargs['task_names']
        div_loss = compute_div(names, proto_activations_taskdict)
            
        loss_dict = criterion(pred, targets)
        
        for k, v in loss_dict.items():
            losses[k].update(v.item())
        
        # Backward
        optimizer.zero_grad()
        # loss_dict['total'].backward()
        total_loss = loss_dict['total']*coefs['task'] + div_loss*coefs['div'] + act_cost*coefs['act'] # + seperate_cost_dict['total']
        total_loss.backward()
        optimizer.step()
        # lr_scheduler.step()
        if idx % 25 == 0:
            progress.display(idx, logger)
            logger.info('diversity loss: {0:.4f}, activation loss: {1:.4f} tatoal_loss: {2:.4f}'.format(act_cost*coefs['act'],div_loss*coefs['div'], total_loss))


def evaluation(stage, net, val_loader, task_kwargs, logger):
    logger.info('evaluating.......')
    from evaluation.evaluate_utils import PerformanceMeter
    performance_meter = PerformanceMeter(task_kwargs)
    net.eval()
    for i, batch in enumerate(tqdm(val_loader)):
        images = batch['image'].cuda(non_blocking=True)
        targets = {task: batch[task].cuda(non_blocking=True) for task in task_kwargs['task_names']}
        min_distances, pred, proto_activations_taskdict = net(images, stage)

        performance_meter.update({t: get_output(pred[t], t) for t in task_kwargs['task_names']}, 
                                 {t: targets[t] for t in task_kwargs['task_names']})

    eval_results = performance_meter.get_score(logger=logger.info, verbose = True)
    return eval_results



def compute_kld(kldloss_name, proto_activations_taskdict):
    kld_loss = []
    for task in kldloss_name:
        proto_activation_task = proto_activations_taskdict[task]  # batch, num_proto, H, W
        proto_activation_task = proto_activation_task.permute(1, 0, 2, 3) # num_proto, batch, H, W, 
        proto_activation_task = proto_activation_task.reshape(proto_activation_task.shape[0], proto_activation_task.shape[1], -1)

        log_cls_activations = torch.nn.functional.log_softmax(proto_activation_task, dim=-1)
        for i in range(log_cls_activations.shape[0]):
            if log_cls_activations.shape[0] < 2 or log_cls_activations.shape[-1] < 2:
                continue

            log_p1_scores = log_cls_activations[i]
            for j in range(i + 1, log_cls_activations.shape[0]):
                log_p2_scores = log_cls_activations[j]
                # add kld1 and kld2 to make 'symmetrical kld'
                kld1 = torch.nn.functional.kl_div(log_p1_scores, log_p2_scores,
                                                log_target=True, reduction='batchmean')
                kld2 = torch.nn.functional.kl_div(log_p2_scores, log_p1_scores,
                                                log_target=True, reduction='batchmean')
                kld = (kld1 + kld2) / 2.0
                kld_loss.append(kld)
                
    if len(kld_loss) > 0:
        kld_loss = torch.stack(kld_loss)
        # to make 'loss' (lower == better) take exponent of the negative (maximum value is 1.0, for KLD == 0.0)
        kld_loss = torch.exp(-kld_loss)
        kld_loss = torch.mean(kld_loss)
    else:
        kld_loss = 0.0
        
    return kld_loss

def compute_js(kldloss_name, proto_activations_taskdict, js_loss_dict):
    
    for task in kldloss_name:
        kld_loss = []
        proto_activation_task = proto_activations_taskdict[task]  # batch, num_proto, H, W
        proto_activation_task = proto_activation_task.permute(1, 0, 2, 3) # num_proto, batch, H, W, 
        proto_activation_task = proto_activation_task.reshape(proto_activation_task.shape[0], proto_activation_task.shape[1], -1)

        # log_cls_activations = torch.nn.functional.log_softmax(proto_activation_task, dim=-1)
        log_cls_activations = proto_activation_task
        for i in range(proto_activation_task.shape[0]):
            if proto_activation_task.shape[0] < 2 or proto_activation_task.shape[-1] < 2:
                continue

            p1_scores = proto_activation_task[i]
            for j in range(i + 1, proto_activation_task.shape[0]):
                p2_scores = proto_activation_task[j]
                middle_score = (p1_scores + p1_scores) / 2
                log_middle_score = torch.nn.functional.log_softmax(middle_score, dim=-1)
                log_p1_scores = torch.nn.functional.log_softmax(p1_scores, dim=-1)
                log_p2_scores = torch.nn.functional.log_softmax(p2_scores, dim=-1)
                # add kld1 and kld2 to make 'symmetrical kld'
                kld1 = torch.nn.functional.kl_div(log_p1_scores, log_middle_score,
                                                log_target=True, reduction='batchmean')
                kld2 = torch.nn.functional.kl_div(log_p2_scores, log_middle_score,
                                                log_target=True, reduction='batchmean')
                kld = (kld1 + kld2) / 2.0
                # kld_loss.append(kld)
                js_loss_dict[task].append(kld)
    # if len(kld_loss) > 0:
    #     kld_loss = torch.stack(kld_loss)
    #     kld_loss = torch.mean(kld_loss)
    #     # to make 'loss' (lower == better) take exponent of the negative (maximum value is 1.0, for KLD == 0.0)
    #     # kld_loss = torch.exp(-kld_loss)
    #     # kld_loss = torch.mean(kld_loss)
    # else:
    #     kld_loss = 0.0
        
    # return kld_loss

def compute_div(seploss_name, proto_activations_taskdict):
    sep_loss = []
    for task in seploss_name:
        proto_activation_task = proto_activations_taskdict[task]  # batch, num_proto, H, W
        ###    cosine similarity start  ##
        pa = proto_activation_task.reshape(proto_activation_task.shape[0], proto_activation_task.shape[1], -1) # batch, num, HW
        pa = torch.nn.functional.normalize(pa, p=2, dim=-1)
        a = pa.unsqueeze(2) # batch, num_proto, 1, HW
        b = pa.unsqueeze(1) # batch, 1, num_proto, HW
        cos_diff = torch.sum(torch.mul(a,b), dim=-1)  # batch, num_proto, num_proto
        # cos_diff_square = torch.mean(cos_diff**2, dim=[-2, -1])
        cos_diff = torch.mean(cos_diff, dim=[-2, -1]) # batch, 
        sep_loss.append(cos_diff)

    if len(sep_loss) > 0:
        sep_loss = torch.stack(sep_loss)
        # to make 'loss' (lower == better) take exponent of the negative (maximum value is 1.0, for KLD == 0.0)
        sep_loss = torch.mean(sep_loss)
    else:
        sep_loss = 0.0
        
    return sep_loss
        
        
def warm_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    log('\twarm')

def stage_decoder(model, log=print):
    for p in model.module.parameters():
        p.requires_grad = False
    for p in model.module.decoders.parameters():
        p.requires_grad = True
    log('\stage two')


def joint(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = True
    log('\tjoint')


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters()) #model.parameters().numel()
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad