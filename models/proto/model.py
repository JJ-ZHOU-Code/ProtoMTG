import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

from resnet_features import resnet18_features, resnet34_features, resnet50_features, resnet101_features, resnet152_features
from densenet_features import densenet121_features, densenet161_features, densenet169_features, densenet201_features
from vgg_features import vgg11_features, vgg11_bn_features, vgg13_features, vgg13_bn_features, vgg16_features, vgg16_bn_features,\
                         vgg19_features, vgg19_bn_features

from receptive_field import compute_proto_layer_rf_info_v2
from models.decoder import UnetDecoder
base_architecture_to_features = {'resnet18': resnet18_features,
                                 'resnet34': resnet34_features,
                                 'resnet50': resnet50_features,
                                 'resnet101': resnet101_features,
                                 'resnet152': resnet152_features,
                                 'densenet121': densenet121_features,
                                 'densenet161': densenet161_features,
                                 'densenet169': densenet169_features,
                                 'densenet201': densenet201_features,
                                 'vgg11': vgg11_features,
                                 'vgg11_bn': vgg11_bn_features,
                                 'vgg13': vgg13_features,
                                 'vgg13_bn': vgg13_bn_features,
                                 'vgg16': vgg16_features,
                                 'vgg16_bn': vgg16_bn_features,
                                 'vgg19': vgg19_features,
                                 'vgg19_bn': vgg19_bn_features}

class ChannelSELayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*

    """

    def __init__(self, num_channels, reduction_ratio=2):
        """

        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor
        """
        batch_size, num_channels, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return fc_out_2, output_tensor


class PMTG(nn.Module):

    def __init__(self, task_kwargs, prototype_kwargs, features, decoder, warm_decoder, img_size, prototype_shape,
                 proto_layer_rf_info, num_classes, init_weights=True,
                 prototype_activation_function='log',
                 add_on_layers_type='bottleneck', proto_combine_mode='sum', weighted=True):

        super(PMTG, self).__init__()
        self.tasks = task_kwargs['task_names']
        self.prototype_per_task = prototype_kwargs['num_per_task']
        self.shared_prototype = prototype_kwargs['num_shared']
        self.combine_prtonum = self.prototype_per_task + self.shared_prototype
        self.num_prototypes = prototype_shape[0]
        assert self.prototype_per_task * len(self.tasks) + self.shared_prototype == self.num_prototypes
        
        self.img_size = img_size
        self.prototype_shape = prototype_shape
        self.num_classes = num_classes
        self.epsilon = 1e-5
        
        self.proto_combine_mode = proto_combine_mode
        
        # prototype_activation_function could be 'log', 'linear',
        # or a generic function that converts distance to similarity score
        self.prototype_activation_function = prototype_activation_function

        '''
        Here we are initializing the class identities of the prototypes
        Without domain specific knowledge we allocate the same number of
        prototypes for each class
        '''
        # assert(self.num_prototypes % self.num_classes == 0)
        # # a onehot indication matrix for each prototype's class identity
        # self.prototype_class_identity = torch.zeros(self.num_prototypes,
        #                                             self.num_classes)

        # num_prototypes_per_class = self.num_prototypes // self.num_classes
        # for j in range(self.num_prototypes):
        #     self.prototype_class_identity[j, j // num_prototypes_per_class] = 1

        self.proto_layer_rf_info = proto_layer_rf_info

        # this has to be named features to allow the precise loading
        self.features = features

        features_name = str(self.features).upper()
        if features_name.startswith('VGG') or features_name.startswith('RES'):
            first_add_on_layer_in_channels = \
                [i for i in features.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
        elif features_name.startswith('DENSE'):
            first_add_on_layer_in_channels = \
                [i for i in features.modules() if isinstance(i, nn.BatchNorm2d)][-1].num_features
        else:
            raise Exception('other base base_architecture NOT implemented')

        if add_on_layers_type == 'bottleneck':
            add_on_layers = []
            current_in_channels = first_add_on_layer_in_channels
            while (current_in_channels > self.prototype_shape[1]) or (len(add_on_layers) == 0):
                current_out_channels = max(self.prototype_shape[1], (current_in_channels // 2))
                add_on_layers.append(nn.Conv2d(in_channels=current_in_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                add_on_layers.append(nn.ReLU())
                add_on_layers.append(nn.Conv2d(in_channels=current_out_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                if current_out_channels > self.prototype_shape[1]:
                    add_on_layers.append(nn.ReLU())
                else:
                    assert(current_out_channels == self.prototype_shape[1])
                    add_on_layers.append(nn.Sigmoid())
                current_in_channels = current_in_channels // 2
            self.add_on_layers = nn.Sequential(*add_on_layers)
        else:
            self.add_on_layers = nn.Sequential(
                nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=self.prototype_shape[1], kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=self.prototype_shape[1], out_channels=self.prototype_shape[1], kernel_size=1),
                nn.Sigmoid()
                )
        
        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape),
                                              requires_grad=True)

        # do not make this just a tensor,
        # since it will not be moved automatically to gpu
        self.ones = nn.Parameter(torch.ones(self.prototype_shape),
                                 requires_grad=False)

        self.weighted = weighted
        self.selayers = ChannelSELayer(num_channels=self.prototype_per_task + self.shared_prototype)
            
        self.decoders = decoder
        self.warm_decoder = warm_decoder

        # self.last_layer = nn.Linear(self.num_prototypes, self.num_classes,
        #                             bias=False) # do not use bias

        if init_weights:
            self._initialize_weights()

    def conv_features(self, x):
        '''
        the feature input to prototype layer
        '''
        features = self.features(x)
        x = self.add_on_layers(features[-1])
        return features, x

    @staticmethod
    def _weighted_l2_convolution(input, filter, weights):
        '''
        input of shape N * c * h * w
        filter of shape P * c * h1 * w1
        weight of shape P * c * h1 * w1
        '''
        input2 = input ** 2
        input_patch_weighted_norm2 = F.conv2d(input=input2, weight=weights)

        filter2 = filter ** 2
        weighted_filter2 = filter2 * weights
        filter_weighted_norm2 = torch.sum(weighted_filter2, dim=(1, 2, 3))
        filter_weighted_norm2_reshape = filter_weighted_norm2.view(-1, 1, 1)

        weighted_filter = filter * weights
        weighted_inner_product = F.conv2d(input=input, weight=weighted_filter)

        # use broadcast
        intermediate_result = \
            - 2 * weighted_inner_product + filter_weighted_norm2_reshape
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(input_patch_weighted_norm2 + intermediate_result)

        return distances

    def _l2_convolution(self, x):
        '''
        apply self.prototype_vectors as l2-convolution filters on input x
        '''
        x2 = x ** 2 
        x2_patch_sum = F.conv2d(input=x2, weight=self.ones) 

        p2 = self.prototype_vectors ** 2 
        p2 = torch.sum(p2, dim=(1, 2, 3)) 
       
        p2_reshape = p2.view(-1, 1, 1)

        xp = F.conv2d(input=x, weight=self.prototype_vectors) 
        intermediate_result = - 2 * xp + p2_reshape 

        distances = F.relu(x2_patch_sum + intermediate_result)
        return distances

    def prototype_distances(self, x):
        '''
        x is the raw input
        '''
        features, conv_features = self.conv_features(x)
        distances = self._l2_convolution(conv_features)
        return features, distances

    def distance_2_similarity(self, distances):
        if self.prototype_activation_function == 'log':
            return torch.log((distances + 1) / (distances + self.epsilon))
        elif self.prototype_activation_function == 'linear':
            return -distances
        else:
            return self.prototype_activation_function(distances)

    def forward(self, x, stage='two', return_weights=False):
        features, distances = self.prototype_distances(x) 
        '''
        we cannot refactor the lines below for similarity scores
        because we need to return min_distances
        '''
        # global min pooling 
        min_distances = -F.max_pool2d(-distances,
                                      kernel_size=(distances.size()[2],
                                                   distances.size()[3]))
        min_distances = min_distances.view(-1, self.num_prototypes) # N x num_prototype
        max_distances = F.max_pool2d( distances,
                                      kernel_size=(distances.size()[2],
                                                   distances.size()[3]))
        max_distances = max_distances.view(-1, self.num_prototypes)
        distances_gap = -torch.log(1-min_distances/max_distances + self.epsilon)
 
        prototype_activations = self.distance_2_similarity(distances)        
        b, n, h, w = prototype_activations.shape
        # update features
        proto_activations_taskdict = {self.tasks[i]: prototype_activations[:,i*self.prototype_per_task:(i+1)*self.prototype_per_task,:,:] for i in range(len(self.tasks))}
        if self.shared_prototype == 0:
            # proto_activations_taskdict['share'] = None
            combine_proto_activations_taskdict = proto_activations_taskdict
        else: 
            proto_activations_taskdict['share'] = prototype_activations[:,-self.shared_prototype:,:,:]     
            combine_proto_activations_taskdict = {task: torch.concat([proto_activations_taskdict[task],proto_activations_taskdict['share']], dim=1) for task in self.tasks}
        weights_taskdict = {}
        if self.weighted: 
            for task in self.tasks:
                weights_taskdict[task], combine_proto_activations_taskdict[task] = self.selayers(combine_proto_activations_taskdict[task])
                
        # activated_features task_specific+shared
        if self.proto_combine_mode == 'concat':
            # concat
            activation_last_features_taskdict = {task: torch.concat([features[-1],combine_proto_activations_taskdict[task]],dim=1) for task in self.tasks}
            activation_features_taskdict = {task: features[:-1] + [activation_last_features_taskdict[task]] for task in self.tasks}
            
            tasks_outputs = {task: self.decoders[task](activation_features_taskdict[task]) for task in self.tasks}

        elif self.proto_combine_mode == 'sum':
            # sum      
            combine_proto_activations_taskdict = {task: combine_proto_activations_taskdict[task].sum(dim=1).view(b,1,h,w) for task in self.tasks}
            # multiply
            activation_last_features_taskdict = {task: features[-1]*combine_proto_activations_taskdict[task] for task in self.tasks}
            activation_features_taskdict = {task: features[:-1] + [activation_last_features_taskdict[task]] for task in self.tasks}
            
            if stage == 'one':
                tasks_outputs = {task: self.warm_decoder(activation_features_taskdict[task]) for task in self.tasks}
            else: 
                tasks_outputs = {task: self.decoders[task](activation_features_taskdict[task]) for task in self.tasks}
        elif self.proto_combine_mode == 'concatcascade':
            # concat
            tasks_outputs = {task: self.decoders[task](features, combine_proto_activations_taskdict[task]) for task in self.tasks}
        elif self.proto_combine_mode == 'proto_as_feature':
            activation_last_features_taskdict = combine_proto_activations_taskdict
            activation_features_taskdict = {task: features[:-1] + [activation_last_features_taskdict[task]] for task in self.tasks}
            tasks_outputs = {task: self.decoders[task](activation_features_taskdict[task]) for task in self.tasks}

        if return_weights: 
            return distances,  tasks_outputs, proto_activations_taskdict, weights_taskdict, combine_proto_activations_taskdict
        return distances_gap, tasks_outputs, proto_activations_taskdict

    def push_forward(self, x):
        '''this method is needed for the pushing operation'''
        features, conv_features = self.conv_features(x)
        distances = self._l2_convolution(conv_features)
        return conv_features, distances
    
    
    def change_stage(self):
        for task in self.tasks:
            self.decoders[task].load_state_dict(self.warm_decoder.state_dict())
            

    def prune_prototypes(self, prototypes_to_prune):
        '''
        prototypes_to_prune: a list of indices each in
        [0, current number of prototypes - 1] that indicates the prototypes to
        be removed
        '''
        prototypes_to_keep = list(set(range(self.num_prototypes)) - set(prototypes_to_prune))
        
        self.prototype_vectors_tmp = self.prototype_vectors
        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape).to('cuda'),
                                              requires_grad=True)
        self.prototype_vectors.data[prototypes_to_keep, ...] = self.prototype_vectors_tmp.data[prototypes_to_keep, ...]

    def __repr__(self):
        # PPNet(self, features, img_size, prototype_shape,
        # proto_layer_rf_info, num_classes, init_weights=True):
        rep = (
            'PPNet(\n'
            '\tfeatures: {},\n'
            '\timg_size: {},\n'
            '\tprototype_shape: {},\n'
            '\tproto_layer_rf_info: {},\n'
            '\tnum_classes: {},\n'
            '\tepsilon: {}\n'
            ')'
        )

        return rep.format(self.features,
                          self.img_size,
                          self.prototype_shape,
                          self.proto_layer_rf_info,
                          self.num_classes,
                          self.epsilon)


    def _initialize_weights(self):
        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Conv2d):
                # every init technique has an underscore _ in the name
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class construct_PMTG(PMTG):
    def __init__(self,
        base_architecture, 
        task_kwargs,
        prototype_kwargs,
        pretrained=True, 
        img_size=224,
        prototype_shape=(2000, 512, 1, 1), 
        num_classes=200,
        prototype_activation_function='log',
        add_on_layers_type='bottleneck',
        encoder_depth: int = 5,
        decoder_use_batchnorm = True,
        decoder_channels = (256, 128, 64, 32, 16), # (1024, 512, 256, 128, 64)
        decoder_attention_type = None,
        proto_combine_mode='sum',
        weighted=True,
        skip = False,
    ):
        # super().__init__()

        features_encoder = base_architecture_to_features[base_architecture](pretrained=pretrained)
        layer_filter_sizes, layer_strides, layer_paddings = features_encoder.conv_info()
        proto_layer_rf_info = compute_proto_layer_rf_info_v2(img_size=img_size,
                                                            layer_filter_sizes=layer_filter_sizes,
                                                            layer_strides=layer_strides,
                                                            layer_paddings=layer_paddings,
                                                            prototype_kernel_size=prototype_shape[2])

        proto_channels = 0
        if proto_combine_mode == 'concat':
            features_encoder._out_channels = list(features_encoder._out_channels)
            features_encoder._out_channels[-1] = features_encoder._out_channels[-1] + prototype_kwargs['num_per_task'] + prototype_kwargs['num_shared']
        if proto_combine_mode == 'concatcascade':
            proto_channels = prototype_kwargs['num_per_task'] + prototype_kwargs['num_shared']
            # features_encoder._out_channels = features_encoder._out_channels + prototype_kwargs['num_per_task'] + prototype_kwargs['num_shared']
        if proto_combine_mode == 'proto_as_feature':
            features_encoder._out_channels = list(features_encoder._out_channels)
            features_encoder._out_channels[-1] = prototype_kwargs['num_per_task'] + prototype_kwargs['num_shared']
        
        decoder_names = task_kwargs['task_names']+['share']
        decoders = torch.nn.ModuleDict(
            {task: UnetDecoder(
                encoder_channels=features_encoder._out_channels,
                decoder_channels=decoder_channels,
                output_channels=task_kwargs['task_outchannel'],
                proto_channels = proto_channels,
                n_blocks=encoder_depth,
                use_batchnorm=decoder_use_batchnorm,
                center=True if base_architecture.startswith("vgg") else False,
                attention_type=decoder_attention_type,
                skip=skip,
            ) 
            for task in task_kwargs['task_names']}
        )
        warm_decoder = UnetDecoder(
                encoder_channels=features_encoder._out_channels,
                decoder_channels=decoder_channels,
                output_channels=task_kwargs['task_outchannel'],
                proto_channels = proto_channels,
                n_blocks=encoder_depth,
                use_batchnorm=decoder_use_batchnorm,
                center=True if base_architecture.startswith("vgg") else False,
                attention_type=decoder_attention_type,
                skip=skip,
            ) 


        super().__init__(
            task_kwargs=task_kwargs,
            prototype_kwargs=prototype_kwargs,
            features=features_encoder,
            decoder=decoders,
            warm_decoder=warm_decoder,
            img_size=img_size,
            prototype_shape=prototype_shape,
            proto_layer_rf_info=proto_layer_rf_info,
            num_classes=num_classes,
            init_weights=True,
            prototype_activation_function=prototype_activation_function,
            add_on_layers_type=add_on_layers_type,
            proto_combine_mode=proto_combine_mode,
            weighted=weighted,)
