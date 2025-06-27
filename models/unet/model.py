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

class Unet(nn.Module):

    def __init__(self, task_kwargs, features, decoder, img_size,):

        super(Unet, self).__init__()
        self.tasks = task_kwargs['task_names']
        
        self.img_size = img_size
        self.epsilon = 1e-4
        
        # this has to be named features to allow the precise loading
        self.features = features

        self.decoders = decoder


    def forward(self, x):
        features = self.features(x)

        tasks_outputs = {task: self.decoders[task](features) for task in self.tasks}

        return tasks_outputs


    def __repr__(self):
        # PPNet(self, features, img_size, prototype_shape,
        # proto_layer_rf_info, num_classes, init_weights=True):
        rep = (
            'PPNet(\n'
            '\tfeatures: {},\n'
            '\timg_size: {},\n'
            '\tepsilon: {}\n'
            ')'
        )

        return rep.format(self.features,
                          self.img_size,
                          self.epsilon)


class construct_Unet(Unet):
    def __init__(self,
        base_architecture, 
        task_kwargs,
        pretrained=True, 
        img_size=256,
        encoder_depth: int = 5,
        decoder_use_batchnorm = True,
        decoder_channels = (256, 128, 64, 32, 16),
        decoder_attention_type = None,
        skip=False,
    ):
        features_encoder = base_architecture_to_features[base_architecture](pretrained=pretrained)

        decoders = torch.nn.ModuleDict(
            {task: UnetDecoder(
                encoder_channels=features_encoder._out_channels,
                decoder_channels=decoder_channels,
                output_channels=task_kwargs['task_outchannel'],
                n_blocks=encoder_depth,
                use_batchnorm=decoder_use_batchnorm,
                center=True if base_architecture.startswith("vgg") else False,
                attention_type=decoder_attention_type,
                skip=skip,
            ) 
            for task in task_kwargs['task_names']}
        )

        super().__init__(
            task_kwargs=task_kwargs,
            features=features_encoder,
            decoder=decoders,
            img_size=img_size,)

