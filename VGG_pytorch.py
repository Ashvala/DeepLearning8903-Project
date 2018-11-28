import torch
import pdb
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import math



__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]

class PrintLayer(nn.Module):
        def __init__(self, id_str):
            self.id_str = id_str
            super(PrintLayer, self).__init__()

        def forward(self, x):
            # Do your print / debug stuff here
            print(self.id_str, x.size())
            return x



class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=False):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
                nn.Linear(160,1024),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(1024, 2048),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(2048, 1024),
                nn.ReLU(True),
                nn.Dropout(),
        )
        self.pitch_layer = nn.Linear(1024, num_classes)
#        self.vel_layer = nn.Linear(1024,5)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.features(x)
        x = x.view(x.size(0), -1)        
        x = self.classifier(x)
        pitch_out = self.pitch_layer(x)
#        vel_out = F.relu(self.vel_layer(x))
        return pitch_out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 1
    for index, v in enumerate(cfg):
        if v[0] == 'A':
            layers += [nn.AvgPool1d(kernel_size=v[1], stride=v[2])]
        elif v[0] == 'M':            
            layers += [nn.MaxPool1d(kernel_size=v[1], stride=v[2])]
        else:
            # v[0] = in_channels
            # v[1] = kernel_size
            # v[2] = stride
            conv1d = nn.Conv1d(in_channels, v[0], kernel_size=v[1], padding=1, stride=v[2])
            if batch_norm:
                layers += [conv1d, nn.BatchNorm1d(v[0]), nn.ReLU(inplace=True)]
            else:
                layers += [conv1d, nn.ReLU(inplace=True)]
#            layers += [PrintLayer("Conv Layer" + str(index))]
            in_channels = v[0]
    return nn.Sequential(*layers)


cfg = {
    'A': [(16,512,16),
          ('M', 8, 8),
          (16,128,1),
          ('M', 2, 2),
          (32,64,1),
          (32,64,1),
          ('A', 2, 1),
          (64,32,1),
          (64,16,1),
          ('A', 2, 1),
          (128,8,1),
          (128,4,1),
          ('A', 2, 1)
    ],
    'A1': [(8,512,32),
          ('M', 8, 4),
          (16,128,2),
          ('M', 4, 2),
          (32,32,2),
          (64,16,2),
        ],
}


def vgg11(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")
    Args:
pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A1']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11']))
    return model


def vgg11_bn(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']))
    return model

