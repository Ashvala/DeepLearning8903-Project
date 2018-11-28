import torch
import pdb
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import math



__all__ = [
    'Alvin', 'alvin_big', 'alvin_sn'
]

class PrintLayer(nn.Module):
        def __init__(self, id_str):
            self.id_str = id_str
            super(PrintLayer, self).__init__()

        def forward(self, x):
            # Do your print / debug stuff here
            print(self.id_str, x.size())
            return x



class Alvin(nn.Module):

    def __init__(self, features, num_classes=128, linear_size=3072):
        super(Alvin, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
                nn.Linear(linear_size,2048),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(2048, 1024),
                nn.ReLU(True),
                nn.Dropout(),
        )
        self.pitch_layer = nn.Linear(1024, num_classes)


    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        pitch_out = self.pitch_layer(x)
        return pitch_out


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
            # v[3] = padding
            conv1d = nn.Conv1d(in_channels, v[0], kernel_size=v[1], stride=v[2],padding=v[3])
            if batch_norm:
                layers += [conv1d, nn.BatchNorm1d(v[0]), nn.ReLU(inplace=True)]
            else:
                layers += [conv1d, nn.ReLU(inplace=True)]
            in_channels = v[0]
    return nn.Sequential(*layers)


cfg = {
    'A': [(16,512,16,1),
          ('M', 8, 8),
          (16,128,1,1),
          ('M', 2, 2),
          (32,64,1,1),
          (32,64,1,1),
          ('A', 2, 1),
          (64,32,1,1),
          (64,16,1,1),
          ('A', 2, 1),
          (128,8,1,1),
          (128,4,1,1),
          ('A', 2, 1)
    ],
    'A1': [(16,64,2,32),
           ('M', 8, 8),
           (32,128,2,16),
           ('M', 8, 8),
           (64,16,2,8),
           (128,8,2,4),
           (256,4,2,2),
           ('M', 4,4),
           (512,4,2,2),
           (1024,4,2,2)

    ],
}


def alvin_sn(**kwargs):
    model = Alvin(make_layers(cfg['A1'], batch_norm=True), **kwargs)
    return model


def alvin_big(**kwargs):
    model = Alvin(make_layers(cfg['A'], batch_norm=True), linear_size=1536, **kwargs)
    return model
