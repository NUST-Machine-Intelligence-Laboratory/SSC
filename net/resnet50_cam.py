import torch.nn as nn
import torch.nn.functional as F
from misc import torchutils
from net import resnet50_drs
import torch




class DRS(nn.Module):

    def __init__(self, delta):
        super(DRS, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.delta = delta

        self.global_max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        b, c, _, _ = x.size()

        #x = self.relu(x)

        """ 1: max extractor """
        x_max = self.global_max_pool(x).view(b, c, 1, 1)
        x_max = x_max.expand_as(x)

        """ 2: suppression controller"""
        control = self.delta

        """ 3: suppressor"""
        x = torch.min(x, x_max * control)

        return x


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.delta = 0.55
        self.relu = DRS(self.delta)

        self.resnet50 = resnet50_drs.resnet50(pretrained=True, strides=(2, 2, 2, 1))

        self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool,
                                    self.resnet50.layer1)
        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)
        self.stage4 = nn.Sequential(self.resnet50.layer4)

        self.classifier = nn.Conv2d(2048, 20, 1, bias=False)

        self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.classifier])

    def forward(self, x):

        x = self.stage1(x)
        x = self.stage2(x).detach()
        x = self.relu(x)

        x = self.stage3(x)
        x = self.stage4(x)
        x = self.classifier(x)
        x_int = x

        x = torchutils.gap2d(x)

        x = x.view(-1, 20)

        return x, x_int

    def train(self, mode=True):
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False

    def trainable_parameters(self):

        return (list(self.backbone.parameters()), list(self.newly_added.parameters()))


class CAM(Net):

    def __init__(self):
        super(CAM, self).__init__()

    def forward(self, x):

        x = self.stage1(x)

        x = self.stage2(x)

        x = self.stage3(x)

        x = self.stage4(x)

        x = self.classifier(x)

        x = F.relu(x)

        x = x[0] + x[1].flip(-1)

        return x
