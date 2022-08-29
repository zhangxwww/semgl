import torch
import torch.nn as nn


class RelationScale(nn.Module):

    def __init__(self, in_features):
        super(RelationScale, self).__init__()
        self.in_features = in_features
        self.out_features = 1

        self.l1 = nn.Sequential(
            nn.Linear(self.in_features, self.in_features * 2),
            nn.BatchNorm1d(self.in_features * 2, momentum=1, affine=True),
            nn.ReLU()
        )
        self.l2 = nn.Sequential(
            nn.Linear(self.in_features * 2, self.in_features * 2),
            nn.BatchNorm1d(self.in_features * 2, momentum=1, affine=True),
            nn.ReLU()
        )
        self.l3 = nn.Linear(self.in_features * 2, self.out_features)

        self.alpha = torch.rand(1, requires_grad=True).cuda()
        nn.init.constant_(self.alpha, 0)
        self.beta = torch.rand(1, requires_grad=True).cuda()
        nn.init.constant_(self.beta, 0)

    def forward(self, x):
        scale = self.l1(x)
        scale = self.l2(scale)
        scale = self.l3(scale)
        scale = torch.sigmoid(scale)
        scale = torch.exp(self.alpha) * scale + torch.exp(self.beta)

        return scale
