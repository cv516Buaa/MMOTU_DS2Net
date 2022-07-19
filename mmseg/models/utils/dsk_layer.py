# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch.nn as nn
from mmcv.cnn import ConvModule

class DSKLayer(nn.Module):
    """Domain Selective Kernel Module."""

    def __init__(self, features, r, stride=1 ,L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(DSKLayer, self).__init__()
        d = max(int(features/r), L)
        self.features = features
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(2):
            self.fcs.append(
                nn.Linear(d, features)
            )
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v