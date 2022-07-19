import torch 
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule
from mmcv.cnn import ConvModule

from ..builder import NECKS

@NECKS.register_module()
class DSKNeck(BaseModule):
    """
    Args:
        in_channels (int): The number of input image channels. Default: 3.
        r: the radio for compute d, the length of z.
        L: the minimum dim of the vector z in paper, default 32.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels=2048,
                 r=8,
                 L=32,
                 init_cfg=None):
        super(DSKNeck, self).__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        d = max(int(in_channels/r), L)
        self.fc = nn.Linear(self.in_channels, d)
        self.fcs = nn.ModuleList([])
        for i in range(2):
            self.fcs.append(
                nn.Linear(d, self.in_channels)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_s, x_t):
        
        x_s_expand = x_s[:, None, :, :, :]
        x_t_expand = x_t[:, None, :, :, :]
        feas = torch.cat([x_s_expand, x_t_expand], dim=1)
        fea_U = torch.sum(feas, dim=1)
        # fea_s = self.gap(fea_U).squeeze_()
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
        fea_v = feas * attention_vectors
        fea_v_s = fea_v[:, 0, :, :, :]
        fea_v_t = fea_v[:, 1, :, :, :]
        out_fea_s = []
        out_fea_s.append(fea_v_s)
        out_fea_t = []
        out_fea_t.append(fea_v_t)
        return tuple(out_fea_s), tuple(out_fea_t)

@NECKS.register_module()
class DS2Neck(BaseModule):
    """
    Args:
        in_channels (int): The number of input image channels. Default: 3.
        r: the radio for compute d, the length of z.
        L: the minimum dim of the vector z in paper, default 32.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels=2048,
                 r=8,
                 L=32,
                 init_cfg=None):
        super(DS2Neck, self).__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        d = max(int(in_channels/r), L)
        self.fc = nn.Linear(self.in_channels, d)
        self.fcs = nn.ModuleList([])
        for i in range(2):
            self.fcs.append(
                nn.Linear(d, self.in_channels)
            )
        self.softmax = nn.Softmax(dim=1)
        ## added by LYU: 2022/05/13
        self.conv = ConvModule(
            self.in_channels,
            self.in_channels,
            3,
            padding=1)
        self.conv_s = ConvModule(
            self.in_channels,
            d,
            3,
            padding=1)
        self.conv_t = ConvModule(
            self.in_channels,
            d,
            3,
            padding=1)

    def forward(self, x_s, x_t):
        
        ## 1. DUM
        x_st = (x_s + x_t) / 2.0
        N_s, C_s, H_s, W_s = x_s.size()
        N_t, C_t, H_t, W_t = x_t.size()
        N_st, C_st, H_st, W_st = x_st.size()
        x_st_z = self.conv(x_st)
        proj_query_s = x_s.view(N_s, C_s, -1)
        proj_query_t = x_t.view(N_t, C_t, -1)
        proj_key = x_st_z.view(N_st, C_st, -1).permute(0, 2, 1)
        energy_s = torch.bmm(proj_query_s, proj_key)
        energy_t = torch.bmm(proj_query_t, proj_key)
        energy_new_s = torch.max(
            energy_s, -1, keepdim=True)[0].expand_as(energy_s) - energy_s
        energy_new_t = torch.max(
            energy_t, -1, keepdim=True)[0].expand_as(energy_t) - energy_t
        attention_s = F.softmax(energy_new_s, dim=-1)
        attention_t = F.softmax(energy_new_t, dim=-1)
        attention = (attention_s + attention_t) / 2.0
        proj_value_s = x_s.view(N_s, C_s, -1)
        proj_value_t = x_t.view(N_t, C_t, -1)
        out_s = torch.bmm(attention, proj_value_s)
        out_t = torch.bmm(attention, proj_value_t)
        out_s = out_s.view(N_s, C_s, H_s, W_s)
        out_t = out_t.view(N_t, C_t, H_t, W_t)
        out_s = self.conv_s(out_s)
        out_t = self.conv_t(out_t)
        
        ## 2. DSM
        x_s_expand = x_s[:, None, :, :, :]
        x_t_expand = x_t[:, None, :, :, :]
        feas = torch.cat([x_s_expand, x_t_expand], dim=1)
        fea_U = torch.sum(feas, dim=1)
        # fea_s = self.gap(fea_U).squeeze_()
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
        fea_v = feas * attention_vectors
        fea_v_s = fea_v[:, 0, :, :, :]
        fea_v_t = fea_v[:, 1, :, :, :]

        DUDS_fea_s = torch.cat([fea_v_s, out_s], dim=1)
        DUDS_fea_t = torch.cat([fea_v_t, out_t], dim=1)
        out_fea_s = []
        out_fea_s.append(DUDS_fea_s)
        out_fea_t = []
        out_fea_t.append(DUDS_fea_t)
        return tuple(out_fea_s), tuple(out_fea_t)

@NECKS.register_module()
class ML_DSKNeck(BaseModule):
    """
    Args:
        in_channels (int): The number of input image channels. Default: 3.
        r: the radio for compute d, the length of z.
        L: the minimum dim of the vector z in paper, default 32.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels=[64, 128, 320, 512],
                 in_index=[0, 1, 2, 3],
                 r=8,
                 L=32,
                 dsk_type='ds2',
                 init_cfg=None):
        super(ML_DSKNeck, self).__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.in_index = in_index
        assert len(self.in_channels) == len(self.in_index)
        self.dsk_necks = []
        for i in range(len(self.in_channels)):
            if dsk_type == 'dsk':
                dsk_neck = DSKNeck(in_channels=self.in_channels[i], L=L)
            elif dsk_type == 'ds2':
                if i == (len(self.in_channels)-1):
                    dsk_neck = DS2Neck(in_channels=self.in_channels[i], L=L)
                else:
                    dsk_neck = DSKNeck(in_channels=self.in_channels[i], L=L)
            dsk_neck_name = f'dsk_neck{i+1}'
            self.add_module(dsk_neck_name, dsk_neck)
            self.dsk_necks.append(dsk_neck_name)

    def forward(self, x_s, x_t):
        out_fea_s = []
        out_fea_t = []
        i = 0
        for module in self.children():
            fea_s, fea_t = module(x_s[i], x_t[i])
            out_fea_s.append(fea_s[0])
            out_fea_t.append(fea_t[0])
            i = i + 1
        return tuple(out_fea_s), tuple(out_fea_t)