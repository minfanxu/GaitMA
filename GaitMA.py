import torch
import torch.nn as nn

from ..base_model import BaseModel
from ..modules import SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs, SeparateBNNecks, Multi_SetBlockWrapper

from einops import rearrange
from einops.layers.torch import Rearrange
import numpy as np
import matplotlib.pyplot as plt
import time

from einops import rearrange
from einops.layers.torch import Rearrange

import torch.nn.functional as F
from torch.nn import init

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)

class MutualCrossAttn(nn.Module):
    def __init__(self, feature_dim=2048, num_head=32):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_head = num_head

        self.layer_norm1 = nn.LayerNorm(self.feature_dim, eps=1e-6).apply(_init_vit_weights)
        self.layer_norm2 = nn.LayerNorm(self.feature_dim, eps=1e-6).apply(_init_vit_weights)

    def forward(self, f_a, f_s):
        B, N, C = f_a.shape

        q1 = f_a
        k1 = v1 = f_s
        # [B, N, C] -> [B, N, n, C//n] -> [B, n, N, C//n]
        q1 = q1.reshape(B, N, self.num_head, C // self.num_head).transpose(1, 2)
        k1 = k1.reshape(B, N, self.num_head, C // self.num_head).transpose(1, 2)
        v1 = v1.reshape(B, N, self.num_head, C // self.num_head).transpose(1, 2)

        attn1 = torch.matmul(q1 / ((C // self.num_head) ** 0.5),
                             k1.transpose(-1, -2))
        attn1 = F.softmax(attn1, dim=-1)
        output1 = torch.matmul(attn1, v1)
        # [B, n, N, C//n] -> [B, N, n, C//n] -> [B, N, C]
        output1 = output1.transpose(1, 2).contiguous().flatten(2)

        q2 = f_s
        k2 = v2 = f_a
        # [B, N, C] -> [B, N, n, C//n] -> [B, n, N, C//n]
        q2 = q2.reshape(B, N, self.num_head, C // self.num_head).transpose(1, 2)
        k2 = k2.reshape(B, N, self.num_head, C // self.num_head).transpose(1, 2)
        v2 = v2.reshape(B, N, self.num_head, C // self.num_head).transpose(1, 2)

        attn2 = torch.matmul(q2 / ((C // self.num_head) ** 0.5),
                             k2.transpose(-1, -2))  # attn2 matrix is equal to the transpose of attn1
        attn2 = F.softmax(attn2, dim=-1)
        output2 = torch.matmul(attn2, v2)
        # [B, n, N, C//n] -> [B, N, n, C//n] -> [B, N, C]
        output2 = output2.transpose(1, 2).contiguous().flatten(2)

        output1 = self.layer_norm1(output1 + f_a)
        output2 = self.layer_norm2(output2 + f_s)
        return output1, output2

class GaitMA(BaseModel):
    def build_network(self, model_cfg):
        self.feature_dim = 256
        num_multi_head = 32
        embed_dim = 128

        self.Backbone = self.get_backbone(model_cfg['backbone_cfg'])
        self.Backbone = Multi_SetBlockWrapper(self.Backbone)
        self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg['bin_num'])

        self.co_attn_align = nn.Sequential(
            nn.Linear(2 * self.feature_dim, self.feature_dim).apply(weights_init_kaiming),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, 2 * self.feature_dim).apply(weights_init_kaiming),
            nn.Sigmoid()
        )
        self.mutual_cross_attn = MutualCrossAttn(feature_dim=self.feature_dim, num_head=num_multi_head)


    def forward(self, inputs):
        ipts_sil, ipts_pose2d_hm, labs, _, _, seqL_sil_batch, seqL_pose2d_hm_batch = inputs

        sils = ipts_sil[0]
        pose2d_hm = ipts_pose2d_hm[0]
        del ipts_sil, ipts_pose2d_hm
         
        if len(sils.size()) == 4:
            sils = sils.unsqueeze(1)
            pose2d_hm = pose2d_hm.unsqueeze(1)
    
        outs_sil ,outs_hm= self.Backbone(sils, pose2d_hm)  # [n, c, s, h, w]

        # Temporal Pooling, TP
        outs_sil = self.TP(outs_sil, seqL_sil_batch, options={"dim": 2})[0]  # [n, c, h, w]
        outs_hm = self.TP(outs_hm, seqL_sil_batch, options={"dim": 2})[0]  # [n, c, h, w]
        # Horizontal Pooling Matching, HPM
        outs_sil = self.HPP(outs_sil)  # [n, c, p]
        outs_hm = self.HPP(outs_hm)

        outs_sil = rearrange(outs_sil, 'n c p -> n p c')
        outs_hm = rearrange(outs_hm, 'n c p -> n p c')
        outs = torch.concat((outs_sil, outs_hm), dim=-1)
        align_score = self.co_attn_align(outs)
        aligned_sil = align_score[:, :, 0: self.feature_dim] * outs_sil + outs_sil
        aligned_hm = align_score[:, :, self.feature_dim:] * outs_hm + outs_hm
        sil_hm, hm_sil = self.mutual_cross_attn(aligned_sil, aligned_hm)
        feat = torch.concat((sil_hm, hm_sil), dim=-1)  # [B, hw, 2d]

        feat = rearrange(feat, 'n p c -> n c p')

        embed_1 = self.FCs(feat)  # [n, c, p]
        embed_2, logits = self.BNNecks(embed_1)  # [n, c, p]
        embed = embed_1


        n, _, s, h, w = sils.size()
        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed_1, 'labels': labs},
                'softmax': {'logits': logits, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': sils.view(n*s, 1, h, w)
            },
            'inference_feat': {
                'embeddings': embed
            }
        }
        
        return retval
