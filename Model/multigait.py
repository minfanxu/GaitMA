import torch

from ..base_model import BaseModel
from ..modules import SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs, SeparateBNNecks, Multi_SetBlockWrapper

from einops import rearrange
from einops.layers.torch import Rearrange
import numpy as np
import matplotlib.pyplot as plt
import time

class Multigait(BaseModel):

    def build_network(self, model_cfg):
        self.Backbone_sil = self.get_backbone(model_cfg['backbone_cfg'])
        self.Backbone_sil = SetBlockWrapper(self.Backbone_sil)
        self.Backbone_hm = self.get_backbone(model_cfg['backbone_cfg'])
        self.Backbone_hm = SetBlockWrapper(self.Backbone_hm)
        self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg['bin_num'])

    def forward(self, inputs):
        ipts_sil, ipts_pose2d_hm, labs, _, _, seqL_sil_batch, seqL_pose2d_hm_batch = inputs

        sils = ipts_sil[0]
        pose2d_hm = ipts_pose2d_hm[0]
        del ipts_sil, ipts_pose2d_hm
         
        if len(sils.size()) == 4:
            sils = sils.unsqueeze(1)
            pose2d_hm = pose2d_hm.unsqueeze(1)
    
        outs_sil = self.Backbone_sil(sils)  # [n, c, s, h, w]
        outs_hm = self.Backbone_hm(pose2d_hm)


        # Temporal Pooling, TP
        outs = self.TP(outs_sil, seqL_sil_batch, options={"dim": 2})[0]  # [n, c, h, w]

        # Horizontal Pooling Matching, HPM
        feat = self.HPP(outs)  # [n, c, p]

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
