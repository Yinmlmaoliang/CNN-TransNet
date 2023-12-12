import torch
import torch.nn as nn
from resnet_rgb import load_resnet_rgb_model
from resnet_depth import load_resnet_depth_model
from mmfp import MMFPv1, MMFPv2
from transformer import Transformer

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.rgb = load_resnet_rgb_model(pretrained=True)
        self.depth = load_resnet_depth_model(pretrained=True)

        self.level1 = MMFPv1(in_channels=64*2, out_channels=256)
        self.level2 = MMFPv2(in_channels=64*2, out_channels=256)
        self.level3 = MMFPv2(in_channels=64*2, out_channels=256)
        self.level4 = MMFPv2(in_channels=128*2, out_channels=256)
        self.level5 = MMFPv2(in_channels=128*2, out_channels=256)
        self.level6 = MMFPv2(in_channels=256*2, out_channels=256)
        self.level7 = MMFPv2(in_channels=256*2, out_channels=256)
        self.level8 = MMFPv2(in_channels=512*2, out_channels=256)
        self.level9 = MMFPv2(in_channels=512*2, out_channels=256)

        self.transformer = Transformer(length=9, num_classes=51, dim=256, depth=20, heads=8, mlp_dim=512,
                                       dropout=0.1, emb_dropout=0.1)

    def forward(self, x1, x2):
        # feature extraction
        conv1_r, res1_r, res2_r, res3_r, res4_r = self.rgb(x1)
        conv1_d, res1_d, res2_d, res3_d, res4_d = self.depth(x2)

        l1 = torch.cat((conv1_r, conv1_d), dim=1)
        l2 = torch.cat((res1_r[0], res1_d[0]), dim=1)
        l3 = torch.cat((res1_r[1], res1_d[1]), dim=1)
        l4 = torch.cat((res2_r[0], res2_d[0]), dim=1)
        l5 = torch.cat((res2_r[1], res2_d[1]), dim=1)
        l6 = torch.cat((res3_r[0], res3_d[0]), dim=1)
        l7 = torch.cat((res3_r[1], res3_d[1]), dim=1)
        l8 = torch.cat((res4_r[0], res4_d[0]), dim=1)
        l9 = torch.cat((res4_r[1], res4_d[1]), dim=1)

        # feature fusion
        l1 = self.level1(l1)
        l2 = self.level2(l2)
        l3 = self.level3(l3)
        l4 = self.level4(l4)
        l5 = self.level5(l5)
        l6 = self.level6(l6)
        l7 = self.level7(l7)
        l8 = self.level8(l8)
        l9 = self.level9(l9)

        # feature enhance
        sequence = torch.cat((l1, l2, l3, l4, l5, l6, l7, l8, l9), dim=1)
        output = self.transformer(sequence)
        return output

