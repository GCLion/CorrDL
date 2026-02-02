import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseSegmentor
from backbones import BuildActivation, BuildNormalization, constructnormcfg
from builder import BuildBackbone

class GraphRandomWalkNPR(nn.Module):
    def __init__(self, walk_steps=3, walk_restart_prob=0.7, graph_threshold=0.5):
        super().__init__()
        self.walk_steps = walk_steps
        self.walk_restart_prob = walk_restart_prob
        self.graph_threshold = graph_threshold

    def interpolate(self, img, factor):
        return F.interpolate(
            F.interpolate(img, scale_factor=factor, mode='bilinear', recompute_scale_factor=True, align_corners=True),
            scale_factor=1/factor, mode='bilinear', recompute_scale_factor=True, align_corners=True
        )

    def _compute_npr(self, x):
        blurred = self.interpolate(x, 0.5)
        return x - blurred  

    def _build_graph(self, x, npr):
        B, C, H, W = x.shape
        N = H * W
        
        x_flat = x.view(B, C, -1).permute(0, 2, 1) 
        npr_flat = npr.view(B, C, -1).permute(0, 2, 1) 
        node_features = torch.cat([x_flat, npr_flat], dim=2) 
        
        node_norm = F.normalize(node_features, dim=2)
        adj_matrix = torch.bmm(node_norm, node_norm.transpose(1, 2)) 
        adj_matrix = adj_matrix * (adj_matrix >= self.graph_threshold).float()
        mask = torch.eye(N, device=x.device).unsqueeze(0).repeat(B, 1, 1)
        adj_matrix = adj_matrix * (1 - mask)
        
        row_sums = adj_matrix.sum(dim=2, keepdim=True)
        adj_matrix = adj_matrix / (row_sums + 1e-8)
        return adj_matrix, node_features

    def _random_walk(self, adj_matrix):
        B, N, _ = adj_matrix.shape
        walk_probs = torch.eye(N, device=adj_matrix.device).unsqueeze(0).repeat(B, 1, 1)
        
        for _ in range(self.walk_steps):
            step_probs = self.walk_restart_prob * torch.bmm(walk_probs, adj_matrix)
            step_probs += (1 - self.walk_restart_prob) * torch.eye(N, device=adj_matrix.device).unsqueeze(0).repeat(B, 1, 1)
            walk_probs = step_probs
        return walk_probs

    def _aggregate_features(self, node_features, walk_probs, H, W):
        B, N, C = node_features.shape
        aggregated = torch.bmm(walk_probs, node_features) 
        return aggregated.permute(0, 2, 1).view(B, C, H, W)

    def forward(self, x):
        npr = self._compute_npr(x)
        adj_matrix, node_features = self._build_graph(x, npr)
        walk_probs = self._random_walk(adj_matrix)
        global_feat = self._aggregate_features(node_features, walk_probs, x.shape[2], x.shape[3])
        return npr, global_feat 

class MSCCNet(BaseSegmentor):
    def __init__(self, cfg, mode):
        super(MSCCNet, self).__init__(cfg, mode)
        norm_cfg = {'type': 'batchnorm2d'}
        act_cfg = {'type': 'relu', 'inplace': True}
        head_cfg = {
            'in_channels': 2048,
            'feats_channels': 512,
            'out_channels': 512,
            'dropout': 0.1,
            'spectral_height': 64,
            'spectral_width': 64,
            'spectral_k': 8,
        }

        # build bottleneck
        self.bottleneck1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(constructnormcfg(placeholder=512, norm_cfg=norm_cfg)),
            BuildActivation(act_cfg),
        )
        self.bottleneck2 = nn.Sequential(
            nn.Conv2d(512*2, 512, kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(constructnormcfg(placeholder=512, norm_cfg=norm_cfg)),
            BuildActivation(act_cfg),
        )
        self.bottleneck3 = nn.Sequential(
            nn.Conv2d(512*4, 512, kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(constructnormcfg(placeholder=512, norm_cfg=norm_cfg)),
            BuildActivation(act_cfg),
        )
        # build fusion bottleneck
        self.bottleneck_fusion = nn.Sequential(
            nn.Conv2d(512*3, 512, kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(constructnormcfg(placeholder=512, norm_cfg=norm_cfg)),
            BuildActivation(act_cfg),
        )
        # build decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(head_cfg['out_channels'], head_cfg['out_channels'], kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(constructnormcfg(placeholder=head_cfg['out_channels'], norm_cfg=norm_cfg)),
            BuildActivation(act_cfg),
            nn.Dropout2d(head_cfg['dropout']),
            nn.Conv2d(head_cfg['out_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
        )

        # # build auxiliary decoder
        backbone_cfg = copy.deepcopy(cfg['backbone'])
        if 'norm_cfg' not in backbone_cfg:
            backbone_cfg.update({'norm_cfg': copy.deepcopy(self.norm_cfg)})
        self.backbone_net = BuildBackbone(backbone_cfg)
    
    def interpolate(self, img, factor):
        return F.interpolate(F.interpolate(img, scale_factor=factor, mode='nearest', recompute_scale_factor=True),
                             scale_factor=1 / factor, mode='nearest', recompute_scale_factor=True)
        
    '''forward'''
    def forward(self, x, targets=None, losses_cfg=None): 
        backbone_outputs = self.transforminputs(self.backbone_net(x), selected_indices=self.cfg['backbone'].get('selected_indices'))
 
        x1 = backbone_outputs[1]
        x1 = self.bottleneck1(x1)
        x2 = backbone_outputs[2]
        x2 = self.bottleneck2(x2)
        x3 = backbone_outputs[3]
        x3 = self.bottleneck3(x3)
        x123 = self.bottleneck_fusion(torch.cat((x1, x2, x3), dim=1))
        
        return x123



class DiffAttention2D(nn.Module):
    def __init__(self, channels, heads=8):
        super().__init__()
        assert channels % heads == 0
        self.heads = heads
        self.d = channels // heads
        self.to_q1 = nn.Conv2d(channels, channels, 1, bias=False)
        self.to_k1 = nn.Conv2d(channels, channels, 1, bias=False)
        self.to_q2 = nn.Conv2d(channels, channels, 1, bias=False)
        self.to_k2 = nn.Conv2d(channels, channels, 1, bias=False)
        self.to_v = nn.Conv2d(channels, channels, 1, bias=False)
        self.unify = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        q1 = self.to_q1(x).view(B, self.heads, self.d, -1)
        k1 = self.to_k1(x).view(B, self.heads, self.d, -1)
        q2 = self.to_q2(x).view(B, self.heads, self.d, -1)
        k2 = self.to_k2(x).view(B, self.heads, self.d, -1)
        v = self.to_v(x).view(B, self.heads, self.d, -1)

        attn1 = torch.einsum('bhcn,bhcm->bhnm', q1, k1) / (self.d**0.5)
        attn2 = torch.einsum('bhcn,bhcm->bhnm', q2, k2) / (self.d**0.5)

        a1 = F.softmax(attn1, dim=-1)
        a2 = F.softmax(attn2, dim=-1)

        diff_attn = a1 - a2
        out = torch.einsum('bhnm,bhcm->bhcn', diff_attn, v)
        out = out.reshape(B, C, H, W)
        return self.unify(out) + x  # 

class ConvRefine(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.refine(x) + x  # 

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=512, n_classes=1, bilinear=True, target_size=(256, 256)):
        super(UNet, self).__init__()
        self.n_channels = n_channels  
        self.n_classes = n_classes   
        self.bilinear = bilinear
        self.target_size = target_size  

        factor = 2 if bilinear else 1
        
        self.inc = DoubleConv(n_channels, 64) 
        self.down1 = Down(64, 128)            # → [B,128,H/2,W/2]
        self.down2 = Down(128, 256)           # → [B,256,H/4,W/4]
        self.down3 = Down(256, 512 // factor) # → [B,256,H/8,W/8] 
        
        self.up1 = Up(512, 256 // factor, bilinear)  # 256+256=512 → [B,128,H/4,W/4]
        self.up2 = Up(256, 128 // factor, bilinear)  # 128+128=256 → [B,64,H/2,W/2]
        self.up3 = Up(128, 64, bilinear)             # 64+64=128 → [B,64,H,W]
        self.outc = OutConv(64, n_classes)  # [B,64,H,W] → [B,1,H,W]
        
        self.final_up = nn.Upsample(size=target_size, mode='bilinear', align_corners=True)

    def forward(self, x):
        # 编码器下采样
        x1 = self.inc(x)       # [B,64,H,W]
        x2 = self.down1(x1)    # [B,128,H/2,W/2]
        x3 = self.down2(x2)    # [B,256,H/4,W/4]
        x4 = self.down3(x3)    # [B,256,H/8,W/8]
        
        x = self.up1(x4, x3)   # [B,128,H/4,W/4] 
        x = self.up2(x, x2)    # [B,64,H/2,W/2] 
        x = self.up3(x, x1)    # [B,64,H,W] 
        
        # 输出掩码并上采样至目标尺寸
        logits = self.outc(x)                  # [B,1,H,W]
        logits = self.final_up(logits)         # [B,1,target_h,target_w]
        return logits.squeeze(1)  #  → [B,target_h,target_w]



class FakeLocator(nn.Module):
    def __init__(self, in_channels=512, num_classes=1):
        super().__init__()
        cfg = {
            'type': 'fcn',
            'num_classes': 1,
            'benchmark': True,
            'align_corners': False,
            'backend': 'nccl',
            'norm_cfg': {'type': 'batchnorm2d'},
            'act_cfg': {'type': 'relu', 'inplace': True},
            'backbone': {
                'type': 'resnet101',
                'series': 'resnet',
                'pretrained': True,
                'outstride': 8,
                'use_stem': True,
                'selected_indices': (0, 1, 2, 3),
            },
            'classifier': {
                'last_inchannels': 2048,
            },
            'head': {
                'in_channels': 2048,
                'feats_channels': 512,
                'out_channels': 512,
                'dropout': 0.1,
                'spectral_height': 64,
                'spectral_width': 64,
                'spectral_k': 8,
            },
            'auxiliary': {
                'in_channels': 1024,
                'out_channels': 512,
                'dropout': 0.1,
            }
        }
        
        self.mscc = MSCCNet(cfg, mode='TRAIN')
        self.diff_attn = DiffAttention2D(in_channels, heads=8)
        self.conv_refine = ConvRefine(in_channels)
        self.unet = UNet(n_channels=512, n_classes=1)
        
        self.graph_module = GraphRandomWalkNPR(
            walk_steps=3,           
            walk_restart_prob=0.7, 
            graph_threshold=0.4     
        )
        
        self.fuse_conv = nn.Sequential(#3*
            nn.Conv2d(4*in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        feat16x = self.mscc(x)  # (B, 512, 32, 32)
        npr_feat, global_feat = self.graph_module(feat16x)  # (B,512,32,32) 和 (B,1024,32,32)
        
        fused_feat = torch.cat([feat16x, npr_feat, global_feat], dim=1)  # (B,512+512+1024,32,32)
        fused_feat = self.fuse_conv(fused_feat)   
        x = self.diff_attn(fused_feat)  
        x = self.conv_refine(x)     
        mask_logits = self.unet(x)   
        
        return mask_logits  


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth 
    
    def forward(self, pred, target):
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        return 1 - dice  


class BCEWithDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1e-6):
        super().__init__()
        self.bce_loss = nn.BCELoss()  # 
        self.dice_loss = DiceLoss(smooth=smooth)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
    
    def forward(self, pred, target):
        bce = self.bce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        total_loss = self.bce_weight * bce + self.dice_weight * dice
        return total_loss


class BCEWithLogitsDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.7, dice_weight=0.3, smooth=1e-6):
        super().__init__()
        self.bce_loss = nn.BCELoss()  
        self.dice_loss = DiceLoss(smooth=smooth)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
    
    def forward(self, logits, target):
        bce = self.bce_loss(logits, target)
        pred = torch.sigmoid(logits)
        dice = self.dice_loss(pred, target)
        
        total_loss = self.bce_weight * bce + self.dice_weight * dice
        return total_loss