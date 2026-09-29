"""Checkpoint-initialized depth encoder with causal observation-action memory."""
import hashlib
from pathlib import Path
import torch
from torch import nn
from isaacgymenvs.vision.efficientnet import EfficientNet
from isaacgymenvs.open_loop.grasp_image_obs import META_DIM,DEPTH_MEAN,DEPTH_STD


class RecurrentGrasp(nn.Module):
    def __init__(self, checkpoint=None, hidden=256, architecture='recurrent_v1'):
        super().__init__()
        if architecture not in ('recurrent_v1', 'residual_gru_v1', 'memory_residual_gru_v1',
                                 'spatial_memory_gru_v1'):
            raise ValueError('Unknown grasp architecture: '+architecture)
        self.architecture = architecture
        self.encoder=EfficientNet.from_name('efficientnet-b0',in_channels=1,num_classes=1)
        self.initial_checkpoint_sha256=None
        if checkpoint is not None:
            path=Path(checkpoint)
            state=torch.load(path,map_location='cpu')
            self.encoder.load_state_dict(state['model'],strict=True)
            self.initial_checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        self.feature_dim=self.encoder._fc.in_features
        self.masks=nn.Sequential(nn.Conv2d(2,16,5,stride=2,padding=2),nn.ELU(),
            nn.Conv2d(16,32,3,stride=2,padding=1),nn.ELU(),nn.AdaptiveAvgPool2d(1),nn.Flatten())
        self.spatial_feature_dim = 256 if architecture == 'spatial_memory_gru_v1' else 0
        if self.spatial_feature_dim:
            # Preserve pixel correspondence between depth, visible target and
            # robot/unknown masks.  The legacy mask branch globally pooled that
            # geometry before it could interact with depth features.
            self.spatial = nn.Sequential(
                nn.Conv2d(6,32,5,stride=2,padding=2),nn.ELU(),
                nn.Conv2d(32,64,3,stride=2,padding=1),nn.ELU(),
                nn.Conv2d(64,64,3,stride=2,padding=1),nn.ELU(),
                nn.AdaptiveAvgPool2d(2),nn.Flatten(),
                nn.Linear(256,self.spatial_feature_dim),nn.LayerNorm(self.spatial_feature_dim),nn.ELU())
        extra = 2 if architecture != 'recurrent_v1' else 0
        self.fuse=nn.Sequential(nn.Linear(2*self.feature_dim+self.spatial_feature_dim+64+META_DIM+extra,hidden),nn.LayerNorm(hidden),nn.ELU())
        if architecture != 'recurrent_v1':
            self.fuse.add_module('regularization', nn.Dropout(.1))
        self.gru=nn.GRU(hidden,hidden)
        self.classifier=nn.Linear(hidden,1)
        if architecture != 'recurrent_v1':
            # Start with the stock local-crop GN, not a random stopping head.
            nn.init.zeros_(self.classifier.weight)
            nn.init.zeros_(self.classifier.bias)
        self.hidden=hidden
        # BN running statistics and stochastic depth stay fixed during small-
        # batch fine-tuning; gradients can still update selected CNN weights.
        self.encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def image_features(self,images,masks=None):
        # images (B,2,112,112): local target crop and full-workspace overview.
        b=images.shape[0]
        x=(images.reshape(2*b,1,*images.shape[-2:])-DEPTH_MEAN)/DEPTH_STD
        result=self.encoder._avg_pooling(self.encoder.extract_features(x)).flatten(1).reshape(b,-1)
        if self.spatial_feature_dim:
            if masks is None or masks.shape != (b,2,2,*images.shape[-2:]):
                raise ValueError('Spatial grasp architecture requires aligned masks')
            normalized=(images-DEPTH_MEAN)/DEPTH_STD
            spatial=torch.cat([normalized[:,0:1],masks[:,0].float(),
                               normalized[:,1:2],masks[:,1].float()],dim=1)
            result=torch.cat([result,self.spatial(spatial)],dim=-1)
        return result

    def forward_features(self,features,masks,meta,hidden=None):
        t,b=features.shape[:2]
        m=self.masks(masks.reshape(t*b*2,2,*masks.shape[-2:]).float()).reshape(t,b,-1)
        inputs = [features,m,meta]
        base = None
        if self.architecture != 'recurrent_v1':
            # Only observed pixels/features enter this prior. Unknown/robot
            # pixels attenuate confidence; no clear simulator image or true
            # target pose is accessed. Memory learns a signed correction.
            memory = self.architecture in ('memory_residual_gru_v1','spatial_memory_gru_v1')
            # ``image_features`` appends the learned spatial descriptor after
            # the two EfficientNet descriptors.  The stock prior must consume
            # exactly the remembered-image descriptor, not that appended tail.
            local = (features[...,self.feature_dim:2*self.feature_dim] if memory else
                     features[...,:self.feature_dim])
            raw = self.encoder._fc(local).squeeze(-1).clamp(-8.,8.)
            missing = masks[:,:,0,1].float().mean(dim=(-1,-2))
            base = (raw * torch.exp(-.1*120.*meta[:,:,21]) if memory else
                    raw * torch.exp(-8.*missing) * meta[:,:,20])
            inputs.append(torch.stack([raw/8., missing],dim=-1))
        x=self.fuse(torch.cat(inputs,-1))
        y,hidden=self.gru(x,hidden)
        logits = self.classifier(y).squeeze(-1)
        return (logits if base is None else logits+base),hidden

    def forward(self,images,masks,meta,hidden=None):
        t,b=images.shape[:2]
        features=self.image_features(images.reshape(t*b,2,*images.shape[-2:]),
            masks.reshape(t*b,2,2,*masks.shape[-2:])).reshape(t,b,-1)
        return self.forward_features(features,masks,meta,hidden)

    def finetune_last_stage(self):
        for p in self.encoder.parameters():p.requires_grad=False
        for module in [self.encoder._blocks[-1],self.encoder._conv_head,self.encoder._bn1]:
            for p in module.parameters():p.requires_grad=True


def load_recurrent_grasp(path,device='cpu'):
    state=torch.load(path,map_location='cpu')
    if state.get('protocol')!='occluded-image-grasp-v1':raise ValueError('Wrong image classifier protocol')
    net=RecurrentGrasp(hidden=state['hidden'],architecture=state.get('architecture','recurrent_v1'))
    net.load_state_dict(state['model'],strict=True)
    return net.to(device).eval(),state
