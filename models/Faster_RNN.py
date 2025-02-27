import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.ops import RoIPool
from torchvision.ops import nms

def normal_init(layer, mean, std):
    """重みを正規分布で初期化する関数"""
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        layer.weight.data.normal_(mean, std)
        if layer.bias is not None:
            layer.bias.data.zero_()
    return layer


class RegionProposalNetwork(nn.Module):
    def __init__(self, in_channels, mid_channels, anchor_ratios=[.5, 1, 2], anchor_scales=[8, 16, 32], feat_stride=16):
        super(RegionProposalNetwork, self).__init__()
        
        self.anchor_ratios = anchor_ratios
        self.anchor_scales = anchor_scales
        self.feat_stride = feat_stride
        n_anchors = len(anchor_ratios) * len(anchor_scales)
        
        #RPNの畳み込み層
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.score_layer = nn.Conv2d(mid_channels, n_anchors * 2, 1, 1, 0)
        self.bbox_layer = nn.Conv2d(mid_channels, n_anchors * 4, 1, 1, 0)
        
        #重みの初期化
        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score_layer, 0, 0.01)
        normal_init(self.bbox_layer, 0, 0.01)
    
    def forward(self, x):
        n, _, h, w = x.shape
        x = F.relu(self.conv1(x))
        
        rpn_scores = self.score_layer(x)
        rpn_bboxes = self.bbox_layer(x)
        
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous().view(n, -1, 2)
        rpn_bboxes = rpn_bboxes.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
        
        return rpn_scores, rpn_bboxes


class FasterRCNN(nn.Module):
    def __init__(self, num_classes, backbone='resnet50', pretrained=True):
        super(FasterRCNN, self).__init__()
        
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            backbone_out_channels = 2048
            self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-2])
        
        self.rpn = RegionProposalNetwork(backbone_out_channels, 512)
        
        self.roi_pool = RoIPool((7, 7), 1/16)
        
        self.classifier = nn.Sequential(
            nn.Linear(backbone_out_channels * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout()
        )
        
        self.cls_score = nn.Linear(4096, num_classes)
        self.bbox_pred = nn.Linear(4096, num_classes * 4)
        
        # 重みの初期化
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                normal_init(layer, 0, 0.01)
        normal_init(self.cls_score, 0, 0.01)
        normal_init(self.bbox_pred, 0, 0.01)
    
    def forward(self, x, scale=1.):
        features = self.feature_extractor(x)
        rpn_scores, rpn_bboxes = self.rpn(features)
        
        rois = self.generate_rois(rpn_scores, rpn_bboxes, scale)
        roi_features = self.roi_pool(features, rois)
        # 特徴量の平坦化
        roi_features = roi_features.view(roi_features.size(0), -1)
        roi_features = self.classifier(roi_features)
        
        cls_scores = self.cls_score(roi_features)
        bbox_preds = self.bbox_pred(roi_features)
        
        return cls_scores, bbox_preds, rois

    def generate_rois(self, rpn_scores, rpn_bboxes, scale=1.):
        """RPNの出力から実際のRoIを生成する"""
        nms_thresh = 0.7
        n_train_pre_nms = 12000
        n_train_post_nms = 2000
        
        scores = rpn_scores.softmax(dim=2)[:, :, 1]
        
        rois = []
        for i in range(rpn_scores.size(0)):
            score = scores[i]
            bbox = rpn_bboxes[i]
            
            length = min(n_train_pre_nms, score.size(0))
            order = score.argsort(descending=True)[:length]
            bbox = bbox[order]
            score = score[order]
            
            keep = nms(bbox, score, nms_thresh)
            keep = keep[:n_train_post_nms]
            bbox = bbox[keep]
            
            bbox = bbox * scale
            
            batch_index = torch.full((bbox.size(0), 1), i, dtype=torch.float32, device=bbox.device)
            roi = torch.cat([batch_index, bbox], dim=1)
            rois.append(roi)
        
        rois = torch.cat(rois, dim=0)
        return rois
            