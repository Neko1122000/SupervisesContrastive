"""ResNet in PyTorch.
ImageNet-Style ResNet
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
Adapted from: https://github.com/bearpaw/pytorch-classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# from transformers import BertTokenizer
import sys
import math
sys.path.insert(0, '/home/trongld/SupervisesContrastive/pre_train/sentence-transformers')

from sentence_transformers.SentenceTransformer import SentenceTransformer


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channel=3, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, layer=100):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


model_dict = {
    'resnet18': [resnet18, 512],
    'resnet34': [resnet34, 512],
    'resnet50': [resnet50, 2048],
    'resnet101': [resnet101, 2048],
}


class SupCEResNet(nn.Module):
    """encoder + classifier"""
    def __init__(self, name='resnet50', num_classes=17):
        super(SupCEResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        model_path = "/home/trongld/SupervisesContrastive/pre_train/vn_sbert_deploy/phobert_base_mean_tokens_NLI_STS"
        self.description_tokenizer = SentenceTransformer(model_path, device="cuda")
        text_dim = self.description_tokenizer.get_sentence_embedding_dimension()

        # dim = 2 * text_dim
        dim = dim_in + text_dim
        # dim = dim_in
        self.fc = nn.Linear(dim, num_classes)

    @staticmethod
    def scale_dot_product_attention(image_feature, text_feature):
        batch_size = image_feature.shape[0]
        result = torch.empty(text_feature.shape)
        if torch.cuda.is_available():
            result = result.cuda(non_blocking=True)

        for i in range(batch_size):
            outputs = torch.matmul(torch.unsqueeze(image_feature[i], dim=1), 
                                    torch.unsqueeze(text_feature[i], dim=0))
            outputs = F.softmax(outputs, dim=1)

            value = torch.matmul(image_feature[i], outputs)
            result[i] = value
        
        return result

    def forward(self, x, text):
        image_feat = self.encoder(x)
        text_feat = self.description_tokenizer.encode(text)
        if torch.cuda.is_available():
            text_feat = torch.tensor(text_feat, dtype=torch.float, device="cuda")

        # Multi modal
        # context_images_vector = SupDualConResNet.scale_dot_product_attention(image_feat, text_feat)

        # raw_output = image_feat
        raw_output = torch.cat([image_feat, text_feat], dim=1)
        # raw_output = torch.cat([context_images_vector, text_feat], dim=1)

        return self.fc(raw_output)


class SupDualConResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='resnet50', n_classes=17):
        super(SupDualConResNet, self).__init__()
        # Image transform
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()

        # Text transform
        model_path = "/home/trongld/SupervisesContrastive/pre_train/vn_sbert_deploy/phobert_base_mean_tokens_NLI_STS"
        self.description_tokenizer = SentenceTransformer(model_path, device="cuda")
        self.text_dim = self.description_tokenizer.get_sentence_embedding_dimension()

        # self.text_layer = nn.Sequential(
        #     nn.Linear(self.text_dim, 128),
        #     nn.ReLU(0.1)
        # )

        # self.image_layer = nn.Sequential(
        #     nn.Linear(dim_in, 128),
        #     nn.ReLU(0.1),
        # )

        # MLP classifier
        # self.mlp = nn.Sequential(
        #     nn.Linear(self.text_dim*2, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 128),
        #     nn.ReLU()
        # )

        # Classifier
        dim = dim_in + self.text_dim
        # dim = 2 * self.text_dim
        # dim = dim_in
        data_2 = torch.ones([dim, n_classes], dtype=torch.float64, device="cuda")
        torch.nn.init.uniform_(data_2, a=-1./math.sqrt(dim_in), b=1./math.sqrt(dim_in))
        self.classifier = nn.parameter.Parameter(data_2, requires_grad=True)


    @staticmethod
    def scale_dot_product_attention(image_feature, text_feature):
        batch_size = image_feature.shape[0]
        result = torch.empty(text_feature.shape)
        if torch.cuda.is_available():
            result = result.cuda(non_blocking=True)

        for i in range(batch_size):
            outputs = torch.matmul(torch.unsqueeze(image_feature[i], dim=1), 
                                    torch.unsqueeze(text_feature[i], dim=0))
            outputs = F.softmax(outputs, dim=1)

            value = torch.matmul(image_feature[i], outputs)
            result[i] = value
        
        return result
        

    def forward(self, x, description):
        # ENCODER
        # Encoder image
        feat = self.encoder(x)
        # feat = self.image_layer(feat)

        des_feat = self.description_tokenizer.encode(description)
        if torch.cuda.is_available():
            des_feat = torch.tensor(des_feat, dtype=torch.float, device="cuda")
        # des_feat = self.text_layer(des_feat)

        # Multi modal
        # context_images_vector = SupDualConResNet.scale_dot_product_attention(feat, des_feat)

        # raw_output = feat
        # raw_output = torch.cat([context_images_vector, des_feat], dim=1)
        raw_output = torch.cat([feat, des_feat], dim=1)
        # raw_output = self.mlp(raw_output)

        result = {
            "cls_feats": raw_output,
            "label_feats": self.classifier,
            "predicts": torch.matmul(raw_output.double(), self.classifier)
        }

        return result