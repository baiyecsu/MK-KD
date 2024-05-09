# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
from os.path import join as pjoin

import math
import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage
from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair

# from . import vit_configs as configs

from model.Res_Transformer import vit_config as configs

from model.Res_Transformer.vit_resnet_skip import ResNetV2

logger = logging.getLogger(__name__)

# 这些常量定义了不同的注意力组件的名字，用于加载预训练的权重或者进行特定的操作。
ATTENTION_Q = "MultiHeadDotProductAttention_1/query/"
ATTENTION_K = "MultiHeadDotProductAttention_1/key/"
ATTENTION_V = "MultiHeadDotProductAttention_1/value/"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out/"
FC_0 = "MlpBlock_3/Dense_0/"
FC_1 = "MlpBlock_3/Dense_1/"
ATTENTION_NORM = "LayerNorm_0/"
MLP_NORM = "LayerNorm_2/"

# 一个辅助函数，用于转换权重格式。当使用卷积操作时，权重的维度需要从 HWIO 转换为 OIHW。
def np2th(weights, conv=False):
    """Converts the weights from numpy format to torch format."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

# Swish激活函数，它是由Google提出的一种自定义激活函数。
def swish(x):
    return x * torch.sigmoid(x)

# 定义了一些激活函数，方便后续在模型中引用。
ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


# 这是多头注意力模块的实现。
class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        # 是否可视化注意力权重
        self.vis = vis  
        # 注意力头的数量
        self.num_attention_heads = config.transformer["num_heads"]  
        # 每个注意力头的大小
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)  
        # 总的维度
        self.all_head_size = self.num_attention_heads * self.attention_head_size  

        # 定义查询、键、值的线性变换层
        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        # 定义输出的线性变换层
        self.out = Linear(config.hidden_size, config.hidden_size)
        # 定义dropout层，用于训练时防止过拟合
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])
        # 定义softmax激活函数
        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        # 重新排列张量的维度，为多头注意力计算做准备
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)  
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        # 对输入的隐藏状态进行线性变换，得到查询、键、值
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        # 对查询、键、值进行维度变换
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # 计算注意力得分
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # 对得分进行缩放
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # 使用softmax得到注意力权重
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        # 使用dropout
        attention_probs = self.attn_dropout(attention_probs)

        # 根据权重和值计算上下文
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        # 输出结果
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights
    

    # MLP模块，通常用于Transformer的Feed Forward层
class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        # 定义两个线性层
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        # 使用GELU作为激活函数
        self.act_fn = ACT2FN["gelu"]
        # 定义dropout层，用于减少过拟合
        self.dropout = Dropout(config.transformer["dropout_rate"])
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        # 使用xavier_uniform初始化权重，有助于模型的收敛
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        # 初始化偏置为小的正态分布
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


# 嵌入层，用于生成位置和patch的嵌入
class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings."""

    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        # 将img_size转换为元组形式
        img_size = _pair(img_size)

        # 根据配置文件，判断是否使用grid-based的patch策略（可能用于ResNet结构）
        if config.patches.get("grid") is not None:
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        # 如果是hybrid模式，使用ResNet来提取特征
        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        # 定义patch的嵌入层，使用卷积层来实现
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        # 定义位置嵌入，这里是可学习的参数
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))
        # 定义dropout层
        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        # 如果是hybrid模式，先通过ResNet提取特征
        if self.hybrid:
            x, features = self.hybrid_model(x)
        else:
            features = None
        # 使用卷积层得到patch的嵌入
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        # 加上位置嵌入
        
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features


# Block类代表了Transformer中的一个编码块
class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        # Transformer块的hidden_size
        self.hidden_size = config.hidden_size 
        # 定义Transformer块中的两个LayerNorm层
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        # 定义Feed Forward层
        self.ffn = Mlp(config)
        # 定义多头注意力层
        self.attn = Attention(config, vis)

    def forward(self, x):
        # 对输入进行Layer Normalization
        h = x
        x = self.attention_norm(x)
        # 通过多头注意力层
        x, weights = self.attn(x)
        # Residual连接
        x = x + h
        # 再次对输入进行Layer Normalization
        h = x
        x = self.ffn_norm(x)
        # 通过Feed Forward层
        x = self.ffn(x)
        # Residual连接
        x = x + h
        return x, weights

    # 从预训练模型中加载权重
    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}/"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size,
                                                                                   self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size,
                                                                                   self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size,
                                                                                   self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))
            
    
# Encoder类代表了Transformer的编码器部分，它包含多个Block类的实例
class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        # 定义Transformer块的列表
        self.layer = nn.ModuleList()
        # 定义Transformer编码器的LayerNorm层
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        # 创建多个Transformer块
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        # 逐个传递每个Transformer块
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            # 如果进行可视化，则收集注意力权重
            if self.vis:
                attn_weights.append(weights)
        # 对编码器的输出进行Layer Normalization
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights

    
# Transformer类代表了完整的Transformer结构
class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        # 定义嵌入层
        self.embeddings = Embeddings(config, img_size=img_size)
        # 定义编码器
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        # 获取嵌入输出
        embedding_output, features = self.embeddings(input_ids)
        # 通过编码器
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights, features
    
    
class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        # nn.UpsamplingBilinear2d接受一个张量作为输入，将该张量的空间维度（宽度和高度）沿着两个方向按照指定的比例进行上采样。
        # 在U-Net模型中，该模块的scale_factor参数设置为2，表示将特征图的宽度和高度分别扩大2倍
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels  # (256, 128, 64, 16)
        # print('decoder_channels',decoder_channels)
        in_channels = [head_channels] + list(decoder_channels[:-1])  # 512 256 128 64
        # print('in_channels',in_channels)
        out_channels = decoder_channels

        if self.config.n_skip != 0:  # 3
            skip_channels = self.config.skip_channels  # [512, 256, 64, 16]
            # print('skip_channels',skip_channels)
            for i in range(4 - self.config.n_skip):  # re-select the skip channels according to n_skip
                # for循环遍历了四个分辨率，从高到低依次对应skip_channels中的元素。对于每个元素，如果它对应的分辨率不需要保留，就将该元素的值设置为0，表示不保留任何通道。
                # 具体来说，如果n_skip=0，则不保留任何通道；如果n_skip=1，则只保留最高分辨率的通道；如果n_skip=2，则保留最高分辨率和次高分辨率的通道；
                # 如果n_skip=3，则保留所有通道，即不进行任何修改。
                skip_channels[3 - i] = 0
                # skip_channels [512, 256, 64, 0]

        else:
            skip_channels = [0, 0, 0, 0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes  # 5
        self.zero_head = zero_head  # False
        self.classifier = config.classifier  # seg
        self.transformer = Transformer(config, img_size, vis)
        self.decoder = DecoderCup(config)

        # self.classification_head = nn.Linear(config.hidden_size, num_classes)
        self.classification_head = nn.Linear(config.decoder_channels[-1], num_classes)

        # 原本他是一个分割的头，我换成了分类的头
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.config = config

        
    def forward(self, x):
        # 如果输入是单通道，复制成三通道
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # 通过 Transformer
        x, _, features = self.transformer(x)

        # 通过解码器
        x = self.decoder(x, features)

        # 使用全局平均池化将特征转化为向量
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)

        # 使用分类头
        logits = self.classification_head(x)

        return logits


    def load_from(self, weights):
        with torch.no_grad():

            res_weight = weights
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1] - 1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            # Encoder whole
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(
                    np2th(res_weight["conv_root/kernel"], conv=True))
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)
                        
                        

CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing': configs.get_testing(),
}


def loss_fn(outputs, labels):
    """
    Compute the cross entropy loss given outputs and labels.

    Returns:
        loss (Variable): cross entropy loss for all images in the batch

    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """
    return nn.CrossEntropyLoss()(outputs, labels)


def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.

    Returns: (float) accuracy in [0,1]
    """
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs == labels) / float(labels.size)


# maintain all metrics required in this dictionary-these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    # could add more metrics such as accuracy for each token type
}



if __name__ == '__main__':
    params = {
        "model_version": "Res-Transformer",
        "subset_percent": 1.0,
        "augmentation": "yes",
        "teacher": "none",
        "alpha": 0,
        "learning_rate": 1e-3,
        "temperature": 1,
        "img_size": 256,
        "num_classes": 4,
        "batch_size": 16,
        "num_epochs": 150,
        "dropout_rate": 0.5,
        "vit_name": "R50-ViT-B_16",
        "n_skip": 3,
        "vit_patches_size": 16,
        "num_workers": 4,
        "save_summary_steps": 100
    }
    config = CONFIGS[params.get("vit_name")]
    config.n_classes = params.get("num_classes")
    config.n_skip = params.get("n_skip")

    model = VisionTransformer(config, img_size=params.get("img_size"), num_classes=params.get("num_classes"))
    print(model)

    input = torch.randn(1, 3, 256, 256)
    out = model(input)
    print(out.shape)
