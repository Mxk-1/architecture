import os
from os.path import exists
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
import math
import copy
import time
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
import altair as alt
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
import torchtext.datasets as datasets
import spacy
import GPUtil
import warnings
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import matplotlib.pyplot as plt

# 设置为False跳过Note执行(例如调试)
warnings.filterwarnings("ignore")
RUN_EXAMPLES = True


# 是否在交互式的笔记本环境中运行
def is_interactive_notebook():
    return __name__ == "__main__"


def show_example(fn, args=[]):
    if __name__ == "__main__" and RUN_EXAMPLES:
        return fn(*args)


class DummyOptimizer(torch.optim.Optimizer):

    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None


class DummyScheduler:

    def step(self):
        None


class EncoderDecoder(nn.Module):
    """
    EncoderDecoder 一个标准的解码器编码器模型
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        # 接收处理屏蔽的src和目标序列
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    """
    Generator 定义一个标准的线性层+softmax步骤
    接收最后的decode结果，并返回词典中每个词的概率
    """

    def __init__(self, d_model, vocab):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)


def clones(Module, N):
    """
    clones 产生N个相同的层
    """
    return nn.ModuleList([copy.deepcopy(Module) for _ in range(N)])


# 编码器
class Encoder(nn.Module):
    """
    Encoder 小编码器的核心构成，传入层 + 层归一化
    """

    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, mask):
        """
        forward 做前向传播需要依次传入每一个层，并且带上掩码
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# 残差连接
class SublayerConnection(nn.Module):
    """
    SublayerConnection 紧跟在层归一化后的残差连接
    """

    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        将残差层应用在所有大小相同的层
        """
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """
    EncoderLayer Encoder的一层，包含自注意力和前馈网络
    """

    def __init__(self, size, self_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    """
    Decoder 带掩码的通用解码器
    """

    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """
    DecoderLayer 解码器的一层，包含自注意力，源注意力和前馈网络
    """

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        forward 依次传入每一层
        """
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # 解码器的第二个attn的k,v是编码器提供的输出，用编码器的x去查解码器的attn输出
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    """
    subsequent_mask 屏蔽后面的位置
    """

    attn_shape = (1, size, size)
    # torch.triu() 接收一个上二维张量作为输入，返回一个上三角矩阵，
    # diagonal = 1 表示从主对角线以上的元素开始填充，0 表示其他元素，返回一个 0/1 的矩阵
    subsequent_mask = torch.triu(torch.ones(attn_shape),
                                 diagonal=1).type(torch.uint8)
    # 下三角矩阵，对角线以下的元素为1（True），其他元素为0
    # 可以屏蔽后面的位置
    return subsequent_mask == 0


# 遮罩示意
def example_mask():
    LS_data = pd.concat([
        pd.DataFrame({
            "Subsequent Mask": subsequent_mask(20)[0][x, y].flatten(),
            "Window": y,
            "Masking": x,
        }) for y in range(20) for x in range(20)
    ])
    return (alt.Chart(LS_data).mark_rect().properties(
        height=250, width=250).encode(
            alt.X("Window:O"),
            alt.Y("Masking:O"),
            alt.Color("Subsequent Mask:Q", scale=alt.Scale(scheme="viridis")),
        ).interactive())


def attention(query, key, value, mask=None, dropout=None):
    """
    attention 计算缩放点积注意力机制
    """
    # 返回query最后一个轴的长度
    d_k = query.size(-1)
    # key.transpose(-2, -1) 将key的倒数第二个轴和倒数第一个轴交换 转置操作
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # 如果存在mask，将mask的值为0的位置替换为-1e9
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    # 按照最后一个轴计算softmax，即对每个query计算softmax
    p_attn = scores.softmax(dim=-1)
    # dropout存在，则进行dropout
    if dropout is not None:
        p_attn = dropout(p_attn)
    # 返回注意力权重和value的乘积 和 注意力权重
    return torch.matmul(p_attn, value), p_attn


if __name__ == "__main__":
    show_example(example_mask)
