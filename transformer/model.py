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


# 层归一化
class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super().__init__()

        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        # 防止在计算标准差时出现 ÷ 0 的错误
        self.eps = eps

    def forward(self, x):
        # 计算 x 的均值，保持维度不变
        mean = x.mean(-1, keepdim=True)
        # 计算 x 的标准差，保持维度不变
        std = x.std(-1, keepdim=True)
        #
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


class MultiHeadedAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        """
        接收多个头的个数和维度进行初始化
        
        Args
            h 头的个数
            d_model 模型维度
            dropout dropout概率
        """
        super().__init__()

        # 确保 d_model 可以被 h 整除
        assert d_model % h == 0

        self.d_k = d_model
        self.h = h

        # 四个线性层的列表
        self.linears = clones(nn.Linear(d_model, d_model), 4)

        # 存储注意力权重
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        # 将掩码 mask 扩展为一个四维向量，使其形状和注意力权重张量相匹配
        if mask is not None:
            # mask(batch_size, seq_len) ==> (batch_size, 1, seq_len)
            mask = mask.unsqueeze(1)
        # 获取 query 张量中第一个维度的大小
        nbatches = query.size(0)
        # 1. 批量计算线性投影从 d_model => h * d_k
        query, key, value = [
            # 2. lin 是 self.linears 列表中的一个线性层
            # 3. x 是 (q, k, v) 元组中的一个元素
            # 1. lin(x) 对 x 进行线性变换，将变换后的结果重新排列 ==> (nbatches, -1, self.h, self.d_k)
            # self.d_k 头的维度
            # 这里用了三个线性层，分别对应 q, k, v
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2. 计算 attention
        x, self.attn = attention(query,
                                 key,
                                 value,
                                 mask=mask,
                                 dropout=self.dropout)

        # 3. 将多头的结果连接起来
        # 将 x 的第二个和第三个维度交换，转换为连续的内存布局，重新排列为 ==> (nbatches, -1, self.h * self.d_k)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1,
                                                self.h * self.d_k)

        del query
        del key
        del value

        # 最后一个线性层
        return self.linears[-1](x)


# 基于位置的前馈网络 == 全连接神经网络
class PositionwiseFeedForward(nn.Module):
    """
    PositionwiseFeedForward 实现一个FFN模型
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        # 两个线性层
        self.w_1 = nn.Linear(d_model, d_ff)
        # 第二个线性层将输出映射回d_model维度
        self.w_2 = nn.Linear(d_ff, d_model)
        # dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))


class Embeddings(nn.Module):

    def __init__(self, d_model, vocab):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    PositionalEncoding 实现位置编码 PE
    """

    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        # 计算位置编码
        self.dropout = nn.Dropout(p=dropout)

        # 在对数空间中计算位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)


def make_model(src_vocab,
               tgt_vocab,
               N=6,
               d_model=512,
               d_ff=2048,
               h=8,
               dropout=0.1):
    """
     _summary_ 从超参数中构建一个模型
    """
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    # 模型
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


class Batch:
    """
    训练期间用于保存一批带掩码的数据的图像
    """

    def __init__(self, src, tgt=None, pad=2):
        self.src = src
        # unsqueeze 在最后一个维度 增加一个维度
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        # 创建一个掩码来隐藏并填充未来的word
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data)

        return tgt_mask
