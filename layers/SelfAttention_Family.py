import torch
import torch.nn as nn
import numpy as np
import math
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
from reformer_pytorch import LSHSelfAttention
from einops import rearrange, repeat
from layers.utils import get_filter
from typing import List
import torch.nn.functional as F


class DSAttention(nn.Module):
    '''De-stationary Attention'''

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(DSAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        tau = 1.0 if tau is None else tau.unsqueeze(
            1).unsqueeze(1)  # B x 1 x 1 x 1
        delta = 0.0 if delta is None else delta.unsqueeze(
            1).unsqueeze(1)  # B x 1 x 1 x S

        # De-stationary Attention, rescaling pre-softmax score with learned de-stationary factors
        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * tau + delta

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None
        
class WeightedAverageAttention(nn.Module):
    def __init__(self, d_model, n_heads=1, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(WeightedAverageAttention, self).__init__()
        
        # Four kinds of Attention
        self.dot_product_attention = DotProductAttention(mask_flag=True, attention_dropout=attention_dropout, output_attention=False)
        self.concat_attention = ConcatAttention(d_model, n_heads, mask_flag=True, attention_dropout=attention_dropout, output_attention=False)
        self.bilinear_attention = BilinearAttention(d_model, n_heads, mask_flag=True, attention_dropout=attention_dropout, output_attention=False)
        self.minus_attention = MinusAttention(d_model, n_heads, mask_flag=True, attention_dropout=attention_dropout, output_attention=False)

        self.weights = nn.Parameter(torch.ones(4) / 4)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        # Ouput of each Attention
        dot_product_out, _ = self.dot_product_attention(queries, keys, values, attn_mask)
        concat_out, _ = self.concat_attention(queries, keys, values, attn_mask)
        bilinear_out, _ = self.bilinear_attention(queries, keys, values, attn_mask)
        minus_out, _ = self.minus_attention(queries, keys, values, attn_mask)

        # Weighted Average Attenntion
        attention_outputs = torch.stack([dot_product_out, concat_out, bilinear_out, minus_out])  # [4, B, L, H, d_v]
        weights = F.softmax(self.weights, dim=0)
        weighted_output = torch.tensordot(weights, attention_outputs, dims=0).squeeze(0)  # [B, L, H, d_v]

        return weighted_output, []
        
class DotProductAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(DotProductAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        '''
        Input shape:
            q               : [B, L, H, d_k]
            k               : [B, S, H, d_k]
            v               : [B, S, H, d_k]
        Output shape:
            output  :   [B, L, H, d_v]
            weights :   [B, H, L, S]
            scores  :   [B, H, L, S]
        '''
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        q = queries.transpose(1,2)                      # [B, H, L, d_k]
        k = keys.permute(0,2,3,1)                       # [B, H, d_k, S]
        v = values.transpose(1,2)                       # [B, H, S, d_k]
        scale = self.scale or 1. / sqrt(E)

        scores = torch.matmul(q, k) * scale

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        weights = self.dropout(torch.softmax(scores, dim=-1))
        output = torch.matmul(weights, v).permute(0, 2, 1, 3)

        if self.output_attention:
            return output.contiguous(), weights
        else:
            return output.contiguous(), None
        
class ConcatAttention(nn.Module):
    def __init__(self, d_model, n_heads=1, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ConcatAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.score_linear = nn.Linear(2 * (d_model // n_heads), 1)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        '''
        Input shape:
            q               : [B, L, H, d_k]
            k               : [B, S, H, d_k]
            v               : [B, S, H, d_k]
        Output shape:
            output  :   [B, L, H, d_v]
            weights :   [B, H, L, S]
            scores  :   [B, H, L, S]
        '''
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        q = queries.transpose(1,2)                      # [B, H, L, d_k]
        k = keys.transpose(1,2)                         # [B, H, S, d_k]
        v = values.transpose(1,2)                       # [B, H, S, d_k]
        scale = self.scale or 1. / sqrt(E)

        q_k_concat = torch.cat([q.unsqueeze(3).repeat(1, 1, 1, S, 1), 
                                k.unsqueeze(2).repeat(1, 1, L, 1, 1)], dim=-1)      # [B, H, L, S, 2*d_k]
        scores = self.score_linear(q_k_concat).squeeze(-1) * scale                  # (B, N, L, S)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        weights = self.dropout(torch.softmax(scores, dim=-1))
        output = torch.matmul(weights, v).permute(0, 2, 1, 3)

        if self.output_attention:
            return output.contiguous(), weights
        else:
            return output.contiguous(), None
        
class BilinearAttention(nn.Module):
    def __init__(self, d_model, n_heads=1, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(BilinearAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.bilinear_weight = nn.Parameter(torch.randn(d_model // n_heads, d_model // n_heads))

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        '''
        Input shape:
            q               : [B, L, H, d_k]
            k               : [B, S, H, d_k]
            v               : [B, S, H, d_k]
        Output shape:
            output  :   [B, L, H, d_v]
            weights :   [B, H, L, S]
            scores  :   [B, H, L, S]
        '''
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        q = queries.transpose(1,2)                      # [B, H, L, d_k]
        k = keys.permute(0,2,3,1)                       # [B, H, d_k, S]
        v = values.transpose(1,2)                       # [B, H, S, d_k]
        scale = self.scale or 1. / sqrt(E)
        scores = torch.bmm(torch.bmm(q.reshape(B*H, L, E), self.bilinear_weight.unsqueeze(0).expand(B*H, -1, -1)), k.reshape(B*H, D, S)).reshape(B, H, L, S) * scale

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        weights = self.dropout(torch.softmax(scores, dim=-1))
        output = torch.matmul(weights, v).permute(0, 2, 1, 3)

        if self.output_attention:
            return output.contiguous(), weights
        else:
            return output.contiguous(), None
        
class MinusAttention(nn.Module):
    def __init__(self, d_model, n_heads=1, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(MinusAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.score_linear = nn.Linear(d_model // n_heads, 1)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        '''
        Input shape:
            q               : [B, L, H, d_k]
            k               : [B, S, H, d_k]
            v               : [B, S, H, d_k]
        Output shape:
            output  :   [B, L, H, d_v]
            weights :   [B, H, L, S]
            scores  :   [B, H, L, S]
        '''
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        q = queries.transpose(1,2)                      # [B, H, L, d_k]
        k = keys.permute(0,2,1,3)                       # [B, H, d_k, S]
        v = values.transpose(1,2)                       # [B, H, S, d_k]
        scale = self.scale or 1. / sqrt(E)

        q_k_minus = q.unsqueeze(3).repeat(1, 1, 1, S, 1) - k.unsqueeze(2).repeat(1, 1, L, 1, 1)
        scores = self.score_linear(q_k_minus).squeeze(-1) * scale       

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        weights = self.dropout(torch.softmax(scores, dim=-1))
        output = torch.matmul(weights, v).permute(0, 2, 1, 3)

        if self.output_attention:
            return output.contiguous(), weights
        else:
            return output.contiguous(), None

class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        # real U = U_part(factor*ln(L_k))*L_q
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(
            L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(
            Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H,
                                                L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            # requires that L_Q == L_V, i.e. for self-attention only
            assert (L_Q == L_V)
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) /
                     L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[
                                                  None, :, None], index, :] = attn
            return context_in, attns
        else:
            return context_in, None

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * \
                 np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * \
            np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(
            queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(
            context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class ReformerLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, causal=False, bucket_size=4, n_hashes=4):
        super().__init__()
        self.bucket_size = bucket_size
        self.attn = LSHSelfAttention(
            dim=d_model,
            heads=n_heads,
            bucket_size=bucket_size,
            n_hashes=n_hashes,
            causal=causal
        )

    def fit_length(self, queries):
        # inside reformer: assert N % (bucket_size * 2) == 0
        B, N, C = queries.shape
        if N % (self.bucket_size * 2) == 0:
            return queries
        else:
            # fill the time series
            fill_len = (self.bucket_size * 2) - (N % (self.bucket_size * 2))
            return torch.cat([queries, torch.zeros([B, fill_len, C]).to(queries.device)], dim=1)

    def forward(self, queries, keys, values, attn_mask, tau, delta):
        # in Reformer: defalut queries=keys
        B, N, C = queries.shape
        queries = self.attn(self.fit_length(queries))[:, :N, :]
        return queries, None


class TwoStageAttentionLayer(nn.Module):
    '''
    The Two Stage Attention (TSA) Layer
    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    '''

    def __init__(self, configs,
                 seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1):
        super(TwoStageAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.time_attention = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                           output_attention=configs.output_attention), d_model, n_heads)
        self.dim_sender = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                       output_attention=configs.output_attention), d_model, n_heads)
        self.dim_receiver = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                         output_attention=configs.output_attention), d_model, n_heads)
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # Cross Time Stage: Directly apply MSA to each dimension
        batch = x.shape[0]
        time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
        time_enc, attn = self.time_attention(
            time_in, time_in, time_in, attn_mask=None, tau=None, delta=None
        )
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)

        # Cross Dimension Stage: use a small set of learnable vectors to aggregate and distribute messages to build the D-to-D connection
        dim_send = rearrange(dim_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b=batch)
        batch_router = repeat(self.router, 'seg_num factor d_model -> (repeat seg_num) factor d_model', repeat=batch)
        dim_buffer, attn = self.dim_sender(batch_router, dim_send, dim_send, attn_mask=None, tau=None, delta=None)
        dim_receive, attn = self.dim_receiver(dim_send, dim_buffer, dim_buffer, attn_mask=None, tau=None, delta=None)
        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)

        final_out = rearrange(dim_enc, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b=batch)

        return final_out

"""
Wavelet Attention
"""
class WaveletAttention(nn.Module):

    def __init__(self, in_channels, out_channels, seq_len_q, seq_len_kv, c=64,
                 k=8, ich=512,
                 L=3,
                 base='legendre',
                 initializer=None, T=1, activation='softmax', output_attention=False,
                 **kwargs):
        super(WaveletAttention, self).__init__()
        print('base', base)

        self.c = c
        self.k = k
        self.L = L
        self.T = T
        self.activation = activation
        H0, H1, G0, G1, PHI0, PHI1 = get_filter(base, k)
        H0r = H0 @ PHI0
        G0r = G0 @ PHI0
        H1r = H1 @ PHI1
        G1r = G1 @ PHI1

        H0r[np.abs(H0r) < 1e-8] = 0
        H1r[np.abs(H1r) < 1e-8] = 0
        G0r[np.abs(G0r) < 1e-8] = 0
        G1r[np.abs(G1r) < 1e-8] = 0
        self.max_item = 3

        self.register_buffer('ec_s', torch.Tensor(
            np.concatenate((H0.T, H1.T), axis=0)))
        self.register_buffer('ec_d', torch.Tensor(
            np.concatenate((G0.T, G1.T), axis=0)))

        self.register_buffer('rc_e', torch.Tensor(
            np.concatenate((H0r, G0r), axis=0)))
        self.register_buffer('rc_o', torch.Tensor(
            np.concatenate((H1r, G1r), axis=0)))

        self.Lk = nn.Linear(ich, c * k)
        self.Lq = nn.Linear(ich, c * k)
        self.Lv = nn.Linear(ich, c * k)
        self.out = nn.Linear(c * k, ich)

        self.output_attention = output_attention

    def forward(self, q, k, v, mask=None, tau=None, delta=None):
        B, N, H, E = q.shape  # (B, N, H, E) torch.Size([3, 768, 8, 2])
        _, S, _, _ = k.shape  # (B, S, H, E) torch.Size([3, 96, 8, 2])

        q = q.view(q.shape[0], q.shape[1], -1)  # (B, N, H*E)
        k = k.view(k.shape[0], k.shape[1], -1)
        v = v.view(v.shape[0], v.shape[1], -1)
        q = self.Lq(q)
        q = q.view(q.shape[0], q.shape[1], self.c, self.k)  # (B, N, E, H)
        k = self.Lk(k)
        k = k.view(k.shape[0], k.shape[1], self.c, self.k)
        v = self.Lv(v)
        v = v.view(v.shape[0], v.shape[1], self.c, self.k)

        if N > S:
            zeros = torch.zeros_like(q[:, :(N - S), :]).float()
            v = torch.cat([v, zeros], dim=1)
            k = torch.cat([k, zeros], dim=1)
        else:
            v = v[:, :N, :, :]
            k = k[:, :N, :, :]
        ns = math.floor(np.log2(N))
        nl = pow(2, math.ceil(np.log2(N)))
        extra_q = q[:, 0:nl - N, :, :]
        extra_k = k[:, 0:nl - N, :, :]
        extra_v = v[:, 0:nl - N, :, :]
        q = torch.cat([q, extra_q], 1)
        k = torch.cat([k, extra_k], 1)
        v = torch.cat([v, extra_v], 1)

        Ud = torch.jit.annotate(List[torch.Tensor], [])
        Us = torch.jit.annotate(List[torch.Tensor], [])

        attn_d_list, attn_s_list = [], []

        for i in range(ns - self.L):
            dq, q = self.wavelet_transform(q)
            dk, k = self.wavelet_transform(k)
            dv, v = self.wavelet_transform(v)  # (B, N, E, H)

            scores_d = torch.einsum("bxeh,byeh->bhxy", dq, dk) / math.sqrt(E)

            if self.activation == 'softmax':
                attn_d = F.softmax(scores_d / self.T, dim=-1)  # (B,H,q,k)
            elif self.activation == 'linear':
                attn_d = scores_d  # (B,H,q,k)
            elif self.activation == 'linear_norm':
                attn_d = scores_d  # (B,H,q,k)
                mins = attn_d.min(dim=-1).unsqueeze(-1).expand(-1, -1, -1, attn_d.shape[3])
                attn_d -= mins
                sums = attn_d.sum(dim=-1).unsqueeze(-1).expand(-1, -1, -1, attn_d.shape[3])
                attn_d /= sums
            Ud += [torch.einsum("bhxy,byeh->bxeh", attn_d, dv)]
            attn_d_list.append(attn_d)

            scores_s = torch.einsum("bxeh,byeh->bhxy", q, k) / math.sqrt(E)

            if self.activation == 'softmax':
                attn_s = F.softmax(scores_s / self.T, dim=-1)  # (B,H,q,k)
            elif self.activation == 'linear':
                attn_s = scores_s  # (B,H,q,k)
            elif self.activation == 'linear_norm':
                attn_s = scores_s  # (B,H,q,k)
                mins = attn_s.min(dim=-1).unsqueeze(-1).expand(-1, -1, -1, attn_s.shape[3])
                attn_s -= mins
                sums = attn_s.sum(dim=-1).unsqueeze(-1).expand(-1, -1, -1, attn_s.shape[3])
                attn_s /= sums
            Us += [torch.einsum("bhxy,byeh->bxeh", attn_s, v)]
            attn_s_list.append(attn_s)

        # reconstruct
        for i in range(ns - 1 - self.L, -1, -1):
            v = v + Us[i]
            v = torch.cat((v, Ud[i]), -1)
            v = self.evenOdd(v)
        v = self.out(v[:, :N, :, :].contiguous().view(B, N, -1))
        if self.output_attention == False:
            return (v.contiguous(), None)
        else:
            return (v.contiguous(), (attn_s_list, attn_d_list))

    def wavelet_transform(self, x):
        xa = torch.cat([x[:, ::2, :, :],
                        x[:, 1::2, :, :],
                        ], -1)
        d = torch.matmul(xa, self.ec_d)
        s = torch.matmul(xa, self.ec_s)
        return d, s

    def evenOdd(self, x):
        B, N, c, ich = x.shape  # (B, N, c, k)
        assert ich == 2 * self.k
        x_e = torch.matmul(x, self.rc_e)
        x_o = torch.matmul(x, self.rc_o)

        x = torch.zeros(B, N * 2, c, self.k,
                        device=x.device)
        x[..., ::2, :, :] = x_e
        x[..., 1::2, :, :] = x_o
        return x

class FourierCrossAttentionW(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len_q, seq_len_kv, modes=16, activation='tanh',
                 mode_select_method='random'):
        super(FourierCrossAttentionW, self).__init__()
        print('corss fourier correlation used!')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes
        self.activation = activation

    def forward(self, q, k, v, mask=None, tau=None, delta=None):
        B, L, E, H = q.shape

        xq = q.permute(0, 3, 2, 1)  # size = [B, H, E, L] torch.Size([3, 8, 64, 512])
        xk = k.permute(0, 3, 2, 1)
        xv = v.permute(0, 3, 2, 1)
        self.index_q = list(range(0, min(int(L // 2), self.modes1)))
        self.index_k_v = list(range(0, min(int(xv.shape[3] // 2), self.modes1)))

        # Compute Fourier coefficients
        xq_ft_ = torch.zeros(B, H, E, len(self.index_q), device=xq.device, dtype=torch.cfloat)
        xq_ft = torch.fft.rfft(xq, dim=-1)
        for i, j in enumerate(self.index_q):
            xq_ft_[:, :, :, i] = xq_ft[:, :, :, j]

        xk_ft_ = torch.zeros(B, H, E, len(self.index_k_v), device=xq.device, dtype=torch.cfloat)
        xk_ft = torch.fft.rfft(xk, dim=-1)
        for i, j in enumerate(self.index_k_v):
            xk_ft_[:, :, :, i] = xk_ft[:, :, :, j]
        xqk_ft = (torch.einsum("bhex,bhey->bhxy", xq_ft_, xk_ft_))
        if self.activation == 'tanh':
            xqk_ft = xqk_ft.tanh()
        elif self.activation == 'softmax':
            xqk_ft = torch.softmax(abs(xqk_ft), dim=-1)
            xqk_ft = torch.complex(xqk_ft, torch.zeros_like(xqk_ft))
        else:
            raise Exception('{} actiation function is not implemented'.format(self.activation))
        xqkv_ft = torch.einsum("bhxy,bhey->bhex", xqk_ft, xk_ft_)

        xqkvw = xqkv_ft
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=xq.device, dtype=torch.cfloat)
        for i, j in enumerate(self.index_q):
            out_ft[:, :, :, j] = xqkvw[:, :, :, i]

        out = torch.fft.irfft(out_ft / self.in_channels / self.out_channels, n=xq.size(-1)).permute(0, 3, 2, 1)
        # size = [B, L, H, E]
        return (out, None)