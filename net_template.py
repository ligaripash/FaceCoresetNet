from collections import namedtuple
import torch
import torch.nn as nn
from torch.nn import Dropout
from torch.nn import MaxPool2d
from torch.nn import Sequential
from torch.nn import Conv2d, Linear
from torch.nn import BatchNorm1d, BatchNorm2d
from torch.nn import ReLU, Sigmoid
from torch.nn import Module
from torch.nn import PReLU
from typing import Optional, Any, Union, Callable
import numpy as np
from torch import Tensor
from torch.nn import functional as F
from torch.nn import ReLU
from torch.nn import Module
from torch.nn import MultiheadAttention
from torch.nn import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn import Dropout
from torch.nn import Linear
from torch.nn import LayerNorm

import pickle
import os
torch.autograd.set_detect_anomaly(True)




def angluar_dist_with_norm(norm, a, b):
    norm_a = a / np.linalg.norm(a)
    norm_b = b / np.linalg.norm(b)
    dist = (1 - np.dot(norm_a, norm_b)) / 2
    gamma = 0.01
    dist = np.power(norm, gamma) * dist
    return dist


def incremental_farthest_search_norms(points, k, norms):
    remaining_points = points[:].tolist()
    norms = norms[:].tolist()
    solution_set = []
    max_index = np.argmax(norms)
    solution_set.append(remaining_points.pop(max_index))
    norms.pop(max_index)
    for _ in range(k-1):
        #distances = [angluar_dist(p, solution_set[0]) for p in remaining_points]
        distances = [np.Inf] * len(remaining_points)
        for i, p in enumerate(remaining_points):
            norm = norms[i]
            for j, s in enumerate(solution_set):
                distances[i] = min(distances[i], angluar_dist_with_norm(norm, p, s))
        max_index = distances.index(max(distances))
        solution_set.append(remaining_points.pop(max_index))
        norms.pop(max_index)
    return np.array(solution_set)

def get_proposal_pos_embed(real_norms, embed_dim):
    # batch_size = real_norms.size(0)
    # max_seq_length = real_norms.size(1)
    #
    # position_enc = torch.zeros(batch_size, max_seq_length, embed_dim,device=real_norms.device)
    # div_term = torch.exp(torch.arange(0, embed_dim, 2, device=real_norms.device).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
    #
    # position_enc[:, :, 0::2] = torch.sin(real_norms.unsqueeze(-1) * div_term)
    # position_enc[:, :, 1::2] = torch.cos(real_norms.unsqueeze(-1) * div_term)
    # return position_enc


    proposals = real_norms
    number_of_points_to_encode = 1
    num_pos_feats = embed_dim / number_of_points_to_encode
    temperature = 10000
    scale = 2 * torch.pi

    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
    dim_t_div_2 = torch.div(dim_t, 2, rounding_mode='floor')
    dim_t = temperature ** (2 * (dim_t_div_2) / num_pos_feats)
    # N, L, 4
    #proposals = proposals.sigmoid() * scale
    #gil - pass proposals act
    proposals = proposals * scale
    # N, L, 4, 128
    pos = proposals[:, :, None] / dim_t
    # N, L, 4, 64, 2
    pos = torch.stack((pos[:, :, 0::2].sin(), pos[:, :, 1::2].cos()), dim=3).flatten(2)
    return pos

class TransformerDecoderLayerOrig(Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to self attention, multihead
            attention and feedforward operations, respectively. Otherwise it's done after.
            Default: ``False`` (after).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)

    Alternatively, when ``batch_first`` is ``True``:
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> memory = torch.rand(32, 10, 512)
        >>> tgt = torch.rand(32, 20, 512)
        >>> out = decoder_layer(tgt, memory)
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn1 = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        self.self_attn2 = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)

        self.multihead_attn1 = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                 **factory_kwargs)

        self.multihead_attn2 = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                  **factory_kwargs)

        self.multihead_attn3 = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                  **factory_kwargs)

        # Implementation of Feedforward model
        #self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        #self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm4 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        self.dropout4 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)


    def forward(self,core_template, norm_encoding_core_template, template_features, norm_encoding_template):

        #for the template is larger than the core template, do cross attention
        q = core_template + norm_encoding_core_template
        q = self.norm1(q + self._sa_block1(q, None, None, None))
        # q = self.norm2(q + self._sa_block2(q, None, None, None))
        k = template_features + norm_encoding_template
        v = template_features
        #v = template_features + norm_encoding_template
        q = self.norm2(core_template + self._my_mha_block1(q, k, v))
        #q = self.norm3(core_template + self._my_mha_block2(q, k, v))

        #v = template_features
        #q = self.norm4(core_template + self._my_mha_block3(q, k, v))
        #q = self.norm4(q + self._my_mha_block3(q, k, v))

        return q

        # if self.norm_first:
        #     x = x + self._sa_block(self.norm1(x), None, None, None)
        #     x = x + self._mha_block(self.norm2(x), None, None, None, None)
        #     x = x + self._ff_block(self.norm3(x))
        # else:
        #     x = self.norm1(x + self._sa_block(x, None, None, None))
        #     x = self.norm2(x + self._my_mha_block(x, memory, None, None, None))
        #     x = self.norm3(x + self._ff_block(x))
        #
        # return x

    def orig_forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            tgt_is_causal: If specified, applies a causal mask as tgt mask.
                Mutually exclusive with providing tgt_mask. Default: ``False``.
            memory_is_causal: If specified, applies a causal mask as tgt mask.
                Mutually exclusive with providing memory_mask. Default: ``False``.
        Shape:
            see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask, memory_is_causal)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask, memory_is_causal))
            x = self.norm3(x + self._ff_block(x))

        return x


    # self-attention block
    def _sa_block1(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.self_attn1(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    def _sa_block2(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.self_attn2(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout4(x)

    def _my_mha_block1(self, q: Tensor, k: Tensor, v: Tensor):
        x = self.multihead_attn1(q, k, v,
                                attn_mask=None,
                                key_padding_mask=None,
                                need_weights=False)[0]
        return self.dropout2(x)

    def _my_mha_block2(self, q: Tensor, k: Tensor, v: Tensor):
        x = self.multihead_attn2(q, k, v,
                                attn_mask=None,
                                key_padding_mask=None,
                                need_weights=False)[0]
        return self.dropout3(x)

    def _my_mha_block3(self, q: Tensor, k: Tensor, v: Tensor):
        x = self.multihead_attn3(q, k, v,
                                attn_mask=None,
                                key_padding_mask=None,
                                need_weights=False)[0]
        return self.dropout4(x)

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)



def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))



class MyTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_heads=8):
        super().__init__()

        # cross attention
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,batch_first=True)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = torch.nn.functional.relu
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        #self.dropout5 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        #self.norm4 = nn.LayerNorm(d_model)


    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt


    def def_forward_no_pos(self, queries, queries_norm_enc, template, template_norm_enc):
        q = k = v = queries
        sa_norm_q = self.self_attn(q, k, v)[0]
        sa_norm_q = q + self.dropout2(sa_norm_q)
        sa_norm_q = self.norm2(sa_norm_q)

        q = sa_norm_q
        k = template
        v = template
        # cross attention
        ca_norm_q = self.cross_attn(q, k, v)[0]
        ca_norm_q = q + self.dropout1(ca_norm_q)
        ca_norm_q = self.norm1(ca_norm_q)
        # ffn
        ca_norm_q = self.forward_ffn(ca_norm_q)

        return ca_norm_q

    def forward(self, queries, queries_norm_enc, template, template_norm_enc):
        # self attention
        return self.def_forward_no_pos(queries, queries_norm_enc, template, template_norm_enc)

        q = k = v = queries + queries_norm_enc
        sa_norm_q = self.self_attn(q, k, v)[0]
        sa_norm_q = q + self.dropout2(sa_norm_q)
        sa_norm_q = self.norm2(sa_norm_q)

        q = sa_norm_q
        k = template + template_norm_enc
        v = template
        # cross attention
        ca_norm_q = self.cross_attn(q, k, v)[0]
        ca_norm_q = q + self.dropout1(ca_norm_q)
        ca_norm_q = self.norm1(ca_norm_q)
        # ffn
        ca_norm_q = self.forward_ffn(ca_norm_q)

        return ca_norm_q

class TemplateAggregateModel(nn.Module):
    """
    Get a set of tramplate features and aggregate them to one descriptor for the template
    """
    def __init__(self, embedding_size=512, coreset_size=5):
        super(TemplateAggregateModel, self).__init__()
        self.coreset_size = coreset_size
        self.embedding_size = embedding_size
        self.gamma_base = torch.tensor(0.0)
        self.gamma = nn.Parameter(torch.tensor(0.01))
        self.tau = nn.Parameter(torch.tensor(0.0))
        self.feature_norm_normalizer = nn.Parameter(torch.logit(torch.tensor(1/20.0)))
        # self.gamma = nn.Parameter(torch.tensor(0.2))
        # self.alpha = nn.Parameter(torch.logit(torch.tensor(0.4)))
        #self.gamma = nn.Parameter(torch.tensor(0.0))
        self.ur_center = nn.Parameter(torch.empty(embedding_size))
        self.dist_from_ur_center_norm_size_balance = nn.Parameter(torch.logit(torch.tensor(0.2)))
        #nn.init.trunc_normal_(self.ur_center)
        # with open('/home/gilsh/std_low_detectability_ir101_ms1mv2.pickle', 'rb') as f:
        #      self.ur_center_empirical = torch.tensor(pickle.load(f))
        #      self.ur_center = nn.Parameter(self.ur_center_empirical)
        #      self.ur_center.requires_grad = False
            #self.ur_center = self.ur_center_empirical

        #self.encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_size, nhead=8, batch_first=True, dropout=0.2)
        #self.my_decoder_layer = MyTransformerDecoderLayer(d_model=embedding_size, n_heads=8,d_ffn=1024,dropout=0.2 )
        self.decoder_layer1 = TransformerDecoderLayerOrig(d_model=embedding_size, nhead=8, batch_first=True, dropout=0.1,dim_feedforward=1024)
        #self.decoder_layer2 = nn.TransformerDecoderLayer(d_model=embedding_size, nhead=8, batch_first=True)
        #self.feature_agg = MultiHeadAttAggregate(d_model=512, nhead=8)
        #self.pool_vector_scaler = nn.Linear(512, 1)
        #self.pos_trans = torch.nn.Linear(embedding_size, embedding_size)
        self.pos_trans_norm = nn.LayerNorm(embedding_size)
        self.quality_measure = nn.Linear(embedding_size, 1)


        #self.register_parameter(name='gamma', param=self.gamma)
        #self.register_parameter(name='dist_from_ur_center_norm_size_balance', param=self.dist_from_ur_center_norm_size_balance)


    def angluar_dist_with_norm(self, candidate_point, norms, input_points):
        """
        a, b are normalized face embeddings.
        Args:
            norm:
            a:
            b:

        Returns:

        """
        input_points = torch.nn.functional.normalize(input_points, p=2.0, dim = -1)
        candidate_point = torch.nn.functional.normalize(candidate_point, p=2.0, dim = -1)

        confidence_by_norms_and_UR = False
        N = input_points.shape[1]
        batch_size = candidate_point.shape[0]
        inner_product = torch.bmm(input_points, candidate_point.transpose(1,2))
        dist = (1 - inner_product) / 2

        if confidence_by_norms_and_UR:
            ur_center_unit = self.ur_center / self.ur_center.norm()
            ur_center_cadidate_inner_product = torch.einsum('ijk,ijk->ij', input_points,
                                                         ur_center_unit.expand(batch_size, N, 512))

            ur_center_cadidate_dist = (1 - ur_center_cadidate_inner_product) / 2
            #The candidate inportance is poportional to the distance of the feature to the UR center and to the feature Norm.
            candidate_importance = self.dist_from_ur_center_norm_size_balance.sigmoid() * ur_center_cadidate_dist +  \
             (1 - self.dist_from_ur_center_norm_size_balance.sigmoid()) * norms.squeeze(-1)
            #candidate_importance = ur_center_cadidate_dist
        else:
            candidate_importance = norms

        candidate_importance = candidate_importance.reshape_as(dist)
        #dist = torch.pow(candidate_importance, self.gamma.relu()) * dist

        quality_dist = torch.pow(candidate_importance, self.gamma_base + self.gamma) * dist

        # grad_hook_quality_dist = quality_dist.register_hook(
        #     lambda grad: print("quality_dist is {0}".format(grad.data.norm(2))))

        #dist = dist.clip(0,1)
        return quality_dist

    def pop(self, t, max_indices, non_max_indices):
        """
        The input is a [N, T, 512] tensor with index to the template dimension.
        Gather the indexed values and remove them from the tensor.
        pop index
        Args:
            t:
            index:

        Returns:
            (N, 1, 512) poped tensor
            (N, T-1, 512) tensor

        """
        if len(max_indices.shape) != len(t.shape):
            a = max_indices[:, :, None].expand([max_indices.shape[0], max_indices.shape[1], t.shape[-1]])
            try:
                popped_tensor = torch.gather(t, 1, a)
            except:
                pass

            a = non_max_indices[:, :, None].expand([non_max_indices.shape[0], non_max_indices.shape[1], t.shape[-1]])
            tensor_after_pop = torch.gather(t, 1, a)
        else:
            popped_tensor = torch.gather(t, 1, max_indices)
            tensor_after_pop = torch.gather(t, 1, non_max_indices)

        return popped_tensor, tensor_after_pop

    def incremental_farthest_search_norms(self, points, k, norms):
        batch_size = points.shape[0]
        template_size = points.shape[1]
        remaining_points = points
        #The solution set has a shape of (#batch, k, 512) and holds the output fps sample
        solution_set = torch.empty((points.shape[0], k, points.shape[-1]), device=points.device, requires_grad=True)

        #The input we sample from - shape (batch, N, 512)
        input_points = points
        N = points.shape[1]

        #We use a mask in the size of the input to mark which points belong to the output
        #We start by adding the points with max norms (the best quality)

        solution_mask = torch.nn.functional.gumbel_softmax(100000000 * norms, hard=True, tau=0.000000000000001, dim=1).bool()
        #We now have a single feature for each batch in our solution.
        #We perform an additional k-1 iterations so we have k features in our solution in the end
        #In each iteration we choose the feature which is farthest from the current solution

        remaining_points = input_points[~solution_mask].reshape(batch_size, N-1,points.shape[2] )
        remaining_norms = norms[~solution_mask].reshape(batch_size, N-1)
        solution_points = input_points[solution_mask]
        solution_norms = norms[solution_mask]

        for solution_size in range(1, k):
            remaining_points_count = N - solution_size
            distances = torch.ones(batch_size, remaining_points_count, device=points.device) * torch.tensor(torch.finfo(float).max)
            for i in range(remaining_points_count):
                #iteration on the remaining points (not in the solution set):
                #Take a point from the remaining point and compute the distances to all points in the solution set
                p = remaining_points[:,i,:]
                #norm = remaining_norms[:,i]
            #for i, p in enumerate(remaining_points):
                for j in range(solution_size):
                    # For each point outside the solution set, compute the distance to each point the solution set
                #for j, s in enumerate(solution_set):
                    s = solution_set[:,j, :]
                    dist_p_s = self.angluar_dist_with_norm(p, s)
                    distances_new = distances.clone()
                    distances_new[:,i] = torch.min(distances[:,i], dist_p_s)
                    distances = distances_new

            #Now distances holds for each point in the 'remaining points' the minimal distance to the solution set
            #We have to choose the points farther away from the solution set
            #max_values, max_indices = torch.max(distances, dim=1)
            max_mask = torch.nn.functional.gumbel_softmax(100000000 * distances, hard=True, tau=0.000000000000001, dim=1)
            selected_point = remaining_points[max_mask]
            solution_points = torch.cat((solution_points, selected_point),1)
            selected_norm = remaining_norms[max_mask]
            solution_norms = torch.cat((solution_norms, selected_norm), 1)

        return solution_points

#Pytorch farthest point sampling

    def farthest_point_sample_copilot(self, x, npoint, norms):

        """
        Input:
            x: pointcloud data, [B, N, C]
            npoint: number of samples
        Return:
            centroids: sampled pointcloud index, [B, npoint]
        """

        if self.training:
            scaler = 1.0
            #tau = self.tau.sigmoid()
            tau = 1.0
            #tau = 1e-10
        else:
            scaler = 1e4
            tau = 1e-10

        #select the feature with highest quality
        norms = norms.unsqueeze(-1)
        max_dist_mask = torch.nn.functional.gumbel_softmax(norms * scaler, hard=True, tau=tau,
                                                           dim=1)
        #max_dist_mask = max_dist_mask.unsqueeze(-1)
        #max_dist_mask.requires_grad = True
        # grad_hook3 = max_dist_mask.register_hook(
        #     lambda grad: print("max dist Grad is {0}".format(grad.data.norm(2))))

        # centroids = torch.zeros(B, npoint, C, device=x.device)
        # centroids.requires_grad = True
        # grad_hook4 = centroids.register_hook(
        #     lambda grad: print("centroids Grad is {0}".format(grad.data.norm(2))))

        # gamma_hook = self.gamma.register_hook(
        #     lambda grad: print("gamma grad is {0}".format(grad.data)))

        #extract the best quality feature
        #core_template = (max_dist_mask * x).sum(1)
        #core_template = core_template.unsqueeze(1)
        core_template = (x.transpose(1,2) @ max_dist_mask).transpose(1,2)
        core_template.require_grad = True
        # grad_hook5 = core_template.register_hook(
        #     lambda grad: print("current_farthest Grad is {0}".format(grad.data.norm(2))))

        #After each extraction compute the distance from the coretemplate to the whule template x
        dist_core_template_to_template = self.angluar_dist_with_norm(core_template, norms, x)
        dist_core_template_to_template.require_grad = True
        # grad_hook6 = dist_core_template_to_template.register_hook(
        #     lambda grad: print("dist_core_template_to_template Grad is {0}".format(grad.data.norm(2))))

        #dist_core_template_to_template = dist_core_template_to_template.squeeze(1)

        for i in range(npoint-1):
            #Extract index of new point
            max_dist_mask = torch.nn.functional.gumbel_softmax(dist_core_template_to_template * scaler , hard=True, tau=tau,
                                                           dim=1)
            # max_dist_mask_hook = max_dist_mask.register_hook(
            #     lambda grad: print("max_dist_mask_hook Grad is {0}".format(grad.data.norm(2))))

            # extract the best quality feature
            # new_core_item = (max_dist_mask * x).sum(1)
            # new_core_item = new_core_item.unsqueeze(1)
            new_core_item = (x.transpose(1,2) @ max_dist_mask).transpose(1,2)

            dist_new_selected_to_template = self.angluar_dist_with_norm(new_core_item, norms, x)
            dist_core_template_to_template = torch.min(dist_core_template_to_template, dist_new_selected_to_template)

            core_template = torch.cat([core_template, new_core_item], dim=1)
            # grad_hook7 = core_template.register_hook(
            #     lambda grad: print("core_template7 Grad is {0}".format(grad.data.norm(2))))

        return core_template





    def incremental_farthest_search_norms_(self, points, k, norms):
        #remaining_points = points[:].tolist()
        batch_size = points.shape[0]
        template_size = points.shape[1]
        remaining_points = points
        #norms = norms[:].tolist()
        #The solution set shape is (N, K, 512)
        #The points shape is (N,T,512) where T is the template size
        solution_set = torch.empty((points.shape[0], k, points.shape[-1]), device=points.device, requires_grad=True)
        #The first sampled point is the point with maximum norm (best quality)
        max_value, max_indices = torch.max(norms, dim=-1)
        max_mask = torch.zeros_like(norms).scatter_(1, max_indices.unsqueeze(1), 1)
        _, non_max_indices = torch.where(max_mask == 0)
        non_max_indices = non_max_indices.reshape(points.shape[0],-1)
        max_indices = max_indices.reshape(points.shape[0],-1)


        popped_feature_tensor, remaining_points = self.pop(remaining_points, max_indices, non_max_indices)

        solution_index = 0
        solution_set = solution_set.clone()
        solution_set[:,solution_index, :] = popped_feature_tensor.squeeze()
        solution_index += 1
        remaining_points_count = template_size - solution_index
        _, remaining_norms = self.pop(norms, max_indices, non_max_indices)
        # solution_set.append(remaining_points.pop(max_index))
        #norms.pop(max_index)
        for _ in range(k - 1):
            #iteration on the ouput k - 1 times (we already have 1 in the solution set)
            # distances = [angluar_dist(p, solution_set[0]) for p in remaining_points]
            #distances = [torch.tensor(np
            # .Inf)] * len(remaining_points)
            distances = torch.ones(batch_size, remaining_points_count, device=points.device) * torch.tensor(torch.finfo(float).max)
            for i in range(remaining_points_count):
                #iteration on the remaining points (not in the solution set):
                #Take a point from the remaining point and compute the distances to all points in the solution set
                p = remaining_points[:,i,:]
                #norm = remaining_norms[:,i]
            #for i, p in enumerate(remaining_points):
                for j in range(solution_index):
                    # For each point outside the solution set, compute the distance to each point the solution set
                #for j, s in enumerate(solution_set):
                    s = solution_set[:,j, :]
                    dist_p_s = self.angluar_dist_with_norm(p, s)
                    distances_new = distances.clone()
                    distances_new[:,i] = torch.min(distances[:,i], dist_p_s)
                    distances = distances_new

            #Now distances holds for each point in the 'remaining points' the minimal distance to the solution set
            #We have to choose the points farther away from the solution set
            #max_values, max_indices = torch.max(distances, dim=1)
            max_mask = torch.nn.functional.gumbel_softmax(100000000 * distances, hard=True, tau=0.000000000000001, dim=1)

            # max_mask = torch.autograd.Variable(torch.zeros_like(distances), requires_grad=True)
            # max_mask_clone = max_mask.clone()
            # max_mask_clone.scatter_(1, max_indices.unsqueeze(1), torch.ones_like(max_indices.unsqueeze(1), dtype=torch.float32))

            _, non_max_indices = torch.where(max_mask == 0)
            non_max_indices = non_max_indices.reshape(batch_size, -1)
            max_indices = max_indices.reshape(batch_size, -1)
            popped_feature_tensor, remaining_points = self.pop(remaining_points, max_indices, non_max_indices)
            solution_set = solution_set.clone()
            solution_set[:, solution_index, :] = popped_feature_tensor.squeeze()
            solution_index += 1
            remaining_points_count = template_size - solution_index
            _, remaining_norms = self.pop(remaining_norms, max_indices, non_max_indices)

            # solution_set.append(remaining_points.pop(max_index))
            # norms.pop(max_index)
        return solution_set

    def aggregate_fps_with_norm_priority(self, template_features, template_norms):
        """
        Combine FPS with norm priority.
        Merge the norm size with farthest point in one metric.
        """

        # temlate features has the shape (N, L, 512) where L is the template size
        # with torch.no_grad():
        #     self.gamma = torch.nn.Parameter(torch.clip(self.gamma, min=0.0001, max=0.2))
        #     self.alpha = torch.nn.Parameter(torch.clip(self.alpha, min=0.0001, max=1.0))
        #Gil
        if self.training:
            K = self.coreset_size
        else:
            K = self.coreset_size

        number_features = template_features.shape[1]
        actual_K = torch.min(torch.tensor([K, torch.tensor(number_features)]))

        #normalized_norms = template_norms * self.feature_norm_normalizer.sigmoid()
        template_features = template_norms.unsqueeze(-1).repeat(1,1,template_features.shape[-1]) * template_features
        template_features.require_grad = True
        template_norms.require_grad = True
        core_template_orig = self.farthest_point_sample_copilot(template_features, actual_K, template_norms)
        core_template = core_template_orig
        norm_encoding_template = get_proposal_pos_embed(template_norms, self.embedding_size)
        #norm_encoding_template = self.pos_trans_norm(self.pos_trans(norm_encoding_template))
        norm_encoding_template = self.pos_trans_norm(norm_encoding_template)

        core_template_norms = core_template.norm(dim=2)
        #core_template_norms = (self.quality_measure(core_template).squeeze(-1) * core_template_norms).sigmoid() + 1.0
        norm_encoding_core_template = get_proposal_pos_embed(core_template_norms, self.embedding_size)
        #norm_encoding_core_template = self.pos_trans_norm(self.pos_trans(norm_encoding_core_template))
        norm_encoding_core_template = self.pos_trans_norm(norm_encoding_core_template)

        #FPS_sample_pos = self.encoder_layer(FPS_sample_pos)

        #delta = self.my_decoder_layer(core_template, norm_encoding_core_template, template_features, norm_encoding_template)
        delta = self.decoder_layer1(core_template, norm_encoding_core_template, template_features, norm_encoding_template)
        #delta = self.decoder_layer2(decoded_features, template_features)
        #template_mean_mult = template_mean.repeat((1, FPS_sample.shape[1], 1))
        #FPS_sample = FPS_sample + template_mean_mult

        core_template = core_template + delta
        agg = torch.mean(core_template, dim=1)
        if 0:
            agg_max, _ = torch.max(core_template, axis=1)
            agg_min, _ = torch.min(core_template, axis=1)
            agg_abs_max = torch.abs(agg_max)
            agg_abs_min = torch.abs(agg_min)
            agg = torch.where(agg_abs_max > agg_abs_min, agg_max, agg_min)

        return agg, core_template_orig

    def forward(self, template_features, template_norms):
        # Template batch shape is [N, T, 512]
        template_norms = torch.squeeze(template_norms, dim=-1)
        aggregated_feature, FPS_sample = self.aggregate_fps_with_norm_priority(template_features, template_norms)
        norms = aggregated_feature.norm(dim=-1).unsqueeze(-1)
        aggregated_feature_norm = aggregated_feature / norms
        return aggregated_feature_norm, norms, FPS_sample


########################################################################################################################

class MultiHeadAttAggregate(nn.Module):
    def __init__(self, d_model=512, nhead=8):
        super(MultiHeadAttAggregate, self).__init__()
        self.enc_layer1 = nn.TransformerEncoderLayer(d_model, nhead,batch_first=True)
        self.enc_layer2 = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.template_agg = nn.Parameter(torch.empty(1,1,d_model))
        nn.init.trunc_normal_(self.template_agg)


    def forward(self, x, pooled_tensor=None):
        avg = torch.mean(x, dim=1, keepdim=True)
        x = x - avg
        pooled_tensor = pooled_tensor.unsqueeze(1) - avg
        agg_broadcast = torch.broadcast_to(self.template_agg, (x.shape[0], 1, x.shape[2]))
        pooled_tensor = pooled_tensor + agg_broadcast
        concat_input = torch.cat([pooled_tensor, x], dim=1)
        #t_concat_input = concat_input.transpose(0,1)
        output = self.enc_layer1(concat_input)
        output = self.enc_layer2(output)

        #output = output + avg
        #return output

        return output[:,0,:].squeeze(1) + avg.squeeze(1)


########################################################################################################################


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        x_shape = [x for x in x.shape]
        x_reshape = x.reshape([-1] + x_shape[-3:])
        features, norms = self.module(x_reshape)
        features = features.reshape((x_shape[0:2] + [-1]))
        norms = norms.reshape((x_shape[0:2] + [-1]))
        return features, norms



def build_model(model_name='ir_50'):
    if model_name == 'ir_101':
        return IR_101(input_size=(112,112))
    elif model_name == 'ir_50':
        return IR_50(input_size=(112,112))
    elif model_name == 'ir_se_50':
        return IR_SE_50(input_size=(112,112))
    elif model_name == 'ir_34':
        return IR_34(input_size=(112,112))
    elif model_name == 'ir_18':
        return IR_18(input_size=(112,112))
    else:
        raise ValueError('not a correct model name', model_name)

def initialize_weights(modules):
    """ Weight initilize, conv2d and linear is initialized with kaiming_normal
    """
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight,
                                    mode='fan_out',
                                    nonlinearity='relu')
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight,
                                    mode='fan_out',
                                    nonlinearity='relu')
            if m.bias is not None:
                m.bias.data.zero_()


class Flatten(Module):
    """ Flat tensor
    """
    def forward(self, input):
        return input.view(input.size(0), -1)


class LinearBlock(Module):
    """ Convolution block without no-linear activation layer
    """
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(LinearBlock, self).__init__()
        self.conv = Conv2d(in_c, out_c, kernel, stride, padding, groups=groups, bias=False)
        self.bn = BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class GNAP(Module):
    """ Global Norm-Aware Pooling block
    """
    def __init__(self, in_c):
        super(GNAP, self).__init__()
        self.bn1 = BatchNorm2d(in_c, affine=False)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn2 = BatchNorm1d(in_c, affine=False)

    def forward(self, x):
        x = self.bn1(x)
        x_norm = torch.norm(x, 2, 1, True)
        x_norm_mean = torch.mean(x_norm)
        weight = x_norm_mean / x_norm
        x = x * weight
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        feature = self.bn2(x)
        return feature


class GDC(Module):
    """ Global Depthwise Convolution block
    """
    def __init__(self, in_c, embedding_size):
        super(GDC, self).__init__()
        self.conv_6_dw = LinearBlock(in_c, in_c,
                                     groups=in_c,
                                     kernel=(7, 7),
                                     stride=(1, 1),
                                     padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = Linear(in_c, embedding_size, bias=False)
        self.bn = BatchNorm1d(embedding_size, affine=False)

    def forward(self, x):
        x = self.conv_6_dw(x)
        x = self.conv_6_flatten(x)
        x = self.linear(x)
        x = self.bn(x)
        return x


class SEModule(Module):
    """ SE block
    """
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(channels, channels // reduction,
                          kernel_size=1, padding=0, bias=False)

        nn.init.xavier_uniform_(self.fc1.weight.data)

        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(channels // reduction, channels,
                          kernel_size=1, padding=0, bias=False)

        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return module_input * x



class BasicBlockIR(Module):
    """ BasicBlock for IRNet
    """
    def __init__(self, in_channel, depth, stride):
        super(BasicBlockIR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            BatchNorm2d(depth),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut


class BottleneckIR(Module):
    """ BasicBlock with bottleneck for IRNet
    """
    def __init__(self, in_channel, depth, stride):
        super(BottleneckIR, self).__init__()
        reduction_channel = depth // 4
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, reduction_channel, (1, 1), (1, 1), 0, bias=False),
            BatchNorm2d(reduction_channel),
            PReLU(reduction_channel),
            Conv2d(reduction_channel, reduction_channel, (3, 3), (1, 1), 1, bias=False),
            BatchNorm2d(reduction_channel),
            PReLU(reduction_channel),
            Conv2d(reduction_channel, depth, (1, 1), stride, 0, bias=False),
            BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut


class BasicBlockIRSE(BasicBlockIR):
    def __init__(self, in_channel, depth, stride):
        super(BasicBlockIRSE, self).__init__(in_channel, depth, stride)
        self.res_layer.add_module("se_block", SEModule(depth, 16))


class BottleneckIRSE(BottleneckIR):
    def __init__(self, in_channel, depth, stride):
        super(BottleneckIRSE, self).__init__(in_channel, depth, stride)
        self.res_layer.add_module("se_block", SEModule(depth, 16))


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''


def get_block(in_channel, depth, num_units, stride=2):

    return [Bottleneck(in_channel, depth, stride)] +\
           [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]


def get_blocks(num_layers):
    if num_layers == 18:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=2),
            get_block(in_channel=64, depth=128, num_units=2),
            get_block(in_channel=128, depth=256, num_units=2),
            get_block(in_channel=256, depth=512, num_units=2)
        ]
    elif num_layers == 34:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=6),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=256, num_units=3),
            get_block(in_channel=256, depth=512, num_units=8),
            get_block(in_channel=512, depth=1024, num_units=36),
            get_block(in_channel=1024, depth=2048, num_units=3)
        ]
    elif num_layers == 200:
        blocks = [
            get_block(in_channel=64, depth=256, num_units=3),
            get_block(in_channel=256, depth=512, num_units=24),
            get_block(in_channel=512, depth=1024, num_units=36),
            get_block(in_channel=1024, depth=2048, num_units=3)
        ]

    return blocks


class Backbone(Module):
    def __init__(self, input_size, num_layers, mode='ir'):
        """ Args:
            input_size: input_size of backbone
            num_layers: num_layers of backbone
            mode: support ir or irse
        """
        super(Backbone, self).__init__()
        assert input_size[0] in [112, 224], \
            "input_size should be [112, 112] or [224, 224]"
        assert num_layers in [18, 34, 50, 100, 152, 200], \
            "num_layers should be 18, 34, 50, 100 or 152"
        assert mode in ['ir', 'ir_se'], \
            "mode should be ir or ir_se"
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64), PReLU(64))
        blocks = get_blocks(num_layers)
        if num_layers <= 100:
            if mode == 'ir':
                unit_module = BasicBlockIR
            elif mode == 'ir_se':
                unit_module = BasicBlockIRSE
            output_channel = 512
        else:
            if mode == 'ir':
                unit_module = BottleneckIR
            elif mode == 'ir_se':
                unit_module = BottleneckIRSE
            output_channel = 2048

        if input_size[0] == 112:
            self.output_layer = Sequential(BatchNorm2d(output_channel),
                                        Dropout(0.4), Flatten(),
                                        Linear(output_channel * 7 * 7, 512),
                                        BatchNorm1d(512, affine=False))
        else:
            self.output_layer = Sequential(
                BatchNorm2d(output_channel), Dropout(0.4), Flatten(),
                Linear(output_channel * 14 * 14, 512),
                BatchNorm1d(512, affine=False))

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel, bottleneck.depth,
                                bottleneck.stride))
        self.body = Sequential(*modules)

        initialize_weights(self.modules())


    def forward(self, x):
        
        # current code only supports one extra image
        # it comes with a extra dimension for number of extra image. We will just squeeze it out for now
        x = self.input_layer(x)

        for idx, module in enumerate(self.body):
            x = module(x)

        x = self.output_layer(x)
        norm = torch.norm(x, 2, 1, True)
        output = torch.div(x, norm)

        return output, norm



def IR_18(input_size):
    """ Constructs a ir-18 model.
    """
    model = Backbone(input_size, 18, 'ir')

    return model


def IR_34(input_size):
    """ Constructs a ir-34 model.
    """
    model = Backbone(input_size, 34, 'ir')

    return model


def IR_50(input_size):
    """ Constructs a ir-50 model.
    """
    model = Backbone(input_size, 50, 'ir')

    return model


def IR_101(input_size):
    """ Constructs a ir-101 model.
    """
    model = Backbone(input_size, 100, 'ir')

    return model


def IR_152(input_size):
    """ Constructs a ir-152 model.
    """
    model = Backbone(input_size, 152, 'ir')

    return model


def IR_200(input_size):
    """ Constructs a ir-200 model.
    """
    model = Backbone(input_size, 200, 'ir')

    return model


def IR_SE_50(input_size):
    """ Constructs a ir_se-50 model.
    """
    model = Backbone(input_size, 50, 'ir_se')

    return model


def IR_SE_101(input_size):
    """ Constructs a ir_se-101 model.
    """
    model = Backbone(input_size, 100, 'ir_se')

    return model


def IR_SE_152(input_size):
    """ Constructs a ir_se-152 model.
    """
    model = Backbone(input_size, 152, 'ir_se')

    return model


def IR_SE_200(input_size):
    """ Constructs a ir_se-200 model.
    """
 