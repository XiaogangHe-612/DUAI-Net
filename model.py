import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns


# =========================================================
# Loss
# =========================================================
class MaskedNLLLoss(nn.Module):
    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight, reduction='sum')

    def forward(self, pred, target, mask):
        mask_ = mask.view(-1, 1)
        if self.weight is None:
            loss = self.loss(pred * mask_, target) / torch.sum(mask)
        else:
            loss = self.loss(pred * mask_, target) / torch.sum(self.weight[target] * mask_.squeeze())
        return loss


# =========================================================
# Helpers
# =========================================================
def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True):
    if torch.cuda.is_available():
        try:
            from apex.normalization import FusedLayerNorm  # type: ignore
            return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
        except ImportError:
            pass
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


def logits_to_confidence(logits: torch.Tensor) -> torch.Tensor:
    """
    logits: (B, N, C)
    return: confidence (B, N)
    """
    prob = F.softmax(logits, dim=-1)
    entropy = -(prob * torch.log(prob.clamp_min(1e-8))).sum(dim=-1)
    if logits.size(-1) > 1:
        entropy = entropy / math.log(logits.size(-1))
    confidence = torch.exp(-entropy)
    return confidence


def build_source_conf_edge_scale(
    confidence: torch.Tensor,
    valid_mask: torch.Tensor = None,
    gamma: float = 0.45,
    alpha: float = 4.0,
    beta: float = 0.55
) -> torch.Tensor:
    """
    source-only edge scale:
        s_{i<-j} = gamma + (1-gamma) * sigmoid(alpha * (c_j - beta))

    confidence: (B, N)
    valid_mask: (B, N)
    return    : (B, N, N)
    """
    B, N = confidence.size()
    src_conf = confidence.unsqueeze(1)  # (B,1,N)
    scale = gamma + (1.0 - gamma) * torch.sigmoid(alpha * (src_conf - beta))  # (B,1,N)
    scale = scale.expand(-1, N, -1).contiguous()  # (B,N,N)

    if valid_mask is not None:
        valid = valid_mask.bool()
        pair_valid = valid.unsqueeze(1) & valid.unsqueeze(2)
        scale = torch.where(pair_valid, scale, torch.ones_like(scale))

    return scale


# =========================================================
# Position-wise FFN
# =========================================================
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.actv = gelu
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(self.actv(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x


# =========================================================
# Dynamic Composition Transformer (DCTransformer)
# =========================================================
def rmsnorm_no_scale(x: torch.Tensor, dim: int, eps: float = 1e-8) -> torch.Tensor:
    return x / torch.sqrt(x.pow(2).mean(dim=dim, keepdim=True) + eps)


def qk_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


class DynamicHeadComposition(nn.Module):
    """
    Compose(A, Q, K; θ)
    A: (B,H,T,S)
    Q: (B,T,Dm)
    K: (B,S,Dm)
    Returns: same shape as A
    """
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        rank: int = 1,
        eps: float = 1e-8,
        use_base_proj: bool = True,
        use_query_branch: bool = True,
        use_key_branch: bool = False,
        paper_init: bool = True
    ):
        super().__init__()
        self.Dm = model_dim
        self.H = num_heads
        self.R = rank
        self.eps = eps

        self.use_base_proj = use_base_proj
        self.use_query_branch = use_query_branch
        self.use_key_branch = use_key_branch

        I = 2 * num_heads * rank

        if use_base_proj:
            self.Wb = nn.Parameter(torch.empty(num_heads, num_heads))

        self.Wq1 = nn.Linear(model_dim, I, bias=False)
        self.Wq2 = nn.Linear(I, I, bias=False)
        self.Wqg = nn.Linear(model_dim, num_heads, bias=False)

        self.Wk1 = nn.Linear(model_dim, I, bias=False)
        self.Wk2 = nn.Linear(I, I, bias=False)
        self.Wkg = nn.Linear(model_dim, num_heads, bias=False)

        if paper_init:
            if use_base_proj:
                nn.init.xavier_normal_(self.Wb)

            nn.init.xavier_normal_(self.Wq1.weight)
            nn.init.xavier_normal_(self.Wk1.weight)

            std_proj = 0.02 / math.sqrt(2 * self.H * self.R * (self.H + self.R))
            std_gate = 0.05 * math.sqrt(2.0 / (self.Dm + self.H))

            nn.init.normal_(self.Wq2.weight, mean=0.0, std=std_proj)
            nn.init.normal_(self.Wk2.weight, mean=0.0, std=std_proj)
            nn.init.normal_(self.Wqg.weight, mean=0.0, std=std_gate)
            nn.init.normal_(self.Wkg.weight, mean=0.0, std=std_gate)
        else:
            if use_base_proj:
                nn.init.xavier_uniform_(self.Wb)
            nn.init.xavier_uniform_(self.Wq1.weight)
            nn.init.xavier_uniform_(self.Wk1.weight)
            nn.init.normal_(self.Wq2.weight, std=0.02)
            nn.init.normal_(self.Wk2.weight, std=0.02)
            nn.init.normal_(self.Wqg.weight, std=0.02)
            nn.init.normal_(self.Wkg.weight, std=0.02)

    def _dyn_from_q(self, Q: torch.Tensor):
        B, T, _ = Q.shape
        h = F.gelu(self.Wq1(Q))
        h = self.Wq2(h)
        q1, q2 = torch.chunk(h, 2, dim=-1)
        q1 = q1.reshape(B, T, self.H, self.R)
        q2 = q2.reshape(B, T, self.R, self.H)

        q1 = rmsnorm_no_scale(q1, dim=2, eps=self.eps)
        qg = torch.tanh(self.Wqg(Q))
        return q1, q2, qg

    def _dyn_from_k(self, K: torch.Tensor):
        B, S, _ = K.shape
        h = F.gelu(self.Wk1(K))
        h = self.Wk2(h)
        k1, k2 = torch.chunk(h, 2, dim=-1)
        k1 = k1.reshape(B, S, self.H, self.R)
        k2 = k2.reshape(B, S, self.R, self.H)
        k1 = rmsnorm_no_scale(k1, dim=2, eps=self.eps)
        kg = torch.tanh(self.Wkg(K))
        return k1, k2, kg

    def forward(self, A: torch.Tensor, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        B, H, T, S = A.shape
        assert H == self.H, f"num_heads mismatch: got {H}, expect {self.H}"

        A_btsH = A.permute(0, 2, 3, 1).contiguous()

        if self.use_base_proj:
            base = A_btsH @ self.Wb
        else:
            base = A_btsH

        out = base

        if self.use_query_branch:
            wq1, wq2, wqg = self._dyn_from_q(Q)
            q_low = torch.einsum("btsh,bthr->btsr", A_btsH, wq1)
            q_high = torch.einsum("btsr,btrh->btsh", q_low, wq2)
            q_gate = A_btsH * wqg.unsqueeze(2)
            out = out + q_high + q_gate

        if self.use_key_branch:
            wk1, wk2, wkg = self._dyn_from_k(K)
            k_low = torch.einsum("btsh,bshr->btsr", A_btsH, wk1)
            k_high = torch.einsum("btsr,bsrh->btsh", k_low, wk2)
            k_gate = A_btsH * wkg.unsqueeze(1)
            out = out + k_high + k_gate

        return out.permute(0, 3, 1, 2).contiguous()


class DynamicCompositionAttention(nn.Module):
    def __init__(
        self,
        head_count: int,
        model_dim: int,
        dropout: float = 0.1,
        rank: int = 1,
        use_pre: bool = False,
        use_post: bool = True,
        use_base_proj: bool = True,
        use_query_branch: bool = True,
        use_key_branch: bool = False,
        use_qk_norm: bool = True,
        paper_init: bool = True
    ):
        super().__init__()
        assert model_dim % head_count == 0
        self.head_count = head_count
        self.model_dim = model_dim
        self.dim_per_head = model_dim // head_count

        self.linear_k = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_v = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_q = nn.Linear(model_dim, head_count * self.dim_per_head)

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(model_dim, model_dim)

        self.use_pre = use_pre
        self.use_post = use_post
        self.use_qk_norm = use_qk_norm

        if use_pre:
            self.compose_pre = DynamicHeadComposition(
                model_dim, head_count, rank=rank,
                use_base_proj=use_base_proj,
                use_query_branch=use_query_branch,
                use_key_branch=use_key_branch,
                paper_init=paper_init
            )
            self.alpha_pre = nn.Parameter(torch.tensor(0.0))

        if use_post:
            self.compose_post = DynamicHeadComposition(
                model_dim, head_count, rank=rank,
                use_base_proj=use_base_proj,
                use_query_branch=use_query_branch,
                use_key_branch=use_key_branch,
                paper_init=paper_init
            )
            self.alpha_post = nn.Parameter(torch.tensor(0.0))

    def forward(self, key, value, query, mask=None):
        B = query.size(0)
        T = query.size(1)
        S = key.size(1)
        H = self.head_count
        Dh = self.dim_per_head

        k = self.linear_k(key).reshape(B, S, H, Dh).transpose(1, 2).contiguous()
        v = self.linear_v(value).reshape(B, S, H, Dh).transpose(1, 2).contiguous()
        q = self.linear_q(query).reshape(B, T, H, Dh).transpose(1, 2).contiguous()

        if self.use_qk_norm:
            q = qk_norm(q)
            k = qk_norm(k)

        q = q / math.sqrt(Dh)
        scores = torch.matmul(q, k.transpose(-2, -1))

        if self.use_pre:
            s2 = self.compose_pre(scores, query, key)
            g = torch.tanh(self.alpha_pre)
            scores = scores + g * (s2 - scores)

        if mask is not None:
            m = mask
            if m.dim() == 2:
                m = m.unsqueeze(1).unsqueeze(1)
            elif m.dim() == 3 and m.size(1) == 1:
                m = m.unsqueeze(2)
            elif m.dim() == 3:
                m = m.unsqueeze(1)
            m = m.expand_as(scores)
            scores = scores.masked_fill(m, -1e10)

        attn = torch.softmax(scores, dim=-1)

        if self.use_post:
            a2 = self.compose_post(attn, query, key)
            g = torch.tanh(self.alpha_post)
            attn = attn + g * (a2 - attn)

        attn_to_return = attn
        attn_drop = self.dropout(attn)
        context = torch.matmul(attn_drop, v)

        context = context.transpose(1, 2).contiguous().reshape(B, T, H * Dh)
        output = self.linear(context)
        return output, attn_to_return


# =========================================================
# Standard Multi-Head Attention (for ablation)
# =========================================================
class MultiHeadedAttention(nn.Module):
    def __init__(self, head_count, model_dim, dropout=0.1):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

        self.linear_k = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_v = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_q = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(model_dim, model_dim)

    def forward(self, key, value, query, mask=None):
        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count

        key = self.linear_k(key).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)
        value = self.linear_v(value).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)
        query = self.linear_q(query).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)

        query = query / math.sqrt(dim_per_head)
        scores = torch.matmul(query, key.transpose(2, 3))

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask, -1e10)

        attn = self.softmax(scores)
        drop_attn = self.dropout(attn)
        context = torch.matmul(drop_attn, value).transpose(1, 2).contiguous() \
            .view(batch_size, -1, head_count * dim_per_head)
        output = self.linear(context)
        return output, attn


# =========================================================
# Positional Encoding
# =========================================================
class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, speaker_emb):
        L = x.size(1)
        pos_emb = self.pe[:, :L]
        x = x + pos_emb + speaker_emb
        return x


# =========================================================
# Transformer Encoder Layer
# =========================================================
class DCTransformerLayer(nn.Module):
    def __init__(
        self,
        d_model,
        heads,
        d_ff,
        dropout,
        dc_rank=2,
        use_standard_mha=False,
        dc_use_qk_norm=True,
        dc_use_pre=False,
        dc_use_key_branch=False
    ):
        super(DCTransformerLayer, self).__init__()

        # Optional ablation switch: use standard multi-head attention instead of DCTransformer attention.
        if use_standard_mha:
            self.self_attn = MultiHeadedAttention(
                head_count=heads,
                model_dim=d_model,
                dropout=dropout
            )
        else:
            self.self_attn = DynamicCompositionAttention(
                head_count=heads,
                model_dim=d_model,
                dropout=dropout,
                rank=dc_rank,
                use_pre=dc_use_pre,
                use_post=True,
                use_base_proj=True,
                use_query_branch=True,
                use_key_branch=dc_use_key_branch,
                use_qk_norm=dc_use_qk_norm,
                paper_init=True
            )

        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, inputs_a, inputs_b, mask):
        if inputs_a.equal(inputs_b):
            if iter != 0:
                inputs_b = self.layer_norm(inputs_b)

            mask = mask.unsqueeze(1)
            context, atten_score = self.self_attn(inputs_b, inputs_b, inputs_b, mask=mask)
        else:
            if iter != 0:
                inputs_b = self.layer_norm(inputs_b)

            mask = mask.unsqueeze(1)
            context, atten_score = self.self_attn(inputs_a, inputs_a, inputs_b, mask=mask)

        out = self.dropout(context) + inputs_b
        return self.feed_forward(out), atten_score


# =========================================================
# Transformer Encoder
# =========================================================
class DCTransformer(nn.Module):
    def __init__(
        self,
        d_model,
        d_ff,
        heads,
        layers,
        dropout=0.1,
        dc_rank=2,
        use_standard_mha=False,
        dc_use_qk_norm=True,
        dc_use_pre=False,
        dc_use_key_branch=False
    ):
        super(DCTransformer, self).__init__()
        self.d_model = d_model
        self.layers = layers
        self.pos_emb = PositionalEncoding(d_model)
        self.transformer_inter = nn.ModuleList([
            DCTransformerLayer(
                d_model,
                heads,
                d_ff,
                dropout,
                dc_rank=dc_rank,
                use_standard_mha=use_standard_mha,
                dc_use_qk_norm=dc_use_qk_norm,
                dc_use_pre=dc_use_pre,
                dc_use_key_branch=dc_use_key_branch
            )
            for _ in range(layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_b, mask, speaker_emb=None):
        if speaker_emb is not None:
            x_b = self.pos_emb(x_b, speaker_emb)
            x_b = self.dropout(x_b)

        atten_score = None
        for i in range(self.layers):
            x_b, atten_score = self.transformer_inter[i](i, x_b, x_b, mask.eq(0))
        return x_b, atten_score


# =========================================================
# Gating Modules
# =========================================================
class Unimodal_GatedFusion(nn.Module):
    def __init__(self, hidden_size, dataset):
        super(Unimodal_GatedFusion, self).__init__()
        self.fc = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, a):
        z = torch.sigmoid(self.fc(a))
        final_rep = z * a
        return final_rep


class EnhancedFilterModule(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        gate = self.gate(x)
        out = gate * x
        return out


# =========================================================
# DARF: Uncertainty-triggered Text-guided Retrieval Fusion
# =========================================================
class DynamicAuxiliaryResidualFusion(nn.Module):
    """
    Dynamic Auxiliary Residual Fusion (DARF).

    fusion_mode:
        'a'  : T <- A
        'v'  : T <- V
        'av' : T <- A and T <- V

    The textual relational representation serves as the query, while
    text uncertainty controls the contribution of auxiliary modalities.
    """
    def __init__(
        self,
        hidden_dim,
        n_head,
        dropout,
        rank=1,
        fuse_coeff=0.1,
        visual_fuse_coeff=0.01,
        fusion_mode='a'
    ):
        super().__init__()
        assert fusion_mode in ['a', 'v', 'av'], "fusion_mode must be one of ['a', 'v', 'av']"
        self.fusion_mode = fusion_mode
        self.fuse_coeff = fuse_coeff
        self.visual_fuse_coeff = visual_fuse_coeff

        if fusion_mode in ['a', 'av']:
            self.t_a_attn = DynamicCompositionAttention(
                head_count=n_head,
                model_dim=hidden_dim,
                dropout=dropout,
                rank=rank,
                use_pre=False,
                use_post=True,
                use_base_proj=True,
                use_query_branch=True,
                use_key_branch=False,
                use_qk_norm=True,
                paper_init=True
            )
            self.audio_gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )

        if fusion_mode in ['v', 'av']:
            self.t_v_attn = DynamicCompositionAttention(
                head_count=n_head,
                model_dim=hidden_dim,
                dropout=dropout,
                rank=rank,
                use_pre=False,
                use_post=True,
                use_base_proj=True,
                use_query_branch=True,
                use_key_branch=False,
                use_qk_norm=True,
                paper_init=True
            )
            self.visual_gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, t_feat, a_feat, v_feat, text_uncertainty, mask):
        """
        t_feat: (B, N, H)
        a_feat: (B, N, H)
        v_feat: (B, N, H)
        text_uncertainty: (B, N) or (B, N, 1)
        mask: (B, N), 1-valid / 0-pad
        """
        attn_mask = mask.eq(0)

        if text_uncertainty.dim() == 2:
            text_uncertainty = text_uncertainty.unsqueeze(-1)

        fused_delta = torch.zeros_like(t_feat)
        g_a = torch.zeros(t_feat.size(0), t_feat.size(1), 1, device=t_feat.device)
        g_v = torch.zeros(t_feat.size(0), t_feat.size(1), 1, device=t_feat.device)
        attn_ta, attn_tv = None, None

        if self.fusion_mode in ['a', 'av']:
            delta_ta, attn_ta = self.t_a_attn(a_feat, a_feat, t_feat, mask=attn_mask)
            g_a = torch.sigmoid(self.audio_gate(torch.cat([t_feat, delta_ta], dim=-1)))
            fused_delta = fused_delta + self.fuse_coeff * g_a * delta_ta

        if self.fusion_mode in ['v', 'av']:
            delta_tv, attn_tv = self.t_v_attn(v_feat, v_feat, t_feat, mask=attn_mask)
            g_v = torch.sigmoid(self.visual_gate(torch.cat([t_feat, delta_tv], dim=-1)))
            fused_delta = fused_delta + self.visual_fuse_coeff * g_v * delta_tv

        t_fused = self.layer_norm(t_feat + text_uncertainty * fused_delta)
        return t_fused, g_a, g_v, attn_ta, attn_tv


# =========================================================
# Dynamic Relational Propagation (DRP)
# =========================================================
class RelationalGraphAttentionLayer(nn.Module):
    """
    Dense relation-aware GAT
    The confidence-derived edge scale is injected as an attention-logit bias.
    """
    def __init__(
        self,
        in_features,
        out_features,
        dropout,
        alpha,
        concat=True,
        relation=True,
        relation_embedding=None,
        relation_dim=10,
        edge_logit_scale=0.5
    ):
        super(RelationalGraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.relation = relation
        self.relation_dim = relation_dim
        self.edge_logit_scale = edge_logit_scale

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        if self.relation:
            assert relation_embedding is not None, "relation_embedding is required when relation=True"
            self.relation_embedding = relation_embedding
            self.a = nn.Parameter(torch.empty(size=(2 * out_features + relation_dim, 1)))
        else:
            self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))

        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.layer_norm = LayerNorm(out_features)

    def forward(self, h, adj, edge_scale=None):
        Wh = torch.matmul(h, self.W)
        a_input = self._prepare_attentional_mechanism_input(Wh)

        if self.relation:
            long_adj = adj.long().to(h.device)
            relation_feat = self.relation_embedding(long_adj)
            a_input = torch.cat([a_input, relation_feat], dim=-1)

        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))

        if edge_scale is not None:
            e = e + self.edge_logit_scale * torch.log(edge_scale.clamp_min(1e-6))

        if self.relation:
            edge_mask = adj > 0
            zero_vec = torch.full_like(e, -9e15)
            attention = torch.where(edge_mask, e, zero_vec)
            attention = F.softmax(attention, dim=2)
        else:
            attention = F.softmax(e, dim=2)

        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        h_prime = self.layer_norm(h_prime)

        if self.concat:
            return F.gelu(h_prime), attention
        else:
            return h_prime, attention

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size(1)
        B = Wh.size(0)
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=1)
        Wh_repeated_alternating = Wh.repeat(1, N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=2)
        return all_combinations_matrix.view(B, N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + f' ({self.in_features} -> {self.out_features})'


class RelationalGraphAttention(nn.Module):
    def __init__(
        self,
        nfeat,
        nhid,
        dropout=0.2,
        alpha=0.2,
        nheads=2,
        relation_embedding=None,
        relation_dim=10,
        edge_logit_scale=0.5
    ):
        super(RelationalGraphAttention, self).__init__()
        self.dropout = dropout

        self.attentions = nn.ModuleList([
            RelationalGraphAttentionLayer(
                nfeat,
                nhid,
                dropout=dropout,
                alpha=alpha,
                concat=True,
                relation=True,
                relation_embedding=relation_embedding,
                relation_dim=relation_dim,
                edge_logit_scale=edge_logit_scale
            )
            for _ in range(nheads)
        ])

        self.out_att = RelationalGraphAttentionLayer(
            nhid * nheads,
            nhid,
            dropout=dropout,
            alpha=alpha,
            concat=True,
            relation=True,
            relation_embedding=relation_embedding,
            relation_dim=relation_dim,
            edge_logit_scale=edge_logit_scale
        )

        self.fc = nn.Linear(nhid, nhid)
        self.layer_norm = LayerNorm(nhid)

    def forward(self, x, adj, edge_scale=None):
        residual = x
        x = F.dropout(x, self.dropout, training=self.training)

        attended_outputs = []
        attention_weights = []
        for att_module in self.attentions:
            att_out, att_w = att_module(x, adj, edge_scale=edge_scale)
            attended_outputs.append(att_out)
            attention_weights.append(att_w)

        x = torch.cat(attended_outputs, dim=-1)
        x = F.dropout(x, self.dropout, training=self.training)

        att_out, att_w = self.out_att(x, adj, edge_scale=edge_scale)
        attention_weights.append(att_w)

        x = F.gelu(att_out)
        x = self.fc(x)
        x = x + residual
        x = self.layer_norm(x)
        return x, attention_weights


def Graphplt(Attention):
    Attention = Attention[-1]
    attention = Attention.cpu().detach().numpy()
    num = len(attention)
    n = math.ceil(math.sqrt(num))
    m = math.ceil(num / n)
    fig = plt.figure(figsize=(20 * n, 20 * m), dpi=75)
    for i in range(num):
        axs = fig.add_subplot(n, m, i + 1)
        sns.heatmap(attention[i], cmap='coolwarm', annot=True, fmt='.2f', ax=axs)
    plt.tight_layout()
    plt.show()


# =========================================================
# Dynamic Uncertainty-Aware Interaction Network (DUAI-Net)
# =========================================================
class DUAINet(nn.Module):
    def __init__(
        self,
        dataset,
        D_text,
        D_visual,
        D_audio,
        n_head,
        n_classes,
        hidden_dim,
        n_speakers,
        dropout,
        use_drp_uncertainty=True,
        uncertainty_detach=True,
        edge_scale_gamma=0.45,
        edge_scale_alpha=4.0,
        edge_scale_beta=0.55,
        edge_logit_scale=0.5,
        relation_num=6,
        relation_dim=10,
        use_darf=True,
        fusion_rank=1,
        fusion_coeff=0.1,
        visual_fusion_coeff=0.01,
        fusion_mode='a',
        dc_rank=2,
        use_standard_mha=False,
        dc_use_qk_norm=True,
        dc_use_pre=False,
        dc_use_key_branch=False
    ):
        super(DUAINet, self).__init__()
        self.n_classes = n_classes
        self.n_speakers = n_speakers
        self.dataset = dataset

        self.use_drp_uncertainty = use_drp_uncertainty
        self.uncertainty_detach = uncertainty_detach
        self.edge_scale_gamma = edge_scale_gamma
        self.edge_scale_alpha = edge_scale_alpha
        self.edge_scale_beta = edge_scale_beta
        self.use_darf = use_darf

        padding_idx = n_speakers
        self.speaker_embeddings = nn.Embedding(n_speakers + 1, hidden_dim, padding_idx=padding_idx)

        self.relation_embedding = nn.Embedding(relation_num, relation_dim)

        self.text_projection = nn.Linear(D_text, hidden_dim)
        self.audio_projection = nn.Linear(D_audio, hidden_dim)
        self.visual_projection = nn.Linear(D_visual, hidden_dim)

        # Audio and visual DCTransformer branches
        self.audio_dc_transformer = DCTransformer(
            d_model=hidden_dim,
            d_ff=hidden_dim,
            heads=n_head,
            layers=1,
            dropout=dropout,
            dc_rank=dc_rank,
            use_standard_mha=use_standard_mha,
            dc_use_qk_norm=dc_use_qk_norm,
            dc_use_pre=dc_use_pre,
            dc_use_key_branch=dc_use_key_branch
        )
        self.visual_dc_transformer = DCTransformer(
            d_model=hidden_dim,
            d_ff=hidden_dim,
            heads=n_head,
            layers=1,
            dropout=dropout,
            dc_rank=dc_rank,
            use_standard_mha=use_standard_mha,
            dc_use_qk_norm=dc_use_qk_norm,
            dc_use_pre=dc_use_pre,
            dc_use_key_branch=dc_use_key_branch
        )
        self.audio_filter = EnhancedFilterModule(hidden_dim)
        self.visual_filter = EnhancedFilterModule(hidden_dim)

        # DRP text branch
        self.drp_inter_propagation = RelationalGraphAttention(
            hidden_dim, hidden_dim,
            dropout=dropout,
            relation_embedding=self.relation_embedding,
            relation_dim=relation_dim,
            edge_logit_scale=edge_logit_scale
        )
        self.drp_intra_propagation = RelationalGraphAttention(
            hidden_dim, hidden_dim,
            dropout=dropout,
            relation_embedding=self.relation_embedding,
            relation_dim=relation_dim,
            edge_logit_scale=edge_logit_scale
        )

        # Dynamic Auxiliary Residual Fusion (DARF)
        self.darf = DynamicAuxiliaryResidualFusion(
            hidden_dim=hidden_dim,
            n_head=n_head,
            dropout=dropout,
            rank=fusion_rank,
            fuse_coeff=fusion_coeff,
            visual_fuse_coeff=visual_fusion_coeff,
            fusion_mode=fusion_mode
        )

        # DRP stage-1 auxiliary classifier
        self.drp_aux_classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
        )

        # Modality-specific classifiers
        self.text_classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
        )
        self.audio_classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
        )
        self.visual_classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
        )

        # Cached statistics for analysis
        self.last_text_confidence = None
        self.last_text_edge_scale = None
        self.last_stage1_aux_logits = None
        self.last_text_fusion_uncertainty = None

        self.last_audio_gate = None
        self.last_visual_gate = None
        self.last_audio_fusion_coeff = None
        self.last_visual_fusion_coeff = None
        self.last_attn_ta = None
        self.last_attn_tv = None

    def set_drp_uncertainty(self, enabled: bool):
        self.use_drp_uncertainty = enabled

    def set_darf(self, enabled: bool):
        self.use_darf = enabled

    def _build_drp_edge_scale(self, stage1_aux_logits, u_mask):
        unc_logits = stage1_aux_logits.detach() if self.uncertainty_detach else stage1_aux_logits
        confidence = logits_to_confidence(unc_logits)
        edge_scale = build_source_conf_edge_scale(
            confidence=confidence,
            valid_mask=u_mask,
            gamma=self.edge_scale_gamma,
            alpha=self.edge_scale_alpha,
            beta=self.edge_scale_beta
        )

        self.last_text_confidence = confidence.detach()
        self.last_text_edge_scale = edge_scale.detach()
        self.last_stage1_aux_logits = stage1_aux_logits.detach()

        return edge_scale

    def _build_text_uncertainty(self, stage1_aux_logits, u_mask):
        unc_logits = stage1_aux_logits.detach() if self.uncertainty_detach else stage1_aux_logits
        confidence = logits_to_confidence(unc_logits)          # (B, N)
        uncertainty = 1.0 - confidence                         # (B, N)
        if u_mask is not None:
            uncertainty = uncertainty * u_mask.float()

        self.last_text_fusion_uncertainty = uncertainty.detach()
        return uncertainty

    def forward(
        self,
        textf,
        visuf,
        acouf,
        u_mask,
        qmask,
        dia_len,
        Self_semantic_adj,
        Cross_semantic_adj,
        Semantic_adj
    ):
        del Semantic_adj

        spk_idx = torch.argmax(qmask, dim=-1)

        pad_id = self.n_speakers
        for i, x in enumerate(dia_len):
            x = int(x)
            if x < spk_idx.size(1):
                spk_idx[i, x:] = pad_id

        spk_embeddings = self.speaker_embeddings(spk_idx)

        # =================================================
        # Text branch: Dynamic Relational Propagation (DRP)
        # =================================================
        textf = self.text_projection(textf.permute(1, 0, 2))  # (B, N, H)

        text_stage1, _ = self.drp_inter_propagation(textf, Cross_semantic_adj, edge_scale=None)

        # Preliminary emotion prediction for uncertainty estimation
        stage1_aux_logits = self.drp_aux_classifier(text_stage1)

        text_edge_scale = None
        if self.use_drp_uncertainty:
            text_edge_scale = self._build_drp_edge_scale(stage1_aux_logits, u_mask)

        text_relational, _ = self.drp_intra_propagation(text_stage1, Self_semantic_adj, edge_scale=text_edge_scale)

        sub_log_prog = []

        # =================================================
        # Audio / Visual branches: Dynamic Composition Transformer (DCTransformer)
        # =================================================
        if visuf is not None and acouf is not None:
            acouf = self.audio_projection(acouf.permute(1, 0, 2))  # (B, N, H)
            visuf = self.visual_projection(visuf.permute(1, 0, 2))  # (B, N, H)

            audio_context, _ = self.audio_dc_transformer(acouf, u_mask, spk_embeddings)
            visual_context, _ = self.visual_dc_transformer(visuf, u_mask, spk_embeddings)

            # Dynamic Auxiliary Residual Fusion (DARF)
            if self.use_darf:
                text_uncertainty = self._build_text_uncertainty(stage1_aux_logits, u_mask)
                text_fused, g_a, g_v, attn_ta, attn_tv = self.darf(
                    text_relational, audio_context, visual_context, text_uncertainty, u_mask
                )
                self.last_audio_gate = g_a.detach()
                self.last_visual_gate = g_v.detach()
                self.last_audio_fusion_coeff = torch.tensor(
                    self.darf.fuse_coeff,
                    device=text_relational.device
                )
                self.last_visual_fusion_coeff = torch.tensor(
                    self.darf.visual_fuse_coeff,
                    device=text_relational.device
                )
                self.last_attn_ta = attn_ta.detach() if attn_ta is not None else None
                self.last_attn_tv = attn_tv.detach() if attn_tv is not None else None
            else:
                text_fused = text_relational
                self.last_audio_gate = None
                self.last_visual_gate = None
                self.last_audio_fusion_coeff = None
                self.last_visual_fusion_coeff = None
                self.last_attn_ta = None
                self.last_attn_tv = None
                self.last_text_fusion_uncertainty = None

            # Modality-specific audio/visual representations
            acouf = self.audio_filter(audio_context)
            visuf = self.visual_filter(visual_context)

            t = self.text_classifier(text_fused)
            a = self.audio_classifier(acouf)
            v = self.visual_classifier(visuf)

            all_final_out = t + a + v

            sub_log_prog.append(F.log_softmax(t, dim=-1))
            sub_log_prog.append(F.log_softmax(a, dim=-1))
            sub_log_prog.append(F.log_softmax(v, dim=-1))
        else:
            t = self.text_classifier(text_relational)
            all_final_out = t

            self.last_audio_gate = None
            self.last_visual_gate = None
            self.last_audio_fusion_coeff = None
            self.last_visual_fusion_coeff = None
            self.last_attn_ta = None
            self.last_attn_tv = None
            self.last_text_fusion_uncertainty = None

            sub_log_prog.append(F.log_softmax(t, dim=-1))

        all_log_prob = F.log_softmax(all_final_out, dim=-1)
        all_prob = F.softmax(all_final_out, dim=-1)
        stage1_aux_log_prob = F.log_softmax(stage1_aux_logits, dim=-1)

        return sub_log_prog, all_log_prob, all_prob, all_final_out, stage1_aux_log_prob