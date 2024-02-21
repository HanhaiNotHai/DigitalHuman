import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(
        self, bert_n_emb, bert_n_head, attn_pdrop, resid_pdrop, block_size, sampler
    ):
        super().__init__()
        assert bert_n_emb % bert_n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(bert_n_emb, bert_n_emb)
        self.query = nn.Linear(bert_n_emb, bert_n_emb)
        self.value = nn.Linear(bert_n_emb, bert_n_emb)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(bert_n_emb, bert_n_emb)
        self.n_head = bert_n_head
        self.causal = True if sampler == 'autoregressive' else False
        if self.causal:
            mask = torch.tril(torch.ones(block_size, block_size))
            self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = (
            self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        q = (
            self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        v = (
            self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)

        present = torch.stack((k, v))
        if self.causal and layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if self.causal and layer_past is None:
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, present


class Block(nn.Module):
    """an unassuming Transformer block"""

    def __init__(
        self, bert_n_emb, resid_pdrop, bert_n_head, attn_pdrop, block_size, sampler
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(bert_n_emb)
        self.ln2 = nn.LayerNorm(bert_n_emb)
        self.attn = CausalSelfAttention(
            bert_n_emb, bert_n_head, attn_pdrop, resid_pdrop, block_size, sampler
        )
        self.mlp = nn.Sequential(
            nn.Linear(bert_n_emb, 4 * bert_n_emb),
            nn.GELU(),  # nice
            nn.Linear(4 * bert_n_emb, bert_n_emb),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x, layer_past=None, return_present=False):

        attn, present = self.attn(self.ln1(x), layer_past)
        x = x + attn
        x = x + self.mlp(self.ln2(x))

        if layer_past is not None or return_present:
            return x, present
        return x


class TransformerLanguage(nn.Module):
    """the full GPT language model, with a context size of block_size"""

    def __init__(
        self,
        codebook_size,
        bert_n_emb,
        bert_n_layers,
        bert_n_head,
        block_size,
        embd_pdrop,
        resid_pdrop,
        attn_pdrop,
        sampler='absorbing',
    ):
        super().__init__()

        self.vocab_size = codebook_size + 1
        self.n_embd = bert_n_emb
        self.block_size = block_size
        self.n_layers = bert_n_layers
        self.codebook_size = codebook_size
        self.causal = sampler == 'autoregressive'
        if self.causal:
            self.vocab_size = codebook_size

        self.text_feature_mapping = nn.Sequential(
            nn.LayerNorm(384),
            nn.Linear(384, 256),
            nn.LayerNorm(256),
            nn.Linear(256, self.n_embd),
            nn.LayerNorm(self.n_embd),
        )

        self.identity_tok_emb = nn.Embedding(self.vocab_size, self.n_embd)
        self.pose_tok_emb = nn.Embedding(self.vocab_size, self.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.block_size, self.n_embd))
        self.start_tok = nn.Parameter(torch.zeros(1, 1, self.n_embd))
        self.drop = nn.Dropout(embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(
            *[
                Block(
                    bert_n_emb,
                    resid_pdrop,
                    bert_n_head,
                    attn_pdrop,
                    block_size,
                    sampler,
                )
                for _ in range(self.n_layers)
            ]
        )
        # decoder head
        self.ln_f = nn.LayerNorm(self.n_embd)
        self.identity_head = nn.Linear(self.n_embd, self.codebook_size, bias=False)
        self.pose_head = nn.Linear(self.n_embd, self.codebook_size, bias=False)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, text_embedding, identity_idx, pose_idx, t=None):
        # each index maps to a (learnable) vector
        identity_token_embeddings = self.identity_tok_emb(identity_idx)
        pose_token_embeddings = self.pose_tok_emb(pose_idx)

        token_embeddings = torch.cat(
            (identity_token_embeddings, pose_token_embeddings), dim=1
        )

        if self.causal:
            token_embeddings = torch.cat(
                (
                    self.start_tok.repeat(token_embeddings.size(0), 1, 1),
                    token_embeddings,
                ),
                dim=1,
            )

        t = token_embeddings.shape[1]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        # each position maps to a (learnable) vector

        position_embeddings = self.pos_emb[:, :t, :]

        x = token_embeddings + position_embeddings
        x = self.drop(x)
        x = torch.cat([self.text_feature_mapping(text_embedding), x], dim=1)

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        identity_logits = self.identity_head(x)
        pose_logits = self.pose_head(x)

        return identity_logits, pose_logits
