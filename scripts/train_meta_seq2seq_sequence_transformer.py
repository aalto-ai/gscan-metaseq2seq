import argparse
import os
import numpy as np
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset
import sys
from torch.utils.data import DataLoader, Subset
from positional_encodings.torch_encodings import PositionalEncoding1D
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from gscan_metaseq2seq.models.embedding import BOWEmbedding
from gscan_metaseq2seq.util.dataset import (
    PaddingDataset,
    ReshuffleOnIndexZeroDataset,
    MapDataset,
)
from gscan_metaseq2seq.util.load_data import load_data, load_data_directories
from gscan_metaseq2seq.util.logging import LoadableCSVLogger
from gscan_metaseq2seq.util.scheduler import transformer_optimizer_config
from gscan_metaseq2seq.util.padding import pad_to


class Attn(nn.Module):
    # batch attention module

    def __init__(self):
        super(Attn, self).__init__()

    def forward(self, Q, K, V, K_padding_mask=None):
        #
        # Input
        #  Q : Matrix of queries; ... x batch_size x n_queries x query_dim
        #  K : Matrix of keys; ... x batch_size x n_memory x query_dim
        #  V : Matrix of values; ... x batch_size x n_memory x value_dim
        #
        # Output
        #  R : soft-retrieval of values; ... x batch_size x n_queries x value_dim
        #  attn_weights : soft-retrieval of values; ... x batch_size x n_queries x n_memory
        orig_q_shape = Q.shape

        query_dim = torch.tensor(float(Q.size(-1)))
        if Q.is_cuda:
            query_dim = query_dim.cuda()

        Q = Q.flatten(0, -3)
        K_T = K.flatten(0, -3).transpose(-2, -1)
        V = V.flatten(0, -3)

        attn_weights = torch.bmm(
            Q.flatten(0, -3), K_T
        )  # ... x batch_size x n_queries x n_memory
        attn_weights = torch.div(attn_weights, torch.sqrt(query_dim))

        if K_padding_mask is not None:
            attn_weights.masked_fill_(
                K_padding_mask.flatten(0, -2)[..., None, :], float("-inf")
            )

        attn_weights = F.softmax(
            attn_weights, dim=-1
        )  # ... x batch_size x n_queries x n_memory
        R = torch.bmm(attn_weights, V)  # ... x batch_size x n_queries x value_dim

        # Hack to ensure that where all elements are padded, which
        # result in attn_weights being nan, we discard those and just use
        # the existing queries
        if K_padding_mask is not None:
            all_padded_mask = K_padding_mask.flatten(0, -2).all(dim=-1)[:, None, None]
            R = R.masked_fill(all_padded_mask, 0) + Q.masked_fill(~all_padded_mask, 0)

        return R.view(orig_q_shape), attn_weights


class EncoderTransformer(nn.Module):
    def __init__(self, embedding_dim, nlayers, nhead, norm_first, dropout_p):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.encoding = nn.Parameter(torch.randn(embedding_dim))
        self.bi = False
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=nhead,
                dim_feedforward=embedding_dim * 4,
                dropout=dropout_p,
                norm_first=norm_first,
            ),
            num_layers=nlayers,
        )

    def forward(self, x, key_padding_mask=None):
        key_padding_mask = (
            torch.zeros_like(x[..., 0]).bool()
            if key_padding_mask is None
            else key_padding_mask
        )

        z_embed_seq = torch.cat(
            [
                x,
                self.encoding[None, None].expand(
                    x.shape[0], 1, self.encoding.shape[-1]
                ),
            ],
            dim=-2,
        )
        z_embed_seq = self.dropout(z_embed_seq)
        padding_bits = torch.cat(
            [key_padding_mask, torch.zeros_like(key_padding_mask[:, :1])], dim=-1
        )

        encoded_seq = self.encoder(
            z_embed_seq.transpose(1, 0),
            src_key_padding_mask=padding_bits,
        )

        return encoded_seq[-1], encoded_seq[:-1]


class StateEncoderTransformer(nn.Module):
    def __init__(
        self,
        n_state_components,
        input_size,
        embedding_dim,
        nlayers,
        nhead,
        norm_first,
        dropout_p,
        pad_word_idx,
    ):
        super().__init__()
        self.n_state_components = n_state_components
        self.embedding_dim = embedding_dim
        self.state_embedding = BOWEmbedding(
            64, self.n_state_components, self.embedding_dim
        )
        self.state_projection = nn.Linear(
            self.n_state_components * self.embedding_dim, self.embedding_dim
        )
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.embedding_projection = nn.Linear(embedding_dim * 2, embedding_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.encoding = nn.Parameter(torch.randn(embedding_dim))
        self.pos_encoding = PositionalEncoding1D(embedding_dim)
        self.bi = False
        self.pad_word_idx = pad_word_idx
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=nhead,
                dim_feedforward=embedding_dim * 4,
                dropout=dropout_p,
                norm_first=norm_first,
            ),
            num_layers=nlayers,
        )

    def forward(self, state_padded, z_padded):
        state_padding_bits = torch.zeros_like(state_padded[..., 0]).bool()
        z_padding_bits = z_padded == self.pad_word_idx

        state_embed_seq = self.state_projection(self.state_embedding(state_padded))

        z_embed_seq = self.embedding(z_padded)
        z_embed_seq = torch.cat([self.pos_encoding(z_embed_seq), z_embed_seq], dim=-1)
        z_embed_seq = self.embedding_projection(z_embed_seq)
        z_embed_seq = torch.cat(
            [
                state_embed_seq,
                z_embed_seq,
                self.encoding[None, None].expand(
                    z_embed_seq.shape[0], 1, self.encoding.shape[-1]
                ),
            ],
            dim=-2,
        )
        z_embed_seq = self.dropout(z_embed_seq)
        padding_bits = torch.cat(
            [
                state_padding_bits,
                z_padding_bits,
                torch.zeros_like(z_padding_bits[:, :1]),
            ],
            dim=-1,
        )

        encoded_seq = self.encoder(
            z_embed_seq.transpose(1, 0),
            src_key_padding_mask=padding_bits,
        )

        # bs x emb_dim, (state_seq_len + z_seq_len) x bs x emb_dim
        return encoded_seq[-1], encoded_seq[:-1]


class StateEncoderDecoderTransformer(nn.Module):
    def __init__(
        self,
        n_state_components,
        input_size,
        embedding_dim,
        nlayers,
        nhead,
        norm_first,
        dropout_p,
        pad_word_idx,
    ):
        super().__init__()
        self.n_state_components = n_state_components
        self.embedding_dim = embedding_dim
        self.state_embedding = BOWEmbedding(
            64, self.n_state_components, self.embedding_dim
        )
        self.state_projection = nn.Linear(
            self.n_state_components * self.embedding_dim, self.embedding_dim
        )
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.embedding_projection = nn.Linear(embedding_dim * 2, embedding_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.encoding = nn.Parameter(torch.randn(embedding_dim))
        self.pos_encoding = PositionalEncoding1D(embedding_dim)
        self.bi = False
        self.pad_word_idx = pad_word_idx
        self.transformer = nn.Transformer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout_p,
            num_encoder_layers=nlayers,
            num_decoder_layers=nlayers,
            norm_first=norm_first,
        )

    def forward(self, state_padded, z_padded):
        state_padding_bits = torch.zeros_like(state_padded[..., 0]).bool()
        z_padding_bits = z_padded == self.pad_word_idx

        state_embed_seq = self.state_projection(self.state_embedding(state_padded))

        z_embed_seq = self.embedding(z_padded)
        z_embed_seq = torch.cat([self.pos_encoding(z_embed_seq), z_embed_seq], dim=-1)
        z_embed_seq = self.embedding_projection(z_embed_seq)
        z_embed_seq = torch.cat(
            [
                z_embed_seq,
                self.encoding[None, None].expand(
                    z_embed_seq.shape[0], 1, self.encoding.shape[-1]
                ),
            ],
            dim=-2,
        )
        state_embed_seq = self.dropout(state_embed_seq)
        z_embed_seq = self.dropout(z_embed_seq)
        padding_bits = torch.cat(
            [z_padding_bits, torch.zeros_like(z_padding_bits[:, :1])], dim=-1
        )

        encoded_seq = self.transformer(
            state_embed_seq.transpose(1, 0),
            z_embed_seq.transpose(1, 0),
            src_key_padding_mask=state_padding_bits,
            tgt_key_padding_mask=padding_bits,
        )

        # bs x emb_dim, z_seq_len x bs x emb_dim
        return encoded_seq[-1], encoded_seq[:-1]


class MetaNetRNN(nn.Module):
    # Meta Seq2Seq encoder
    #
    # Encodes query items in the context of the support set, which is stored in external memory.
    #
    #  Architecture
    #   1) Transformer encoder for input symbols in query and support items (either shared or separate)
    #   2) Transformer encoder for output symbols in the support items only
    #   3) Key-value memory for embedding query items with support context
    #   3) MLP to reduce the dimensionality of the context-sensitive embedding
    def __init__(
        self,
        embedding_dim,
        nlayers,
        nhead,
        pad_word_idx,
        pad_action_idx,
        norm_first=False,
        dropout_p=0.1,
        tie_encoders=True,
    ):
        #
        # Input
        #  embedding_dim : number of hidden units in Transformer encoder, and size of all embeddings
        #  input_size : number of input symbols
        #  output_size : number of output symbols
        #  nlayers : number of hidden layers in Transformer encoder
        #  dropout : dropout applied to symbol embeddings and Transformer
        #  tie_encoders : use the same encoder for the support and query items? (default=True)
        #
        super(MetaNetRNN, self).__init__()
        self.nlayers = nlayers
        self.embedding_dim = embedding_dim
        self.dropout_p = dropout_p
        self.attn = Attn()
        self.support_encoder = EncoderTransformer(
            embedding_dim=embedding_dim,
            nlayers=nlayers,
            nhead=nhead,
            norm_first=norm_first,
            dropout_p=dropout_p,
        )
        if tie_encoders:
            self.query_encoder = self.support_encoder
        else:
            self.query_encoder = EncoderTransformer(
                embedding_dim=embedding_dim,
                nlayers=nlayers,
                nhead=nhead,
                norm_first=norm_first,
                dropout_p=dropout_p,
            )
        self.output_encoder = EncoderTransformer(
            embedding_dim=embedding_dim,
            nlayers=nlayers,
            nhead=nhead,
            norm_first=norm_first,
            dropout_p=dropout_p,
        )
        self.pad_word_idx = pad_word_idx
        self.pad_action_idx = pad_action_idx
        self.hidden = nn.Linear(embedding_dim * 2, embedding_dim)
        self.tanh = nn.Tanh()

    def forward(
        self,
        query,
        x_supports,
        y_supports,
        query_key_padding_mask=None,
        x_supports_key_padding_mask=None,
        y_supports_key_padding_mask=None,
        supports_key_padding_mask=None,
    ):
        x_supports_key_padding_mask = (
            torch.zeros_like(x_supports[..., 0]).bool()
            if x_supports_key_padding_mask is None
            else x_supports_key_padding_mask
        )
        y_supports_key_padding_mask = (
            torch.zeros_like(y_supports[..., 0]).bool()
            if y_supports_key_padding_mask is None
            else y_supports_key_padding_mask
        )
        query_key_padding_mask = (
            torch.zeros_like(query[..., 0]).bool()
            if query_key_padding_mask is None
            else query_key_padding_mask
        )
        supports_key_padding_mask = (
            torch.zeros_like(x_supports[..., 0, 0]).bool()
            if supports_key_padding_mask is None
            else supports_key_padding_mask
        )
        #
        # Forward pass over an episode
        #
        # Input
        #   sample: episode dict wrapper for ns support and nq query examples (see 'build_sample' function in training code)
        #
        # Output
        #   context_last : [nq x embedding]; last step embedding for each query example
        #   embed_by_step: embedding at every step for each query [max_xq_length x nq x embedding_dim]
        #   attn_by_step : attention over support items at every step for each query [max_xq_length x nq x ns]
        #   seq_len : length of each query [nq list]
        #
        xs_padded = x_supports  # support set input sequences; LongTensor (ns x max_xs_length x emb_dim)
        ys_padded = y_supports  # support set output sequences; LongTensor (ns x max_ys_length x emb_dim)
        xq_padded = query  # query set input sequences; LongTensor (nq x max_xq_length x emb_dim)

        # xs_state = state[:, None].expand(-1, xs_padded.shape[1], -1).flatten(0, 1)
        xs_padded_flat = xs_padded.flatten(0, 1)  # (bs x ns) x xs_seq_len x emb_dim
        ys_padded_flat = ys_padded.flatten(0, 1)  # (bs x ns) x ys_seq_len x emb_dim
        xs_key_padding_mask_flat = x_supports_key_padding_mask.flatten(0, 1)
        ys_key_padding_mask_flat = y_supports_key_padding_mask.flatten(0, 1)

        # Embed the input sequences for support and query set
        embed_xs, _ = self.support_encoder(
            xs_padded_flat, key_padding_mask=xs_key_padding_mask_flat
        )  # bs x embedding, _
        embed_xq, embed_xq_by_step = self.query_encoder(
            xq_padded, key_padding_mask=query_key_padding_mask
        )  # bs x embedding, (state_seq_len? + xq_seq_len) x bs x emb_dim (embedding at each step)

        # Embed the output sequences for support set
        embed_ys, _ = self.output_encoder(
            ys_padded_flat, key_padding_mask=ys_key_padding_mask_flat
        )  # (bs x ns) x emb_dim

        embed_xs = embed_xs.unflatten(0, (xs_padded.shape[0], xs_padded.shape[1]))
        embed_ys = embed_ys.unflatten(0, (ys_padded.shape[0], ys_padded.shape[1]))

        # The original design had all queries using the same supports. In this
        # design each query uses only its own supports and nothing else. This means
        # that in contrast to the second last dimension being the batch size, it
        # should instead be 1 as we only have a single query for each of ns.

        # We have xq_len x bs x ns x e here
        embed_xs_expanded = embed_xs.expand(embed_xq_by_step.shape[0], -1, -1, -1)
        embed_ys_expanded = embed_ys.expand(embed_xq_by_step.shape[0], -1, -1, -1)

        # For embed_xq_by_step we should have seq_len x bs x 1 x e
        embed_xq_by_step_expanded = embed_xq_by_step[:, :, None, :]

        supports_key_padding_mask_expanded = supports_key_padding_mask[None].expand(
            embed_xq_by_step.shape[0], -1, -1
        )

        # The attention is Q(Sx^T) Sy which means that
        # for row i you have QiSx_0 Sy_0 ... Qi Sx_n Sy_n
        # eg a (seq_len x bs x 1 x e) x (seq_len x bs x e x ns) => seq_len x bs x 1 x ns
        # matrix. Remember only that there is one query per batch of supports.
        # Then if your values are seq_len x bs x ns x e, you have
        # (seq_len x bs x 1 x ns) x (seq_len x bs x ns x e) => (seq_len x bs x 1 x e)
        # which transposing gets you (1 x seq_len x bs x e), eg, encoded sequences
        # of queries with respect to each of their supports
        #
        # Then if your values are

        # Compute context based on key-value memory at each time step for queries
        value_by_step, attn_by_step = self.attn(
            embed_xq_by_step_expanded,  # (xq_seq_len) x bs x 1 x embedding_dim
            embed_xs_expanded,  # seq_len x bs x ns x e
            embed_ys_expanded,  # seq_len x bs x ns x e
            supports_key_padding_mask_expanded,  # seq_len x bs x ns
        )  # => (seq_len x bs x 1 x e)

        value_by_step = value_by_step.squeeze(-2)  # seq_len x bs x e

        # Unflatten everything
        # embed_xs = embed_xs.unflatten(0, (xs_padded.shape[0], xs_padded.shape[1])) # (bs x ns) x emb_dim
        # embed_ys = embed_ys.unflatten(0, (ys_padded.shape[0], ys_padded.shape[1])) # (bs x ns) x emb_dim

        concat_by_step = torch.cat(
            (embed_xq_by_step, value_by_step), -1
        )  # max_xq_length x nq x embedding_dim*2
        context_by_step = self.tanh(
            self.hidden(concat_by_step)
        )  # max_xq_length x nq x embedding_dim

        # Grab the last context for each query
        #
        # (state_seq_len? + xq_seq_len) x bs x emb_dim
        return context_by_step


class EncoderDecoderTransformer(nn.Module):
    def __init__(
        self,
        hidden_size,
        output_size,
        nlayers,
        nhead,
        norm_first,
        dropout_p=0.1,
    ):
        #
        # Input
        #  hidden_size : number of hidden units in Transformer, and embedding size for output symbols
        #  output_size : number of output symbols
        #  nlayers : number of hidden layers
        #  dropout_p : dropout applied to symbol embeddings and Transformers
        #
        super().__init__()
        self.nlayers = nlayers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout_p)
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_size,
                dim_feedforward=hidden_size * 4,
                dropout=dropout_p,
                nhead=nhead,
                norm_first=norm_first,
            ),
            num_layers=nlayers,
        )
        self.out = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        decoder_inputs,
        encoder_outputs,
        encoder_key_padding_mask=None,
        decoder_key_padding_mask=None,
    ):
        # Run batch decoder forward for a single time step.
        #
        # Input
        #  input: LongTensor of length batch_size x seq_len (left-shifted targets)
        #  memory: encoder state
        #
        # Output
        #   output : unnormalized output probabilities, batch_size x output_size
        #
        # Embed each input symbol
        # state, state_padding_bits = extract_padding(state)
        encoder_key_padding_mask = (
            torch.zeros_like(encoder_outputs[..., 0]).bool()
            if encoder_key_padding_mask is None
            else encoder_key_padding_mask
        )
        decoder_key_padding_mask = (
            torch.zeros_like(decoder_inputs[..., 0]).bool()
            if decoder_key_padding_mask is None
            else decoder_key_padding_mask
        )

        decoded = self.transformer(
            tgt=decoder_inputs.transpose(0, 1),
            memory=encoder_outputs,
            tgt_key_padding_mask=decoder_key_padding_mask,
            memory_key_padding_mask=encoder_key_padding_mask,
            tgt_mask=nn.Transformer.generate_square_subsequent_mask(
                decoder_inputs.shape[-2]
            ).to(decoder_inputs.device),
        ).transpose(0, 1)

        return self.out(decoded)


def init_parameters(module, scale=1e-2):
    if type(module) in [nn.LayerNorm]:
        return

    if type(module) in [nn.MultiheadAttention]:
        torch.nn.init.normal_(module.in_proj_weight, 0, scale)
        return

    if type(module) in [nn.Conv2d]:
        return

    if getattr(module, "weight", None) is not None:
        torch.nn.init.normal_(module.weight, 0, scale)

    if getattr(module, "bias", None) is not None:
        torch.nn.init.zeros_(module.bias)


class StateImaginationMetaLearner(pl.LightningModule):
    def __init__(
        self,
        n_state_components,
        x_categories,
        y_categories,
        embed_dim,
        dropout_p,
        nlayers,
        nhead,
        pad_word_idx,
        pad_action_idx,
        sos_action_idx,
        eos_action_idx,
        norm_first=False,
        lr=1e-4,
        wd=1e-2,
        warmup_proportion=0.001,
        decay_power=-1,
        predict_steps=64,
        metalearn_dropout_p=0.0,
        metalearn_include_permutations=False,
    ):
        super().__init__()
        self.state_embedding = BOWEmbedding(64, n_state_components, embed_dim)
        self.state_projection = nn.Linear(n_state_components * embed_dim, embed_dim)
        self.x_embedding = nn.Embedding(x_categories, embed_dim)
        self.y_embedding = nn.Embedding(y_categories, embed_dim)
        self.pos_encoding = PositionalEncoding1D(embed_dim)
        self.encoder = MetaNetRNN(
            embed_dim,
            nlayers,
            nhead,
            pad_word_idx,
            pad_action_idx,
            norm_first,
            dropout_p,
        )
        self.decoder = EncoderDecoderTransformer(
            hidden_size=embed_dim,
            output_size=y_categories,
            norm_first=norm_first,
            nlayers=nlayers,
            nhead=nhead,
            dropout_p=dropout_p,
        )
        self.y_categories = y_categories
        self.pad_word_idx = pad_word_idx
        self.pad_action_idx = pad_action_idx
        self.sos_action_idx = sos_action_idx
        self.eos_action_idx = eos_action_idx

        self.apply(init_parameters)
        self.save_hyperparameters()

    def configure_optimizers(self):
        return transformer_optimizer_config(
            self,
            self.hparams.lr,
            warmup_proportion=self.hparams.warmup_proportion,
            weight_decay=self.hparams.wd,
            decay_power=self.hparams.decay_power,
        )

    def encode(self, support_state, x_supports, y_supports, queries):
        return self.encoder(support_state, x_supports, y_supports, queries)

    def decode_autoregressive(
        self, decoder_in, encoder_outputs, encoder_padding, decoder_padding
    ):
        return self.decoder(
            decoder_in, encoder_outputs, encoder_padding, decoder_padding
        )

    def forward(
        self,
        query_state,
        support_state,
        x_supports,
        y_supports,
        user_support_mask,
        queries,
        decoder_in,
    ):
        # We expand the support_state if it was not already expanded.
        support_state = (
            support_state
            if support_state.ndim == 4
            else support_state[:, None].expand(-1, x_supports.shape[1], -1, -1)
        )

        query_state_mask = (query_state == 0).all(dim=-1)
        support_state_mask = (support_state == 0).all(dim=-1)
        x_support_mask = x_supports == self.pad_word_idx
        y_support_mask = y_supports == self.pad_action_idx
        query_mask = queries == self.pad_word_idx
        target_mask = decoder_in == self.pad_action_idx
        support_mask = x_support_mask.all(dim=-1)

        # Mask sequences that are actually padding and also those
        # specified in support_mask
        support_mask = support_mask | user_support_mask

        query_mask = torch.cat([query_state_mask, query_mask], dim=-1)
        x_support_mask = torch.cat([support_state_mask, x_support_mask], dim=-1)

        embed_query_state = self.state_projection(self.state_embedding(query_state))
        embed_support_states = self.state_projection(
            self.state_embedding(support_state)
        )

        embed_query = self.x_embedding(queries)
        embed_query = embed_query + self.pos_encoding(embed_query)
        embed_query_and_state = torch.cat([embed_query_state, embed_query], dim=-2)
        embed_x_supports = self.x_embedding(x_supports)
        embed_x_supports = embed_x_supports + self.pos_encoding(
            embed_x_supports.view(
                -1, embed_x_supports.shape[-2], embed_x_supports.shape[-1]
            )
        ).view(*embed_x_supports.shape)
        embed_x_supports_and_state = torch.cat(
            [embed_support_states, embed_x_supports], dim=-2
        )
        embed_y_supports = self.y_embedding(y_supports)
        embed_y_supports = embed_y_supports + self.pos_encoding(
            embed_y_supports.view(
                -1, embed_y_supports.shape[-2], embed_y_supports.shape[-1]
            )
        ).view(*embed_y_supports.shape)
        embed_target = self.y_embedding(decoder_in)

        encoded = self.encoder(
            embed_query_and_state,
            embed_x_supports_and_state,
            embed_y_supports,
            query_key_padding_mask=query_mask,
            x_supports_key_padding_mask=x_support_mask,
            y_supports_key_padding_mask=y_support_mask,
            supports_key_padding_mask=support_mask,
        )
        return self.decode_autoregressive(
            embed_target, encoded, query_mask, target_mask
        )

    def training_step(self, x, idx):
        (
            query_state,
            support_state,
            queries,
            targets,
            x_supports,
            y_supports,
        ) = x
        actions_mask = targets == self.pad_action_idx

        decoder_in = torch.cat(
            [torch.ones_like(targets)[:, :1] * self.sos_action_idx, targets], dim=-1
        )

        # Mask metalearn_dropout_p % of the supports
        support_mask = torch.randperm(x_supports.shape[1], device=self.device).expand(
            x_supports.shape[0], x_supports.shape[1]
        ) < int(x_supports.shape[1] * self.hparams.metalearn_dropout_p)

        # Now do the training
        preds = self.forward(
            query_state,
            support_state,
            x_supports,
            y_supports,
            support_mask,
            queries,
            decoder_in,
        )[:, :-1]

        # Ultimately we care about the cross entropy loss
        loss = F.cross_entropy(
            preds.flatten(0, -2),
            targets.flatten().long(),
            ignore_index=self.pad_action_idx,
        )

        argmax_preds = preds.argmax(dim=-1)
        argmax_preds[actions_mask] = self.pad_action_idx
        exacts = (argmax_preds == targets).all(dim=-1).to(torch.float).mean()

        self.log("tloss", loss, prog_bar=True)
        self.log("texact", exacts, prog_bar=True)
        self.log(
            "tacc",
            (preds.argmax(dim=-1)[~actions_mask] == targets[~actions_mask])
            .float()
            .mean(),
            prog_bar=True,
        )

        return loss

    def validation_step(self, x, idx, dl_idx=0):
        (
            query_state,
            support_state,
            queries,
            targets,
            x_supports,
            y_supports,
        ) = x
        actions_mask = targets == self.pad_action_idx

        decoder_in = torch.cat(
            [torch.ones_like(targets)[:, :1] * self.sos_action_idx, targets], dim=-1
        )

        support_mask = torch.zeros_like(x_supports[..., 0]).bool()

        # Now do the training
        preds = self.forward(
            query_state,
            support_state,
            x_supports,
            y_supports,
            support_mask,
            queries,
            decoder_in,
        )[:, :-1]

        # Ultimately we care about the cross entropy loss
        loss = F.cross_entropy(
            preds.flatten(0, -2),
            targets.flatten().long(),
            ignore_index=self.pad_action_idx,
        )

        argmax_preds = preds.argmax(dim=-1)
        argmax_preds[actions_mask] = self.pad_action_idx
        exacts = (argmax_preds == targets).all(dim=-1).to(torch.float).mean()

        self.log("vloss", loss, prog_bar=True)
        self.log("vexact", exacts, prog_bar=True)
        self.log(
            "vacc",
            (preds.argmax(dim=-1)[~actions_mask] == targets[~actions_mask])
            .float()
            .mean(),
            prog_bar=True,
        )

    def predict_step(self, x, idx, dl_idx=0):
        (
            query_state,
            support_state,
            queries,
            targets,
            x_supports,
            y_supports,
        ) = x
        decoder_in = torch.ones_like(targets)[:, :1] * self.sos_action_idx
        support_mask = torch.zeros_like(x_supports[..., 0]).bool()

        padding = queries == self.pad_word_idx

        # We do autoregressive prediction, predict for as many steps
        # as there are targets, but don't use teacher forcing
        logits = []

        with torch.no_grad():
            encoded = self.encoder(
                query_state,
                # We expand the support_state if it was not already expanded.
                support_state
                if support_state.ndim == 4
                else support_state[:, None].expand(-1, x_supports.shape[1], -1, -1),
                x_supports,
                y_supports,
                support_mask,
                queries,
            )

            for i in range(targets.shape[1]):
                logits.append(
                    self.decode_autoregressive(decoder_in, encoded, padding, None)[
                        :, -1
                    ]
                )
                decoder_out = logits[-1].argmax(dim=-1)
                decoder_in = torch.cat([decoder_in, decoder_out[:, None]], dim=1)

            decoded_eq_mask = (
                (decoder_in == self.eos_action_idx).int().cumsum(dim=-1).bool()[:, :-1]
            )
            decoded = decoder_in[:, 1:]
            decoded[decoded_eq_mask] = self.pad_action_idx
            logits = torch.stack(logits, dim=1)

        exacts = (decoded == targets).all(dim=-1)

        return decoded, logits, exacts


class SequenceImaginationMetaLearner(pl.LightningModule):
    def __init__(
        self,
        x_categories,
        y_categories,
        embed_dim,
        dropout_p,
        nlayers,
        nhead,
        pad_word_idx,
        pad_action_idx,
        sos_action_idx,
        eos_action_idx,
        norm_first=False,
        lr=1e-4,
        wd=1e-2,
        warmup_proportion=0.001,
        decay_power=-1,
        predict_steps=64,
        metalearn_dropout_p=0.0,
        metalearn_include_permutations=False,
    ):
        super().__init__()
        self.x_embedding = nn.Embedding(x_categories, embed_dim)
        self.y_embedding = nn.Embedding(y_categories, embed_dim)
        self.pos_encoding = PositionalEncoding1D(embed_dim)
        self.encoder = MetaNetRNN(
            embed_dim,
            nlayers,
            nhead,
            pad_word_idx,
            pad_action_idx,
            norm_first,
            dropout_p,
        )
        self.decoder = EncoderDecoderTransformer(
            hidden_size=embed_dim,
            output_size=y_categories,
            norm_first=norm_first,
            nlayers=nlayers,
            nhead=nhead,
            dropout_p=dropout_p,
        )
        self.y_categories = y_categories
        self.pad_word_idx = pad_word_idx
        self.pad_action_idx = pad_action_idx
        self.sos_action_idx = sos_action_idx
        self.eos_action_idx = eos_action_idx

        self.apply(init_parameters)
        self.save_hyperparameters()

    def configure_optimizers(self):
        return transformer_optimizer_config(
            self,
            self.hparams.lr,
            warmup_proportion=self.hparams.warmup_proportion,
            weight_decay=self.hparams.wd,
            decay_power=self.hparams.decay_power,
        )

    def encode(self, x_supports, y_supports, queries):
        return self.encoder(x_supports, y_supports, queries)

    def decode_autoregressive(
        self, decoder_in, encoder_outputs, encoder_padding, decoder_padding
    ):
        return self.decoder(
            decoder_in, encoder_outputs, encoder_padding, decoder_padding
        )

    def forward(
        self,
        x_supports,
        y_supports,
        user_support_mask,
        queries,
        decoder_in,
    ):
        x_support_mask = x_supports == self.pad_word_idx
        y_support_mask = y_supports == self.pad_action_idx
        query_mask = queries == self.pad_word_idx
        target_mask = decoder_in == self.pad_action_idx
        support_mask = x_support_mask.all(dim=-1)

        # Mask sequences that are actually padding and also those
        # specified in support_mask
        support_mask = support_mask | user_support_mask

        embed_query = self.x_embedding(queries)
        embed_query = embed_query + self.pos_encoding(embed_query)
        embed_x_supports = self.x_embedding(x_supports)
        embed_x_supports = embed_x_supports + self.pos_encoding(
            embed_x_supports.view(
                -1, embed_x_supports.shape[-2], embed_x_supports.shape[-1]
            )
        ).view(*embed_x_supports.shape)
        embed_y_supports = self.y_embedding(y_supports)
        embed_y_supports = embed_y_supports + self.pos_encoding(
            embed_y_supports.view(
                -1, embed_y_supports.shape[-2], embed_y_supports.shape[-1]
            )
        ).view(*embed_y_supports.shape)
        embed_target = self.y_embedding(decoder_in)

        encoded = self.encoder(
            embed_query,
            embed_x_supports,
            embed_y_supports,
            query_key_padding_mask=query_mask,
            x_supports_key_padding_mask=x_support_mask,
            y_supports_key_padding_mask=y_support_mask,
            supports_key_padding_mask=support_mask,
        )
        return self.decode_autoregressive(
            embed_target, encoded, query_mask, target_mask
        )

    def training_step(self, x, idx):
        (
            queries,
            targets,
            x_supports,
            y_supports,
        ) = x
        actions_mask = targets == self.pad_action_idx

        decoder_in = torch.cat(
            [torch.ones_like(targets)[:, :1] * self.sos_action_idx, targets], dim=-1
        )

        # Mask metalearn_dropout_p % of the supports
        support_mask = torch.randperm(x_supports.shape[1], device=self.device).expand(
            x_supports.shape[0], x_supports.shape[1]
        ) < int(x_supports.shape[1] * self.hparams.metalearn_dropout_p)

        # Now do the training
        preds = self.forward(
            x_supports,
            y_supports,
            support_mask,
            queries,
            decoder_in,
        )[:, :-1]

        # Ultimately we care about the cross entropy loss
        loss = F.cross_entropy(
            preds.flatten(0, -2),
            targets.flatten().long(),
            ignore_index=self.pad_action_idx,
        )

        argmax_preds = preds.argmax(dim=-1)
        argmax_preds[actions_mask] = self.pad_action_idx
        exacts = (argmax_preds == targets).all(dim=-1).to(torch.float).mean()

        self.log("tloss", loss, prog_bar=True)
        self.log("texact", exacts, prog_bar=True)
        self.log(
            "tacc",
            (preds.argmax(dim=-1)[~actions_mask] == targets[~actions_mask])
            .float()
            .mean(),
            prog_bar=True,
        )

        return loss

    def validation_step(self, x, idx, dl_idx=0):
        (
            queries,
            targets,
            x_supports,
            y_supports,
        ) = x
        actions_mask = targets == self.pad_action_idx

        decoder_in = torch.cat(
            [torch.ones_like(targets)[:, :1] * self.sos_action_idx, targets], dim=-1
        )

        support_mask = torch.zeros_like(x_supports[..., 0]).bool()

        # Now do the training
        preds = self.forward(
            x_supports,
            y_supports,
            support_mask,
            queries,
            decoder_in,
        )[:, :-1]

        # Ultimately we care about the cross entropy loss
        loss = F.cross_entropy(
            preds.flatten(0, -2),
            targets.flatten().long(),
            ignore_index=self.pad_action_idx,
        )

        argmax_preds = preds.argmax(dim=-1)
        argmax_preds[actions_mask] = self.pad_action_idx
        exacts = (argmax_preds == targets).all(dim=-1).to(torch.float).mean()

        self.log("vloss", loss, prog_bar=True)
        self.log("vexact", exacts, prog_bar=True)
        self.log(
            "vacc",
            (preds.argmax(dim=-1)[~actions_mask] == targets[~actions_mask])
            .float()
            .mean(),
            prog_bar=True,
        )

    def predict_step(self, x, idx, dl_idx=0):
        (
            queries,
            targets,
            x_supports,
            y_supports,
        ) = x
        decoder_in = torch.ones_like(targets)[:, :1] * self.sos_action_idx
        support_mask = torch.zeros_like(x_supports[..., 0]).bool()
        padding = queries == self.pad_word_idx

        # We do autoregressive prediction, predict for as many steps
        # as there are targets, but don't use teacher forcing
        logits = []

        with torch.no_grad():
            encoded = self.encoder(
                x_supports,
                y_supports,
                support_mask,
                queries,
            )

            for i in range(targets.shape[1]):
                logits.append(
                    self.decode_autoregressive(decoder_in, encoded, padding, None)[
                        :, -1
                    ]
                )
                decoder_out = logits[-1].argmax(dim=-1)
                decoder_in = torch.cat([decoder_in, decoder_out[:, None]], dim=1)

            decoded_eq_mask = (
                (decoder_in == self.eos_action_idx).int().cumsum(dim=-1).bool()[:, :-1]
            )
            decoded = decoder_in[:, 1:]
            decoded[decoded_eq_mask] = self.pad_action_idx
            logits = torch.stack(logits, dim=1)

        exacts = (decoded == targets).all(dim=-1)

        return decoded, logits, exacts


def compress_permute(arrays, generator, pad_idx):
    concat_array = arrays.reshape(-1)
    max_index = concat_array.max()
    putback_values = np.arange(max_index + 1, dtype=int)
    putback_mask = putback_values >= pad_idx

    # First create an array of 1s where the indices are
    zeros = np.zeros(max_index + 1, dtype=int)
    zeros[concat_array] = 1
    zeros[putback_mask] = 0

    zeros_bool = zeros.astype(bool)

    # Then determine how many non-zero elements we have in that array
    num_nonzero = zeros[zeros_bool].shape[0]

    # Then assign the nonzero elements to something
    # in a permutation in the range of num_nonzero
    zeros[zeros_bool] = generator.permutation(num_nonzero)

    # Pad index gets mapped back to pad_idx
    zeros[putback_mask] = putback_values[putback_mask]

    # Now look up the original indices for each array
    # in the permutation
    return zeros[arrays]


def regular_permute(arrays, generator, pad_idx, categories):
    # Do the permutation
    permutation = np.arange(x_categories)
    permutation[0:pad_idx] = permutation[0:pad_idx][self.generator.permutation(pad_idx)]

    return permutation[arrays]


class PermuteActionsDataset(Dataset):
    def __init__(
        self,
        dataset,
        x_categories,
        y_categories,
        pad_word_idx,
        pad_action_idx,
        shuffle=True,
        compress=True,
        seed=0,
    ):
        super().__init__()
        self.dataset = dataset
        self.x_categories = x_categories
        self.y_categories = y_categories
        self.pad_word_idx = pad_word_idx
        self.pad_action_idx = pad_action_idx
        self.shuffle = shuffle
        self.compress = compress
        self.generator = np.random.default_rng(seed)

    def state_dict(self):
        return {"random_state": self.generator.__getstate__()}

    def load_state_dict(self, sd):
        if "random_state" in sd:
            self.generator.__setstate__(sd["random_state"])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        queries, targets, x_supports, y_supports = [
            np.copy(x) for x in self.dataset[idx]
        ]

        # Compute permutations of outputs
        if self.shuffle:
            if self.compress:
                query_and_x_supports = compress_permute(
                    np.concatenate([queries[None], x_supports], axis=0),
                    self.generator,
                    self.pad_word_idx,
                )
                query_and_y_supports = compress_permute(
                    np.concatenate([targets[None], y_supports], axis=0),
                    self.generator,
                    self.pad_action_idx,
                )
            else:
                query_and_x_supports = regular_permute(
                    np.concatenate([queries[None], x_supports], axis=0),
                    self.generator,
                    self.pad_word_idx,
                    self.x_categories,
                )
                query_and_y_supports = regular_permute(
                    np.concatenate([targets[None], y_supports], axis=0),
                    self.generator,
                    self.pad_action_idx,
                    self.y_categories,
                )
            queries, x_supports = query_and_x_supports[0], query_and_x_supports[1:]
            targets, y_supports = query_and_y_supports[0], query_and_y_supports[1:]

        return (
            queries,
            targets,
            x_supports,
            y_supports,
        )


class ShuffleDemonstrationsDataset(Dataset):
    def __init__(self, dataset, seed=0):
        super().__init__()
        self.dataset = dataset
        self.generator = np.random.default_rng(seed)

    def state_dict(self):
        return {"random_state": self.generator.__getstate__()}

    def load_state_dict(self, sd):
        if "random_state" in sd:
            self.generator.__setstate__(sd["random_state"])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        (
            queries,
            targets,
            x_supports,
            y_supports,
        ) = self.dataset[idx]
        x_supports = np.copy(x_supports)
        y_supports = np.copy(y_supports)

        support_permutation = self.generator.permutation(x_supports.shape[0])

        return (
            queries,
            targets,
            x_supports[support_permutation],
            y_supports[support_permutation],
        )


class ReorderSupportsByDistanceDataset(Dataset):
    def __init__(self, dataset, limit):
        super().__init__()
        self.dataset = dataset
        self.limit = limit

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        (queries, targets, x_supports, y_supports, similarity_logit) = self.dataset[idx]

        order = (-np.array(similarity_logit)).argsort()[: self.limit]

        return (
            queries,
            targets,
            [x_supports[i] for i in order],
            [y_supports[i] for i in order],
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-demonstrations", type=str, required=True)
    parser.add_argument("--valid-demonstrations-directory", type=str, required=True)
    parser.add_argument("--dictionary", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-batch-size", type=int, default=1024)
    parser.add_argument("--valid-batch-size", type=int, default=128)
    parser.add_argument("--batch-size-mult", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--nlayers", type=int, default=8)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--norm-first", action="store_true")
    parser.add_argument("--dropout-p", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-2)
    parser.add_argument("--warmup-proportion", type=float, default=0.1)
    parser.add_argument("--decay-power", type=int, default=-1)
    parser.add_argument("--iterations", type=int, default=2500000)
    parser.add_argument("--disable-shuffle", action="store_true")
    parser.add_argument("--check-val-every", type=int, default=500)
    parser.add_argument("--limit-val-size", type=int, default=None)
    parser.add_argument("--enable-progress", action="store_true")
    parser.add_argument("--restore-from-checkpoint", action="store_true")
    parser.add_argument("--version", type=str, default=None)
    parser.add_argument("--dataset-name", type=str, default="gscan")
    parser.add_argument("--tag", type=str, default="none")
    parser.add_argument("--metalearn-dropout-p", type=float, default=0.0)
    parser.add_argument("--metalearn-demonstrations-limit", type=int, default=6)
    parser.add_argument("--metalearn-include-permutations", action="store_true")
    parser.add_argument("--pad-instructions-to", type=int, default=8)
    parser.add_argument("--pad-actions-to", type=int, default=128)
    parser.add_argument("--pad-state-to", type=int, default=36)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--limit-load", type=int, default=None)
    parser.add_argument("--dataloader-ncpus", type=int, default=1)
    args = parser.parse_args()

    exp_name = "cogs"
    model_name = f"meta_imagination_transformer_l_{args.nlayers}_h_{args.nhead}_d_{args.hidden_size}"
    dataset_name = args.dataset_name
    effective_batch_size = args.train_batch_size * args.batch_size_mult
    exp_name = f"{exp_name}_s_{args.seed}_m_{model_name}_it_{args.iterations}_b_{effective_batch_size}_d_{dataset_name}_t_{args.tag}_drop_{args.dropout_p}"
    model_dir = f"models/{exp_name}/{model_name}"
    model_path = f"{model_dir}/{exp_name}.pt"
    print(model_path)
    print(
        f"Batch size {args.train_batch_size}, mult {args.batch_size_mult}, total {args.train_batch_size * args.batch_size_mult}"
    )

    os.makedirs(model_dir, exist_ok=True)

    if os.path.exists(f"{model_path}"):
        print(f"Skipping {model_path} as it already exists")
        return

    seed = args.seed
    iterations = args.iterations

    (
        (WORD2IDX, ACTION2IDX),
        (meta_train_demonstrations, meta_valid_demonstrations_dict),
    ) = load_data_directories(
        args.train_demonstrations, args.dictionary, limit_load=args.limit_load
    )

    IDX2WORD = {i: w for w, i in WORD2IDX.items()}
    IDX2ACTION = {i: w for w, i in ACTION2IDX.items()}

    pad_word = WORD2IDX["[pad]"]
    pad_action = ACTION2IDX["[pad]"]
    sos_action = ACTION2IDX["[sos]"]
    eos_action = ACTION2IDX["[eos]"]

    pl.seed_everything(0)
    meta_train_dataset = ReshuffleOnIndexZeroDataset(
        PermuteActionsDataset(
            PaddingDataset(
                ReorderSupportsByDistanceDataset(
                    MapDataset(
                        MapDataset(meta_train_demonstrations, lambda x: x), lambda x: x
                    ),
                    args.metalearn_demonstrations_limit,
                ),
                (
                    args.pad_instructions_to,
                    args.pad_actions_to,
                    (args.metalearn_demonstrations_limit, args.pad_instructions_to),
                    (args.metalearn_demonstrations_limit, args.pad_actions_to),
                ),
                (
                    pad_word,
                    pad_action,
                    pad_word,
                    pad_action,
                ),
            ),
            len(WORD2IDX),
            len(ACTION2IDX),
            pad_word,
            pad_action,
            shuffle=not args.disable_shuffle,
            # We are testing different random initializations, but
            # keeping the dataloading order constant
            seed=0,
        )
    )

    pl.seed_everything(seed)
    meta_module = SequenceImaginationMetaLearner(
        len(IDX2WORD),
        len(IDX2ACTION),
        args.hidden_size,
        args.dropout_p,
        args.nlayers,
        args.nhead,
        pad_word,
        pad_action,
        sos_action,
        eos_action,
        norm_first=args.norm_first,
        lr=args.lr,
        decay_power=args.decay_power,
        warmup_proportion=args.warmup_proportion,
        metalearn_dropout_p=args.metalearn_dropout_p,
        metalearn_include_permutations=args.metalearn_include_permutations,
    )
    print(meta_module)

    train_dataloader = DataLoader(
        meta_train_dataset,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_ncpus,
    )

    check_val_opts = {}
    interval = args.check_val_every / len(train_dataloader)

    # Every check_val_interval steps, regardless of how large the training dataloader is
    if interval > 1.0:
        check_val_opts["check_val_every_n_epoch"] = math.floor(interval)
    else:
        check_val_opts["val_check_interval"] = interval

    checkpoint_cb = ModelCheckpoint(
        monitor="vexact/dataloader_idx_0",
        auto_insert_metric_name=False,
        save_top_k=5,
        mode="max",
    )

    logs_root_dir = f"{args.log_dir}/{exp_name}/{model_name}/{dataset_name}/{seed}"
    most_recent_version = args.version

    meta_trainer = pl.Trainer(
        logger=[
            TensorBoardLogger(logs_root_dir, version=most_recent_version),
            LoadableCSVLogger(
                logs_root_dir, version=most_recent_version, flush_logs_every_n_steps=10
            ),
        ],
        callbacks=[pl.callbacks.LearningRateMonitor(), checkpoint_cb],
        max_steps=iterations,
        num_sanity_val_steps=10,
        gpus=1 if torch.cuda.is_available() else 0,
        precision=16 if torch.cuda.is_available() else None,
        default_root_dir=logs_root_dir,
        accumulate_grad_batches=args.batch_size_mult,
        enable_progress_bar=sys.stdout.isatty() or args.enable_progress,
        gradient_clip_val=0.2,
        **check_val_opts,
    )

    valid_dataloaders = [
        DataLoader(
            PermuteActionsDataset(
                PaddingDataset(
                    ReorderSupportsByDistanceDataset(
                        MapDataset(
                            MapDataset(
                                Subset(
                                    demonstrations,
                                    np.random.permutation(len(demonstrations))[
                                        : args.limit_val_size
                                    ],
                                ),
                                lambda x: x,
                            ),
                            lambda x: x,
                        ),
                        args.metalearn_demonstrations_limit,
                    ),
                    (
                        args.pad_instructions_to,
                        args.pad_actions_to,
                        (args.metalearn_demonstrations_limit, args.pad_instructions_to),
                        (args.metalearn_demonstrations_limit, args.pad_actions_to),
                    ),
                    (
                        pad_word,
                        pad_action,
                        pad_word,
                        pad_action,
                    ),
                ),
                len(WORD2IDX),
                len(ACTION2IDX),
                pad_word,
                pad_action,
                shuffle=True,
                compress=True,
            ),
            batch_size=max([args.train_batch_size, args.valid_batch_size]),
            pin_memory=True,
        )
        for demonstrations in meta_valid_demonstrations_dict.values()
    ]

    meta_trainer.fit(meta_module, train_dataloader, valid_dataloaders)
    print(f"Done, saving {model_path}")
    meta_trainer.save_checkpoint(f"{model_path}")


if __name__ == "__main__":
    main()
