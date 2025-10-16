import argparse
import copy
import os
import numpy as np
import math

# Import pillow and matplotlib first before torch pulls in a different libc
import matplotlib.pyplot as plt
from PIL import Image
import gc

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

from positional_encodings.torch_encodings import (
    PositionalEncoding1D,
    PositionalEncoding2D,
)

from gscan_metaseq2seq.models.embedding import BOWEmbedding
from gscan_metaseq2seq.util.dataset import (
    PaddingDataset,
    ReshuffleOnIndexZeroDataset,
    MapDataset,
    ReorderSupportsByDistanceDataset,
)
from gscan_metaseq2seq.util.load_data import load_data, load_data_directories
from gscan_metaseq2seq.util.logging import LoadableCSVLogger, ConsoleLogger
from gscan_metaseq2seq.util.scheduler import transformer_optimizer_config
from gscan_metaseq2seq.util.padding import pad_to

from gscan_metaseq2seq.util.solver import (
    create_vocabulary,
    create_world,
    state_to_situation,
    reinitialize_world,
)


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


class ImagePatchEncoding(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        patches = self.conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        return patches.flatten(1, 2)


class DictEmbedding(nn.Module):
    def __init__(self, keys, embed_dim, padding_key=None):
        super().__init__()
        for i, key in enumerate(keys):
            self.register_buffer(key, torch.tensor(i, dtype=torch.long))
        self.embedding = nn.Embedding(
            len(keys), embed_dim, padding_idx=getattr(self, padding_key).item()
        )

    def __getitem__(self, key):
        return self.embedding(getattr(self, key))


class BigImgTransformerLearner(pl.LightningModule):
    def __init__(
        self,
        in_channels,
        patch_size,
        in_vocab_size,
        out_vocab_size,
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
        max_context_size=1024,
        need_support_states=False
    ):
        super().__init__()
        self.patch_encoding = ImagePatchEncoding(in_channels, embed_dim, patch_size)
        self.in_embedding = nn.Embedding(
            in_vocab_size, embed_dim, padding_idx=pad_word_idx
        )
        self.out_embedding = nn.Embedding(
            out_vocab_size, embed_dim, padding_idx=pad_action_idx
        )
        self.pos_encoding = PositionalEncoding1D(embed_dim)
        self.special_tokens = DictEmbedding(
            [
                "pad",
                "sos",
                "eos",
                "sep_state_support",
                "sep_instr_support",
                "sep_target_support",
                "sep_query_state",
                "sep_query_instr",
                "sep_query_end",
            ],
            embed_dim,
            padding_key="pad",
        )
        self.dropout = nn.Dropout(p=dropout_p)
        self.norm = nn.LayerNorm(embed_dim)
        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=nhead,
            dropout=dropout_p,
            norm_first=norm_first,
            activation=F.silu,
            num_encoder_layers=nlayers,
            num_decoder_layers=nlayers,
        )
        self.out = nn.Linear(embed_dim, out_vocab_size)
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
            optimizer_kwargs={"fused": True},
        )

    def encode(self, support_state, x_supports, y_supports, queries):
        return self.encoder(support_state, x_supports, y_supports, queries)

    def decode_autoregressive(self, decoder_in, encoder_outputs, encoder_padding):
        return self.decoder(decoder_in, encoder_outputs, encoder_padding)

    def assemble_multimodal_inputs(
        self,
        query_state_img,
        support_state_imgs,
        query_instruction,
        query_decoder_in,
        support_instructions,
        support_targets,
    ):
        # We have to do some fairly heavy lifting in the forward function
        # to assemble the input sequence from the multimodal inputs
        encoded_query_state_img = self.patch_encoding(query_state_img.float())
        support_img_padding = (support_state_imgs.flatten(-3, -1) == 0.0).all(dim=-1)
        encoded_support_state_img = self.patch_encoding(
            support_state_imgs.float().flatten(0, 1)
        ).unflatten(0, (support_state_imgs.shape[0], support_state_imgs.shape[1]))

        encoded_query_instruction = self.in_embedding(query_instruction)
        encoded_support_instructions = self.in_embedding(support_instructions)
        encoded_support_outputs = self.out_embedding(support_targets)
        encoded_query_decoder_in = self.out_embedding(query_decoder_in)

        # Now that we have everything, we have to select elements according
        # to what is not padded, and try to keep things in batches
        #
        # The basic idea is that we assemble everything into a big sequence
        # first and also keep track of a big "padding sequence" and its inverse.

        # B x S x L x E => B x (S x L) x E
        context_in = torch.cat(
            [
                torch.cat(
                    [
                        self.special_tokens["sep_state_support"][None, None, None]
                        .expand(
                            encoded_support_state_img.shape[0],
                            encoded_support_state_img.shape[1],
                            1,
                            -1,
                        )
                        .clone(),
                        *([encoded_support_state_img] if self.hparams.need_support_states else []),
                        self.special_tokens["sep_instr_support"][None, None, None]
                        .expand(
                            encoded_support_state_img.shape[0],
                            encoded_support_state_img.shape[1],
                            1,
                            -1,
                        )
                        .clone(),
                        encoded_support_instructions,
                        self.special_tokens["sep_target_support"][None, None, None]
                        .expand(
                            encoded_support_state_img.shape[0],
                            encoded_support_state_img.shape[1],
                            1,
                            -1,
                        )
                        .clone(),
                        encoded_support_outputs,
                    ],
                    dim=-2,
                ).flatten(1, 2),
                self.special_tokens["sep_query_state"][None, None].expand(
                    encoded_query_state_img.shape[0], 1, -1
                ),
                encoded_query_state_img,
                self.special_tokens["sep_query_instr"][None, None].expand(
                    encoded_query_instruction.shape[0], 1, -1
                ),
                encoded_query_instruction,
                self.special_tokens["sep_query_end"][None, None].expand(
                    encoded_query_instruction.shape[0], 1, -1
                ),
            ],
            dim=-2,
        )

        sep_nopad_token = torch.zeros_like(
            self.special_tokens["sep_state_support"][..., 0][
                None, None, None
            ]
            .expand(
                encoded_support_state_img.shape[0],
                encoded_support_state_img.shape[1],
                1,
            )
            .clone()
        ).bool()

        # Now we make the corresponding padding sequence. A value
        # of True means padded
        #
        # B x S x L => B x (S x L)
        padding_sequence = torch.cat(
            [
                torch.cat(
                    [
                        sep_nopad_token,
                        *([support_img_padding[:, :, None].expand(
                            encoded_support_state_img.shape[0],
                            encoded_support_state_img.shape[1],
                            encoded_support_state_img.shape[2],
                        )] if self.hparams.need_support_states else []),
                        sep_nopad_token,
                        support_instructions == self.pad_word_idx,
                        sep_nopad_token,
                        support_targets == self.pad_action_idx,
                    ],
                    dim=-1,
                ).flatten(1, 2),
                sep_nopad_token[:, 0],
                torch.zeros_like(encoded_query_state_img[..., 0]).bool(),
                sep_nopad_token[:, 0],
                query_instruction == self.pad_word_idx,
                sep_nopad_token[:, 0],
            ],
            dim=-1,
        )
        inv_padding_sequence = ~padding_sequence

        # Now that we have the padding sequences, we can assemble
        # the selection indices by taking the cumulative sum of
        # padding - 1
        non_padding_target_idx = (
            inv_padding_sequence.int().cumsum(dim=-1) - 1
        ) * inv_padding_sequence.int()

        # All the padding goes to indices that are now offset by the max
        # value of the non-padding selection index + 1, eg, if the
        # non-padding ended at index 10, then padding starts at index 11
        padding_target_idx = (
            padding_sequence.int().cumsum(dim=-1)
            - 1
            + (non_padding_target_idx.max(dim=-1).values[..., None] + 1)
        ) * padding_sequence.int()

        # Add them together to get the target index. This is
        # where we want all the elements to go in the padding_idx
        # so that the padding elements go to the end.
        target_idx = non_padding_target_idx  + padding_target_idx

        # Now we have to argsort the target_idx to get the reselection_idx.
        # This tells us the order in which to select elements from the
        # original sequence such that we get all the non-padding elements
        # first and then all the padding at the end. We can use torch.gather
        # with this.
        reselection_idx = target_idx.argsort(dim=-1)

        # Assign a single padding value to all the padding in the
        # original tensor
        context_in = (
            context_in * inv_padding_sequence[..., None].float()
            + self.special_tokens["pad"][None, None].expand(
                context_in.shape[0], context_in.shape[1], -1
            )
            * padding_sequence[..., None].float()
        )

        # Now we can select from the original tensor according to the
        # the reselection_idx
        context_in = torch.gather(
            context_in,
            -2,
            reselection_idx[..., None].expand(
                reselection_idx.shape[0], reselection_idx.shape[1], context_in.shape[-1]
            ),
        )
        context_pad = torch.gather(padding_sequence, -1, reselection_idx)

        # Now we can truncate the context_in according to the
        # max_context_size
        context_in = context_in[..., : self.hparams.max_context_size, :]
        context_pad = context_pad[..., : self.hparams.max_context_size]

        return (
            context_in,
            context_pad,
            encoded_query_decoder_in,
            query_decoder_in == self.pad_action_idx,
        )

    def forward(self, x):
        (
            query_state_img,
            query_instruction,
            query_decoder_in,
            support_state_imgs,
            support_instructions,
            support_targets,
        ) = x
        (
            context_in,
            context_pad,
            decoder_in,
            decoder_pad,
        ) = self.assemble_multimodal_inputs(
            query_state_img,
            query_instruction,
            query_decoder_in,
            support_state_imgs,
            support_instructions,
            support_targets,
        )

        # Now we can add positional encodings to everything
        # and do the usual norm + dropout
        context_in = context_in + self.pos_encoding(context_in)
        context_in = self.dropout(self.norm(context_in))
        decoder_in = decoder_in + self.pos_encoding(decoder_in)
        decoder_in = self.dropout(self.norm(decoder_in))

        # Regular old transformer
        decoded_sequence = self.transformer(
            src=context_in.transpose(0, 1),
            tgt=decoder_in.transpose(0, 1),
            src_key_padding_mask=context_pad,
            memory_key_padding_mask=context_pad,
            tgt_key_padding_mask=decoder_pad,
            tgt_mask=nn.Transformer.generate_square_subsequent_mask(
                decoder_pad.shape[-1]
            )
            .bool()
            .to(decoder_in.device),
        ).transpose(0, 1)

        return self.out(decoded_sequence)

    def training_step(self, x, idx):
        (
            query_state_img,
            support_state_imgs,
            query_instruction,
            targets,
            support_instructions,
            support_targets,
        ) = x

        decoder_in = torch.cat(
            [torch.ones_like(targets)[:, :1] * self.sos_action_idx, targets],
            dim=-1,
        )[:, :-1]

        preds = self.forward(
            (
                query_state_img,
                support_state_imgs,
                query_instruction,
                decoder_in,
                support_instructions,
                support_targets,
            )
        )

        actions_mask = targets == self.pad_action_idx

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

    def validation_step(self, x, idx, dataloader_idx=0):
        (
            query_state_img,
            support_state_imgs,
            query_instruction,
            targets,
            support_instructions,
            support_targets,
        ) = x

        decoder_in = torch.cat(
            [torch.ones_like(targets)[:, :1] * self.sos_action_idx, targets],
            dim=-1,
        )[:, :-1]

        preds = self.forward(
            (
                query_state_img,
                support_state_imgs,
                query_instruction,
                decoder_in,
                support_instructions,
                support_targets,
            )
        )

        actions_mask = targets == self.pad_action_idx

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

        return loss

    def predict_step(self, x, idx, dl_idx=0):
        (
            x_permutation,
            y_permutation,
            query_state,
            support_state,
            queries,
            targets,
            x_supports,
            y_supports,
        ) = x
        decoder_in = torch.ones_like(targets)[:, :1] * self.sos_action_idx
        support_mask = torch.zeros_like(x_supports[..., 0]).bool()

        if self.hparams.metalearn_include_permutations:
            # We concatenate the x and y permutations to the supports, this way
            # they get encoded and also go through the attention process. The
            # permutations are never masked.
            x_supports = torch.cat([x_permutation[..., None, :], x_supports], dim=1)
            y_supports = torch.cat([y_permutation[..., None, :], y_supports], dim=1)
            support_mask = torch.cat(
                [torch.zeros_like(x_permutation[..., :1]).bool(), support_mask], dim=1
            )

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
                    self.decode_autoregressive(decoder_in, encoded, padding)[:, -1]
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


class PermuteActionsDataset(Dataset):
    def __init__(
        self,
        dataset,
        x_categories,
        y_categories,
        pad_word_idx,
        pad_action_idx,
        shuffle=True,
        seed=0,
    ):
        super().__init__()
        self.dataset = dataset
        self.x_categories = x_categories
        self.y_categories = y_categories
        self.pad_word_idx = pad_word_idx
        self.pad_action_idx = pad_action_idx
        self.shuffle = shuffle
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
            query_state,
            support_state,
            queries,
            targets,
            x_supports,
            y_supports,
        ) = self.dataset[idx]

        x_permutation = np.arange(self.x_categories)
        y_permutation = np.arange(self.y_categories)

        # Compute permutations of outputs
        if self.shuffle:
            # Do the permutation
            x_permutation[0 : self.pad_word_idx] = x_permutation[0 : self.pad_word_idx][
                self.generator.permutation(self.pad_word_idx)
            ]
            y_permutation[0 : self.pad_action_idx] = y_permutation[
                0 : self.pad_action_idx
            ][self.generator.permutation(self.pad_action_idx)]

            # Only permute the outputs, not the inputs
            y_supports = [y_permutation[np.array(ys)] for ys in y_supports]
            targets = y_permutation[np.array(targets)]

        return (
            query_state,
            support_state,
            queries,
            targets,
            x_supports,
            y_supports,
        )


class ShuffleDemonstrationsDataset(Dataset):
    def __init__(self, dataset, active, seed=0):
        super().__init__()
        self.dataset = dataset
        self.active = active
        self.generator = np.random.default_rng(seed)

    def state_dict(self):
        return {"random_state": self.generator.__getstate__()}

    def load_state_dict(self, sd):
        if "random_state" in sd:
            self.generator.__setstate__(sd["random_state"])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if not self.active:
            return self.dataset[idx]

        (
            query_state,
            support_state,
            queries,
            targets,
            x_supports,
            y_supports,
        ) = self.dataset[idx]

        support_permutation = self.generator.permutation(len(x_supports))

        return (
            query_state,
            [support_state[i] for i in support_permutation]
            if isinstance(support_state, list)
            else support_state,
            queries,
            targets,
            [x_supports[i] for i in support_permutation]
            if isinstance(x_supports, list)
            else x_supports[support_permutation],
            [y_supports[i] for i in support_permutation]
            if isinstance(y_supports, list)
            else y_supports[support_permutation],
        )


def render_one_state(
    instruction, state, word2idx, world, vocabulary, colors, nouns, image_downsample
):
    words, situation = state_to_situation(
        instruction, state, word2idx, colors, nouns, need_target=False
    )

    reinitialize_world(world, situation, vocabulary)

    img = world.render(mode="rgb_array")[::image_downsample, ::image_downsample] / 255.0

    return img


class MetalearningImageRenderingDemonstrationsDataset(Dataset):
    def __init__(
        self, train_demonstrations, word2idx, colors, nouns, image_downsample=5
    ):
        super().__init__()
        self.train_demonstrations = train_demonstrations
        self.word2idx = word2idx
        self.colors = [None] + sorted(colors)
        self.nouns = [None] + sorted(nouns)

        vocabulary = create_vocabulary()
        world = create_world(vocabulary)

        self.world = world
        self.vocabulary = vocabulary
        self.image_downsample = image_downsample

    def __getitem__(self, i):
        (
            query_state,
            support_state,
            queries,
            targets,
            x_supports,
            y_supports,
        ) = self.train_demonstrations[i]

        query_state_img = render_one_state(
            queries,
            query_state,
            self.word2idx,
            self.world,
            self.vocabulary,
            self.colors,
            self.nouns,
            self.image_downsample,
        )
        support_state_imgs = np.stack(
            [
                render_one_state(
                    support_instr,
                    support_state_instance,
                    self.word2idx,
                    self.world,
                    self.vocabulary,
                    self.colors,
                    self.nouns,
                    self.image_downsample,
                )
                for support_instr, support_state_instance in zip(
                    x_supports, support_state
                )
            ]
        )

        return (
            query_state_img,
            support_state_imgs,
            queries,
            targets,
            x_supports,
            y_supports,
        )

    def __len__(self):
        return len(self.train_demonstrations)


class ModelEmaV2(nn.Module):
    """ Model Exponential Moving Average V2

    Keep a moving average of everything in the model state_dict (parameters and buffers).
    V2 of this module is simpler, it does not match params/buffers based on name but simply
    iterates in order. It works with torchscript (JIT of full model).

    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

    A smoothed version of the weights is necessary for some training schemes to perform well.
    E.g. Google's hyper-params for training MNASNet, MobileNet-V3, EfficientNet, etc that use
    RMSprop with a short 2.4-3 epoch decay period and slow LR decay rate of .96-.99 requires EMA
    smoothing of weights to match results. Pay attention to the decay constant you are using
    relative to your update count per epoch.

    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually in a separate
    process, or after the training stops converging.

    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """
    def __init__(self, model, decay=0.9999, device=None):
        super(ModelEmaV2, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


# Cell
class EMACallback(pl.callbacks.Callback):
    """
    Model Exponential Moving Average. Empirically it has been found that using the moving average
    of the trained parameters of a deep network is better than using its trained parameters directly.

    If `use_ema_weights`, then the ema parameters of the network is set after training end.
    """

    def __init__(self, decay=0.9999, use_ema_weights: bool = True):
        self.decay = decay
        self.ema = None
        self.use_ema_weights = use_ema_weights

    def on_fit_start(self, trainer, pl_module):
        "Initialize `ModelEmaV2` from timm to keep a copy of the moving average of the weights"
        self.ema = ModelEmaV2(pl_module, decay=self.decay, device=None)

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        "Update the stored parameters using a moving average"
        # Update currently maintained parameters.
        self.ema.update(pl_module)

    def on_validation_epoch_start(self, trainer, pl_module):
        "do validation using the stored parameters"
        # save original parameters before replacing with EMA version
        self.store(pl_module.parameters())

        # update the LightningModule with the EMA weights
        # ~ Copy EMA parameters to LightningModule
        self.copy_to(self.ema.module.parameters(), pl_module.parameters())

    def on_validation_end(self, trainer, pl_module):
        "Restore original parameters to resume training later"
        self.restore(pl_module.parameters())

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if self.ema is not None:
            return {"state_dict_ema": self.ema.state_dict()}

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        if self.ema is not None:
            self.ema.module.load_state_dict(checkpoint["state_dict_ema"])

    def store(self, parameters):
        "Save the current parameters for restoring later."
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

    def copy_to(self, shadow_parameters, parameters):
        "Copy current parameters into given collection of parameters."
        for s_param, param in zip(shadow_parameters, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def on_train_end(self, trainer, pl_module):
        # update the LightningModule with the EMA weights
        if self.use_ema_weights:
            self.copy_to(self.ema.module.parameters(), pl_module.parameters())



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
    parser.add_argument("--shuffle-demonstrations", action="store_true")
    parser.add_argument("--activation-checkpointing", action="store_true")
    parser.add_argument("--image-downsample", type=int, default=5)
    parser.add_argument("--patch-size", type=int, default=12)
    parser.add_argument("--max-context-size", type=int, default=1024)
    parser.add_argument("--need-support-states", action="store_true")
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=0.995)
    args = parser.parse_args()

    exp_name = "meta_gscan"
    model_name = f"meta_img_encdec_big_transformer_l_{args.nlayers}_h_{args.nhead}_d_{args.hidden_size}"
    dataset_name = args.dataset_name
    effective_batch_size = args.train_batch_size * args.batch_size_mult
    exp_name = f"{exp_name}_s_{args.seed}_m_{model_name}_it_{args.iterations}_b_{effective_batch_size}_d_{dataset_name}_t_{args.tag}_drop_{args.dropout_p}_ml_d_limit_{args.metalearn_demonstrations_limit}"
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

    torch.set_float32_matmul_precision("medium")
    print("Flash attention:", torch.backends.cuda.flash_sdp_enabled())

    seed = args.seed
    iterations = args.iterations

    (
        (
            WORD2IDX,
            ACTION2IDX,
            color_dictionary,
            noun_dictionary,
        ),
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
        PaddingDataset(
            MetalearningImageRenderingDemonstrationsDataset(
                PermuteActionsDataset(
                    ShuffleDemonstrationsDataset(
                        ReorderSupportsByDistanceDataset(
                            MapDataset(
                                MapDataset(
                                    meta_train_demonstrations,
                                    lambda x: (
                                        x[2],
                                        x[3],
                                        x[0],
                                        x[1],
                                        x[4],
                                        x[5],
                                        x[6],
                                    ),
                                ),
                                lambda x: (
                                    x[0],
                                    [x[1]] * len(x[-1])
                                    if not isinstance(x[1][0], list)
                                    else x[1],
                                    x[2],
                                    x[3],
                                    x[4],
                                    x[5],
                                    x[6],
                                ),
                            ),
                            args.metalearn_demonstrations_limit,
                        ),
                        args.shuffle_demonstrations,
                    ),
                    len(WORD2IDX),
                    len(ACTION2IDX),
                    pad_word,
                    pad_action,
                    shuffle=not args.disable_shuffle,
                ),
                WORD2IDX,
                color_dictionary,
                noun_dictionary,
            ),
            (
                None,
                (args.metalearn_demonstrations_limit, None, None, None),
                32,
                128,
                (args.metalearn_demonstrations_limit, 32),
                (args.metalearn_demonstrations_limit, 128),
            ),
            (None, 0.0, pad_word, pad_action, pad_word, pad_action),
        )
    )

    pl.seed_everything(seed)
    meta_module = BigImgTransformerLearner(
        3,
        args.patch_size,
        len(WORD2IDX),
        len(ACTION2IDX),
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
        max_context_size=args.max_context_size,
        need_support_states=args.need_support_states
    )
    print(meta_module)

    pl.seed_everything(0)
    train_dataloader = DataLoader(
        meta_train_dataset,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_ncpus,
        prefetch_factor=4,
    )

    check_val_opts = {}
    interval = args.check_val_every / len(train_dataloader)

    # Every check_val_interval steps, regardless of how large the training dataloader is
    if interval > 1.0:
        check_val_opts["check_val_every_n_epoch"] = math.floor(interval)
    else:
        check_val_opts["val_check_interval"] = interval

    logs_root_dir = f"{args.log_dir}/{exp_name}/{model_name}/{dataset_name}/{seed}"
    most_recent_version = args.version

    strategy_kwargs = (
        {
            "strategy": pl.strategies.FSDPStrategy(
                activation_checkpointing=(
                    [nn.TransformerEncoderLayer, nn.TransformerDecoderLayer]
                ),
            )
        }
        if args.activation_checkpointing
        else {}
    )

    callbacks = [
        pl.callbacks.LearningRateMonitor(),
        ModelCheckpoint(save_last=True, save_top_k=0, save_on_train_epoch_end=True),
    ]

    if args.ema:
        callbacks.append(EMACallback(decay=args.ema_decay))

    meta_trainer = pl.Trainer(
        logger=[
            TensorBoardLogger(
                logs_root_dir,
                version=most_recent_version,
            ),
            LoadableCSVLogger(
                logs_root_dir, version=most_recent_version, flush_logs_every_n_steps=100
            )
        ],
        callbacks=callbacks,
        max_steps=iterations,
        num_sanity_val_steps=1,
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed",
        default_root_dir=logs_root_dir,
        accumulate_grad_batches=args.batch_size_mult,
        enable_progress_bar=sys.stdout.isatty() or args.enable_progress,
        # gradient_clip_val=1.0,
        **strategy_kwargs,
        **check_val_opts,
    )

    valid_dataloaders = [
        DataLoader(
            PaddingDataset(
                MetalearningImageRenderingDemonstrationsDataset(
                    ReorderSupportsByDistanceDataset(
                        MapDataset(
                            MapDataset(
                                Subset(
                                    demonstrations,
                                    np.random.permutation(len(demonstrations))[
                                        : args.limit_val_size
                                    ],
                                ),
                                lambda x: (x[2], x[3], x[0], x[1], x[4], x[5], x[6]),
                            ),
                            lambda x: (
                                x[0],
                                [x[1]] * len(x[-1])
                                if not isinstance(x[1][0], list)
                                else x[1],
                                x[2],
                                x[3],
                                x[4],
                                x[5],
                                x[6],
                            ),
                        ),
                        args.metalearn_demonstrations_limit,
                    ),
                    WORD2IDX,
                    color_dictionary,
                    noun_dictionary,
                ),
                (
                    None,
                    (args.metalearn_demonstrations_limit, None, None, None),
                    32,
                    128,
                    (args.metalearn_demonstrations_limit, 32),
                    (args.metalearn_demonstrations_limit, 128),
                ),
                (None, 0.0, pad_word, pad_action, pad_word, pad_action),
            ),
            num_workers=2,
            batch_size=max([args.train_batch_size, args.valid_batch_size]),
            pin_memory=True,
        )
        for demonstrations in meta_valid_demonstrations_dict.values()
    ]

    meta_trainer.fit(meta_module, train_dataloader, valid_dataloaders, ckpt_path="last")
    print(f"Done, saving {model_path}")
    meta_trainer.save_checkpoint(f"{model_path}")


if __name__ == "__main__":
    main()
