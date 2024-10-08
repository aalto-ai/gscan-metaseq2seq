import argparse
import functools
import copy
import itertools
import os
import math
import torch
import torch.nn as nn
import numpy as np
import sys
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm.auto import tqdm


from gscan_metaseq2seq.util.dataset import PaddingDataset, ReshuffleOnIndexZeroDataset
from gscan_metaseq2seq.util.load_data import load_data_directories
from gscan_metaseq2seq.util.logging import LoadableCSVLogger
from gscan_metaseq2seq.models.enc_dec_transformer.enc_dec_transformer_model import (
    TransformerLearner,
)

from sentence_transformers import SentenceTransformer

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

    def __init__(self, decay=0.9999, use_ema_weights=True):
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


def determine_padding(demonstrations):
    max_instruction_len, max_action_len, max_state_len = (0, 0, 0)

    for instr, actions, state in tqdm(demonstrations, desc="Determining padding"):
        max_instruction_len = max(max_instruction_len, len(instr))
        max_action_len = max(max_action_len, len(actions))
        max_state_len = max(max_state_len, len(state))

    return max_instruction_len, max_action_len, max_state_len


def determine_state_profile(train_demonstrations, valid_demonstrations_dict):
    state_component_max_len = (functools.reduce(
        lambda x, o: np.stack([
            x, o
        ]).max(axis=0),
        map(lambda x: np.stack(x[-1]).max(axis=0),
            itertools.chain.from_iterable([
                train_demonstrations, *valid_demonstrations_dict.values()
            ]))
    ) + 1).tolist()
    state_feat_len = len(state_component_max_len)
    return state_component_max_len, state_feat_len


def debug_pred_inst(instruction, exacts, IDX2WORD, IDX2ACTION):
    WORD2IDX = {w: i for i, w in IDX2WORD.items()}
    ACTION2IDX = {w: i for i, w in IDX2ACTION.items()}
    print(WORD2IDX)
    print(ACTION2IDX)
    return [(" ".join([IDX2WORD[w] for w in s if w != WORD2IDX['[pad]']])) for s in instruction[~exacts].numpy()]



def debug_pred(instruction, target, decoded, exacts, state, IDX2WORD, IDX2ACTION, IDX2COLOR, IDX2OBJECT):
    WORD2IDX = {w: i for i, w in IDX2WORD.items()}
    ACTION2IDX = {w: i for i, w in IDX2ACTION.items()}
    print(WORD2IDX)
    print(ACTION2IDX)
    return [
        (
            "inst: " + " ".join([IDX2WORD[w] for w in s if w != WORD2IDX['[pad]']]),
            " - tgt: " + " ".join([IDX2ACTION[w] for w in t if w != ACTION2IDX['[pad]']]),
            " - pred: " + " ".join([IDX2ACTION[w] for w in d if w != ACTION2IDX['[pad]']]),
            " - state: " + "\n - ".join([
                f"{IDX2COLOR[q[1] - 1]} {IDX2OBJECT[q[2] - 1]} - {q[-2:]}"
                for q in state
            ])
         )
         for s, t, d, state in zip(instruction[~exacts].numpy(), target[~exacts].numpy(), decoded[~exacts].numpy(), state[~exacts].numpy())
    ]

def debug_pred_all(instruction, target, decoded, exacts, state, IDX2WORD, IDX2ACTION, IDX2COLOR, IDX2OBJECT):
    WORD2IDX = {w: i for i, w in IDX2WORD.items()}
    ACTION2IDX = {w: i for i, w in IDX2ACTION.items()}
    print(WORD2IDX)
    print(ACTION2IDX)
    return [
        (
            f"inst {e}: " + " ".join([IDX2WORD[w] for w in s if w != WORD2IDX['[pad]']]),
            " - tgt: " + " ".join([IDX2ACTION[w] for w in t if w != ACTION2IDX['[pad]']]),
            " - pred: " + " ".join([IDX2ACTION[w] for w in d if w != ACTION2IDX['[pad]']]),
            " - state: " + "\n - ".join([
                f"{IDX2COLOR[q[1] - 1]} {IDX2OBJECT[q[2] - 1]} - {q[-2:]}"
                for q in state
            ])
         )
         for s, t, d, state, e in zip(instruction.numpy(), target.numpy(), decoded.numpy(), state.numpy(), exacts)
    ]

def debug_pred_inv(instruction, target, decoded, exacts, state, IDX2WORD, IDX2ACTION, IDX2COLOR, IDX2OBJECT):
    WORD2IDX = {w: i for i, w in IDX2WORD.items()}
    ACTION2IDX = {w: i for i, w in IDX2ACTION.items()}
    print(WORD2IDX)
    print(ACTION2IDX)
    return [
        (
            " ".join([IDX2WORD[w] for w in s if w != WORD2IDX['[pad]']]),
            " ".join([IDX2ACTION[w] for w in t if w != ACTION2IDX['[pad]']]),
            " ".join([IDX2ACTION[w] for w in d if w != ACTION2IDX['[pad]']]),
            "\n - ".join([
                f"{IDX2COLOR[q[1] - 1]} {IDX2OBJECT[q[2] - 1]} - {q[-2:]}"
                for q in state
            ])
         )
         for s, t, d, state in zip(instruction[exacts].numpy(), target[exacts].numpy(), decoded[exacts].numpy(), state[exacts].numpy())
    ]



def debug_pred_spec(instruction, target, decoded, exacts, IDX2WORD, IDX2ACTION, index):
    WORD2IDX = {w: i for i, w in IDX2WORD.items()}
    ACTION2IDX = {w: i for i, w in IDX2ACTION.items()}
    print(WORD2IDX)
    print(ACTION2IDX)
    return [(" ".join([IDX2WORD[w] for w in s if w != WORD2IDX['[pad]']]), " ".join([IDX2ACTION[w] for w in t if w != ACTION2IDX['[pad]']]), " ".join([IDX2ACTION[w] for w in d if w != ACTION2IDX['[pad]']])) for s, t, d in zip(instruction[~exacts].numpy()[index][None], target[~exacts].numpy()[index][None], decoded[~exacts].numpy()[index][None])]


def retokenize_with_sbert(demonstrations, IDX2WORD):
    tokenizer = SentenceTransformer("all-MiniLM-L6-v2").tokenizer
    instructions, actions, states = list(zip(*demonstrations))
    retokenized_instructions = tokenizer.batch_encode_plus([
        " ".join([IDX2WORD[w] for w in instruction])
        for instruction in instructions
    ]).input_ids

    return list(zip(retokenized_instructions, actions, states))


class SentenceTransformerWrapper(nn.Module):
    def __init__(self, module, pad_token_id, output_dim):
        super().__init__()
        self.module = module
        self.output_transform = nn.Linear(384, output_dim)
        self.pad_token_id = pad_token_id

    def forward(self, x):
        with torch.no_grad():
            out = self.module({
                "input_ids": x,
                "attention_mask": x != self.pad_token_id
            })["token_embeddings"]

        return self.output_transform(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-demonstrations", type=str, required=True)
    parser.add_argument("--valid-demonstrations-directory", type=str, required=True)
    parser.add_argument("--dictionary", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--valid-batch-size", type=int, default=128)
    parser.add_argument("--batch-size-mult", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--nlayers", type=int, default=8)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--dropout-p", type=float, default=0.1)
    parser.add_argument("--norm-first", action="store_true")
    parser.add_argument("--precision", type=int, choices=(16, 32), default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-2)
    parser.add_argument("--warmup-proportion", type=float, default=0.1)
    parser.add_argument("--decay-power", type=int, default=-1)
    parser.add_argument("--iterations", type=int, default=300000)
    parser.add_argument("--check-val-every", type=int, default=1000)
    parser.add_argument("--limit-val-size", type=int, default=None)
    parser.add_argument("--enable-progress", action="store_true")
    parser.add_argument("--restore-from-checkpoint", action="store_true")
    parser.add_argument("--version", type=int, default=None)
    parser.add_argument("--tag", type=str, default="none")
    parser.add_argument("--dataset-name", type=str, default="gscan")
    parser.add_argument("--pad-instructions-to", type=int, default=32)
    parser.add_argument("--pad-actions-to", type=int, default=128)
    parser.add_argument("--pad-state-to", type=int, default=36)
    parser.add_argument("--determine-padding", action="store_true")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--dataloader-ncpus", type=int, default=1)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--limit-load", type=int)
    parser.add_argument(
        "--state-profile", choices=("gscan", "reascan", "state-calflow", "babyai-codeworld", "messenger"), default="gscan"
    )
    parser.add_argument(
        "--determine-state-profile", action="store_true"
    )
    parser.add_argument(
        "--use-state-component-lengths",
        action="store_true",
        help="Use state component lengths (here for backward compatibility)"
    )
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--use-sbert-embeddings", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    exp_name = "gscan"
    model_name = f"transformer_encoder_only_decode_actions_l_{args.nlayers}_h_{args.nhead}_d_{args.hidden_size}"
    dataset_name = args.dataset_name
    effective_batch_size = args.train_batch_size * args.batch_size_mult
    exp_name = f"{exp_name}_s_{args.seed}_m_{model_name}_it_{args.iterations}_b_{effective_batch_size}_d_{dataset_name}_t_{args.tag}_drop_{args.dropout_p}"
    model_dir = f"{args.log_dir}/models/{exp_name}/{model_name}"
    model_path = f"{model_dir}/{exp_name}.pt"
    print(model_path)
    print(
        f"Batch size {args.train_batch_size}, mult {args.batch_size_mult}, total {args.train_batch_size * args.batch_size_mult}"
    )

    torch.set_float32_matmul_precision("medium")
    print("Flash attention:", torch.backends.cuda.flash_sdp_enabled())

    os.makedirs(model_dir, exist_ok=True)

    seed = args.seed
    iterations = args.iterations

    (
        dictionaries,
        (train_demonstrations, valid_demonstrations_dict),
    ) = load_data_directories(args.train_demonstrations, args.dictionary, limit_load=args.limit_load)

    WORD2IDX = dictionaries[0]
    ACTION2IDX = dictionaries[1]

    IDX2WORD = {i: w for w, i in WORD2IDX.items()}
    IDX2ACTION = {i: w for w, i in ACTION2IDX.items()}

    pad_word = WORD2IDX["[pad]"]
    pad_action = ACTION2IDX["[pad]"]
    sos_action = ACTION2IDX["[sos]"]
    eos_action = ACTION2IDX["[eos]"]

    STATE_PROFILES = {
        "gscan": [4, len(dictionaries[2]), len(dictionaries[3]), 1, 4, 8, 8],
        "babyai-codeworld": [16, len(dictionaries[2]), len(dictionaries[3]), 4, 4, 64, 64],
        "reascan": [
            4,
            len(dictionaries[2]),
            len(dictionaries[3]),
            1,
            4,
            8,
            8,
            4,
            len(dictionaries[3]),
            1,
        ],
        "messenger": [
            len(dictionaries[2]),
            32,
            32
        ],
        "state-calflow": [
            8, # token type
            # These are upper bounds, not exact lengths
            512, # possible name
            512, # possible event or time
            512, # possible event, or time
            64 # number
        ]
    }
    if args.determine_state_profile:
        state_component_max_len, state_feat_len = determine_state_profile(
            train_demonstrations,
            valid_demonstrations_dict
        )
    else:
        state_component_max_len = STATE_PROFILES[args.state_profile]
        state_feat_len = len(state_component_max_len)

    print("State component lengths", state_component_max_len)

    pad_instructions_to, pad_actions_to, pad_state_to = (
        args.pad_instructions_to,
        args.pad_actions_to,
        args.pad_state_to
    )

    if args.use_sbert_embeddings:
        train_demonstrations = retokenize_with_sbert(train_demonstrations, IDX2WORD)
        valid_demonstrations_dict = {
            k: retokenize_with_sbert(v, IDX2WORD)
            for k, v in valid_demonstrations_dict.items()
        }
        pad_word = SentenceTransformer("all-MiniLM-L6-v2").tokenizer.pad_token_id

    if args.determine_padding:
        pad_instructions_to, pad_actions_to, pad_state_to = determine_padding(
            itertools.chain.from_iterable([
                train_demonstrations, *valid_demonstrations_dict.values()
            ])
        )

    print(f"Paddings instr: {pad_instructions_to} ({pad_word}) act: {pad_actions_to} ({pad_action}) state: {pad_state_to} (0)")

    pl.seed_everything(0)
    train_dataset = ReshuffleOnIndexZeroDataset(
        PaddingDataset(
            train_demonstrations,
            (
                pad_instructions_to,
                pad_actions_to,
                (pad_state_to, state_feat_len),
            ),
            (pad_word, pad_action, 0),
        )
    )

    pl.seed_everything(seed)
    meta_module = TransformerLearner(
        state_feat_len,
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
        lr=args.lr,
        norm_first=args.norm_first,
        decay_power=args.decay_power,
        warmup_proportion=args.warmup_proportion,
        state_component_lengths=(
            state_component_max_len
            if args.use_state_component_lengths else None
        ),
        custom_embedding_model=(
            SentenceTransformerWrapper(
                SentenceTransformer("all-MiniLM-L6-v2"),
                pad_word,
                args.hidden_size
            )
            if args.use_sbert_embeddings else None
        )
    )
    print(meta_module)

    pl.seed_everything(0)
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, pin_memory=True
    )

    check_val_opts = {}
    interval = args.check_val_every / len(train_dataloader)

    # Every check_val_interval steps, regardless of how large the training dataloader is
    if interval > 1.0:
        check_val_opts["check_val_every_n_epoch"] = math.floor(interval)
    else:
        check_val_opts["val_check_interval"] = interval

    checkpoint_cb = ModelCheckpoint(save_last=True, save_top_k=0)

    logs_root_dir = f"{args.log_dir}/{exp_name}/{model_name}/{dataset_name}/{seed}"
    most_recent_version = args.version

    callbacks = [
        pl.callbacks.LearningRateMonitor(),
        checkpoint_cb,
    ]

    if args.ema:
        callbacks.append(EMACallback(decay=args.ema_decay))

    trainer = pl.Trainer(
        logger=[
            TensorBoardLogger(logs_root_dir, version=most_recent_version),
            LoadableCSVLogger(
                logs_root_dir, version=most_recent_version, flush_logs_every_n_steps=10
            ),
        ],
        callbacks=callbacks,
        max_steps=iterations,
        num_sanity_val_steps=10,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision=(
            "bf16-mixed" if (
                args.precision == 16 and
                torch.cuda.is_bf16_supported()
            ) else (
                "16-mixed" if args.precision == 16 and torch.cuda.is_available()
                else (
                    "32"
                )
            )
        ),
        default_root_dir=logs_root_dir,
        accumulate_grad_batches=args.batch_size_mult,
        enable_progress_bar=sys.stdout.isatty() or args.enable_progress,
        gradient_clip_val=0.2,
        **check_val_opts,
    )

    print(valid_demonstrations_dict.keys())
    valid_dataloaders = [
        DataLoader(
            PaddingDataset(
                Subset(
                    demonstrations,
                    np.arange(len(demonstrations))[
                        : args.limit_val_size
                    ],
                ),
                (
                    pad_instructions_to,
                    pad_actions_to,
                    (pad_state_to, state_feat_len),
                ),
                (pad_word, pad_action, 0),
            ),
            batch_size=max([args.train_batch_size, args.valid_batch_size]),
            pin_memory=True,
        )
        for demonstrations in valid_demonstrations_dict.values()
    ]

    if args.debug:
        preds = trainer.predict(
            meta_module,
            valid_dataloaders[0],
            ckpt_path=model_path if os.path.exists(model_path) else "last"
        )
        instruction, state, decoded, logits, exacts, target = preds[0]
        import pprint
        dbg = debug_pred_all(instruction, target, decoded, exacts, state, IDX2WORD, IDX2ACTION, dictionaries[2], dictionaries[3])
        pprint.pprint(dbg)

        import pdb
        pdb.set_trace()

    if not os.path.exists(f"{model_path}"):
        trainer.fit(
            meta_module,
            train_dataloader,
            valid_dataloaders,
            ckpt_path="last",
        )
        trainer.save_checkpoint(f"{model_path}")
        print(f"Done, saving {model_path}")
        return

    print(f"Skipping {model_path} as it already exists")
    trainer.validate(
        meta_module, valid_dataloaders, ckpt_path=model_path
    )


if __name__ == "__main__":
    main()
