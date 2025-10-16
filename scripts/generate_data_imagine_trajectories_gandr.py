import argparse
import itertools
import os
from collections import defaultdict
import math
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl

import minigrid

import faiss
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize
from pytorch_lightning.loggers import TensorBoardLogger
from positional_encodings.torch_encodings import PositionalEncoding1D

from gscan_metaseq2seq.models.embedding import BOWEmbedding
from gscan_metaseq2seq.util.dataset import (
    PaddingDataset,
    MapDataset,
)
from gscan_metaseq2seq.util.load_data import (
    load_data_directories,
    load_concat_pickle_files_from_directory,
)
from gscan_metaseq2seq.util.logging import LoadableCSVLogger
from gscan_metaseq2seq.util.scheduler import transformer_optimizer_config
from gscan_metaseq2seq.models.enc_dec_transformer.enc_dec_transformer_model import (
    TransformerLearner,
    autoregressive_model_unroll_predictions,
)
from generate_data_retrieval import (
    compute_sorted_set_bsr,
    compute_sorted_bsr,
    vectorize_all_example_situations
)
from train_transformer import determine_padding, determine_state_profile
from sentence_transformers import SentenceTransformer
from generate_data_imagine_trajectories import compute_resume_points

from tqdm.auto import tqdm, trange


def batched(iterable, n):
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while True:
        batch = list(itertools.islice(it, n))

        if not batch:
            break

        yield batch


def train_transformer(
    weights_path,
    dataset,
    validation_datasets,
    seed,
    transformer_train_iterations,
    batch_size,
    state_feat_len,
    x_categories,
    y_categories,
    pad_word_idx,
    pad_action_idx,
    sos_action_idx,
    eos_action_idx,
    hidden_size=128,
    nlayers=8,
    nhead=8,
    device="cpu",
):
    dropout_p = 0.0
    train_batch_size = batch_size
    batch_size_mult = 1
    dataset_name = "gscan"
    check_val_every = 8000

    exp_name = "transformer"
    model_name = f"transformer_l_{nlayers}_h_{nhead}_d_{hidden_size}"
    dataset_name = dataset_name
    effective_batch_size = train_batch_size * batch_size_mult
    exp_name = f"{exp_name}_s_{seed}_m_{model_name}_it_{transformer_train_iterations}_b_{effective_batch_size}_d_{dataset_name}_drop_{dropout_p}"
    model_dir = f"models/{exp_name}/{model_name}"
    model_path = f"{model_dir}/{exp_name}.pt"
    print(model_path)
    print(
        f"Batch size {train_batch_size}, mult {batch_size_mult}, total {train_batch_size * batch_size_mult}"
    )

    pl.seed_everything(seed)
    train_dataloader = DataLoader(
        dataset, batch_size=train_batch_size, pin_memory=True, shuffle=True
    )
    validation_dataloaders = [
        DataLoader(
            ds,
            batch_size=train_batch_size,
            pin_memory=True,
        )
        for ds in validation_datasets
    ]

    logs_root_dir = f"logs/{exp_name}/{model_name}/{dataset_name}/{seed}"

    transformer_model_weights = torch.load(weights_path)
    transformer_model_state_dict = transformer_model_weights["state_dict"] if "state_dict" in transformer_model_weights else transformer_model_weights
    loaded_hparams = transformer_model_weights["hyper_parameters"] if "hyper_parameters" in transformer_model_weights else None
    print(state_feat_len)
    print(loaded_hparams)
    transformer_model_hparams = loaded_hparams or dict(
        n_state_components=state_feat_len,
        x_categories=x_categories,
        y_categories=y_categories,
        embed_dim=512,
        dropout_p=0.0,
        nlayers=12,
        nhead=8,
        pad_word_idx=pad_word_idx,
        pad_action_idx=pad_action_idx,
        sos_action_idx=sos_action_idx,
        eos_action_idx=eos_action_idx,
        wd=1e-2,
        lr=1e-4,
        decay_power=-1,
        warmup_proportion=0.1,
    )
    loaded_hparams["n_state_components"] = state_feat_len

    transformer_model = TransformerLearner(
        **transformer_model_hparams
    )
    transformer_model.load_state_dict(transformer_model_state_dict)
    print(transformer_model)

    pl.seed_everything(seed)
    trainer = pl.Trainer(
        logger=[
            TensorBoardLogger(logs_root_dir),
            LoadableCSVLogger(logs_root_dir, flush_logs_every_n_steps=10),
        ],
        callbacks=[pl.callbacks.LearningRateMonitor()],
        max_steps=transformer_train_iterations,
        num_sanity_val_steps=10,
        accelerator="gpu" if torch.cuda.is_available() else 0,
        devices=1 if torch.cuda.is_available() else 0,
        precision="16-mixed" if device == "cuda" else 32,
        default_root_dir=logs_root_dir,
        accumulate_grad_batches=batch_size_mult,
        gradient_clip_val=0.2,
    )

    trainer.fit(transformer_model, train_dataloader, validation_dataloaders)

    return transformer_model


def to_count_matrix(action_word_arrays, action_vocab_size):
    count_matrix = np.zeros(
        (len(action_word_arrays), action_vocab_size)
    )

    for i, action_array in enumerate(action_word_arrays):
        for element in action_array:
            count_matrix[i, element] += 1

    return count_matrix


def to_tfidf(tfidf_transformer, count_matrix):
    tfidf_array = np.array(tfidf_transformer.transform(count_matrix).todense()).astype("float32")

    return tfidf_array


def transformer_predict(transformer_learner, state, instruction, decode_len):
    state = state.to(transformer_learner.device)
    instruction = instruction.to(transformer_learner.device)
    dummy_targets = torch.zeros(
        instruction.shape[0],
        decode_len,
        dtype=torch.long,
        device=transformer_learner.device,
    )

    decoded, logits, exacts, _ = autoregressive_model_unroll_predictions(
        transformer_learner,
        (state, instruction),
        dummy_targets,
        transformer_learner.sos_action_idx,
        transformer_learner.eos_action_idx,
        transformer_learner.pad_action_idx,
        quiet=True
    )

    return decoded, logits


def balance_dims(first, *rest, factors=None):
    first_abs_sum = np.linalg.norm(first, axis=-1)
    rest_abs_sums = [
        np.linalg.norm(r, axis=-1) for r in rest
    ]
    rest_abs_ratios = [
        first_abs_sum / rs for rs in rest_abs_sums
    ]

    return np.concatenate([
        first
    ] + [
        r * ra[..., None] * factor
        for r, ra, factor in zip(rest, rest_abs_ratios, factors or ([1] * len(rest_abs_ratios)))
    ], axis=-1)


def gandr_like_search(
    transformer_prediction_model,
    index,
    train_dataset,
    train_sentences_unique_list_lookup,
    train_sentences_unique_all_token_encodings,
    tfidf_transformer,
    state_pca,
    component_max_sizes,
    sentence_transformer,
    dataloader,
    sample_n,
    decode_len,
    pad_word_idx,
    pad_action_idx,
    IDX2WORD,
    word_vocab_size,
    action_vocab_size,
    device="cpu",
):
    transformer_prediction_model.to(device)
    transformer_prediction_model.eval()

    for batch in dataloader:
        instruction, targets, state = batch

        # We need to vectorize all example situations here
        # as well, but the dataset is padded, so we have to un-pad it somehow?
        state_encodings = state_pca.transform(vectorize_all_example_situations([
            s[~(s == 0).all()].numpy() for s in state
        ], component_max_sizes).astype(np.float32))

        predicted_targets, logits = transformer_predict(
            transformer_prediction_model,
            state,
            instruction,
            decode_len,
        )
        instruction_sentences = [
            " ".join([IDX2WORD[w] for w in inst[inst != pad_word_idx].cpu().numpy()])
            for inst in instruction
        ]
        sentence_vectors = sentence_transformer.encode(
            instruction_sentences,
            normalize_embeddings=True
        )
        token_encodings = list(map(lambda x: x.cpu().numpy(), sentence_transformer.encode(
            instruction_sentences,
            output_value='token_embeddings',
            normalize_embeddings=True
        )))
        token_encodings = [
            v / (np.linalg.norm(v, axis=-1)[:, None] + 1e-7)
            for v in token_encodings
        ]
        count_matrix = to_count_matrix(
            [
                t[t != pad_action_idx]
                for t in predicted_targets
            ],
            action_vocab_size,
        )
        actions_count_matrix_tfidf = to_tfidf(
            tfidf_transformer,
            count_matrix
        )
        non_action_component = (
            balance_dims(
                state_encodings,
                sentence_vectors,
                factors=[(4 / 3)]
             )
        )
        unscaled_vectors = (
            balance_dims(
                actions_count_matrix_tfidf,
                non_action_component,
                factors=[(1 / 4)]
            )
        )
        scaled_vectors = normalize(unscaled_vectors, axis=1)

        near_neighbour_distances_batch, near_neighbour_indices_batch = index.search(
            scaled_vectors,
            sample_n,
        )

        sorted_near_neighbour_indices_by_coverage = [
            compute_sorted_bsr(
                query_token_embeds,
                [
                    train_sentences_unique_all_token_encodings[x]
                    for x in train_sentences_unique_list_lookup[query_near_neighbour_indices]
                ],
                16
            )[:16]
            for query_token_embeds, query_near_neighbour_indices in zip(
                token_encodings,
                near_neighbour_indices_batch
            )
        ]

        near_neighbour_supports_batch = [
            (
                [train_dataset[i] for i in near_neighbour_indices[np.array(retrieval_indices)]],
                near_neighbour_distances[np.array(retrieval_indices)],
            )
            for near_neighbour_indices, near_neighbour_distances, retrieval_indices in zip(
                near_neighbour_indices_batch,
                near_neighbour_distances_batch,
                sorted_near_neighbour_indices_by_coverage
            )
        ]

        for i, (supports, distances) in enumerate(near_neighbour_supports_batch):
            if True:
                print("Command:", " ".join([
                    IDX2WORD[w] for w in instruction[i].numpy()
                    if w != pad_word_idx
                ]))
                print("Supports:", "\n".join([
                    (
                        " ".join([IDX2WORD[w] for w in si if w != pad_word_idx]) +
                        " " +
                        " ".join([str(w) for w in si if w != pad_action_idx])
                    )
                    for si, sa in zip([s[-3] for s in supports], [s[-2] for s in supports])
                ]))
            yield (
                instruction[i].numpy(),
                targets[i].numpy(),
                state[i].numpy(),
                [s[-1] for s in supports],  # support state
                [s[-3] for s in supports],  # support instruction
                [s[-2] for s in supports],  # support actions
                distances,
            )


class StateCLIPTransformer(pl.LightningModule):
    def __init__(
        self,
        lr=0.0001,
        wd=1e-2,
        emb_dim=128,
        nlayers=8,
        nhead=4,
        dropout=0.1,
        norm_first=False,
        decay_power=-1,
        warmup_proportion=0.14,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.pos_encoding = PositionalEncoding1D(emb_dim)
        self.embedding = BOWEmbedding(16, 7, emb_dim)
        self.projection = nn.Linear(emb_dim * 7, emb_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=emb_dim,
                dim_feedforward=emb_dim * 4,
                dropout=dropout,
                nhead=nhead,
                norm_first=norm_first,
            ),
            num_layers=nlayers,
        )
        self.latent = nn.Parameter(torch.randn(emb_dim))
        self.project = nn.Linear(emb_dim, 16 * 7)

    def configure_optimizers(self):
        return transformer_optimizer_config(
            self,
            self.hparams.lr,
            weight_decay=self.hparams.wd,
            decay_power=self.hparams.decay_power,
            warmup_proportion=self.hparams.warmup_proportion,
        )

    def encode_to_vector(self, state):
        mask = (state == 0).all(dim=-1)
        embedded_state = self.projection(self.embedding(state))
        latent_seq = self.latent[None, None].expand(embedded_state.shape[0], 1, -1)

        encoded_state_with_latent = self.transformer_encoder(
            torch.cat([embedded_state, latent_seq], dim=1).transpose(1, 0),
            src_key_padding_mask=torch.cat(
                [mask, torch.zeros_like(mask[:, :1])], dim=1
            ),
        ).transpose(0, 1)

        return encoded_state_with_latent[:, -1]

    def forward(self, state):
        mask = (state == 0).all(dim=-1)
        embedded_state = self.projection(self.embedding(state))
        latent_seq = self.latent[None, None].expand(embedded_state.shape[0], 1, -1)

        orig_key_padding_mask = torch.cat([mask, torch.zeros_like(mask[:, :1])], dim=1)
        dropout_key_padding_mask = (
            torch.rand_like(orig_key_padding_mask.float()) < 0.3
        ) * orig_key_padding_mask

        orig_encoded_state_with_latent = self.transformer_encoder(
            torch.cat([embedded_state, latent_seq], dim=1).transpose(1, 0),
            src_key_padding_mask=torch.cat(
                [mask, torch.zeros_like(mask[:, :1])], dim=1
            ),
        ).transpose(0, 1)

        dropout_encoded_state_with_latent = self.transformer_encoder(
            torch.cat([embedded_state, latent_seq], dim=1).transpose(1, 0),
            src_key_padding_mask=dropout_key_padding_mask,
        ).transpose(0, 1)

        return orig_encoded_state_with_latent[
            :, -1
        ] @ dropout_encoded_state_with_latent[:, -1].transpose(0, 1)

    def training_step(self, x, idx):
        (state,) = x

        # Split into the seven components, then flatten
        # into one big sequence
        outer_product = self.forward(state)

        label = torch.arange(state.shape[0], device=state.device, dtype=torch.long)
        loss = (
            F.cross_entropy(outer_product, label)
            + F.cross_entropy(outer_product.transpose(1, 0), label)
        ) / 2

        self.log("tloss", loss)

        return loss


def train_state_autoencoder(
    weights_path,
    dataset,
    seed,
    mlm_train_iterations,
    batch_size,
    hidden_size=128,
    nlayers=4,
    nhead=8,
    device="cpu",
):
    dropout_p = 0.1
    train_batch_size = batch_size
    batch_size_mult = 1
    dataset_name = "gscan"
    check_val_every = 8000

    exp_name = "state_autoencoder"
    model_name = f"transformer_l_{nlayers}_h_{nhead}_d_{hidden_size}"
    dataset_name = dataset_name
    effective_batch_size = train_batch_size * batch_size_mult
    exp_name = f"{exp_name}_s_{seed}_m_{model_name}_it_{mlm_train_iterations}_b_{effective_batch_size}_d_{dataset_name}_drop_{dropout_p}"
    model_dir = f"models/{exp_name}/{model_name}"
    model_path = f"{model_dir}/{exp_name}.pt"
    print(model_path)
    print(
        f"Batch size {train_batch_size}, mult {batch_size_mult}, total {train_batch_size * batch_size_mult}"
    )

    train_dataloader = DataLoader(
        dataset, batch_size=train_batch_size, pin_memory=True, shuffle=True
    )

    logs_root_dir = f"logs/{exp_name}/{model_name}/{dataset_name}/{seed}"

    model = StateCLIPTransformer(
        nlayers=nlayers,
        nhead=nhead,
        emb_dim=hidden_size,
        dropout=dropout_p,
        norm_first=True,
        lr=1e-4,
        decay_power=-1,
        warmup_proportion=0.1,
    )
    print(model)

    if weights_path:
        model.load_state_dict(torch.load(weights_path))

    trainer = pl.Trainer(
        logger=[
            TensorBoardLogger(logs_root_dir),
            LoadableCSVLogger(logs_root_dir, flush_logs_every_n_steps=10),
        ],
        callbacks=[pl.callbacks.LearningRateMonitor()],
        max_steps=mlm_train_iterations,
        num_sanity_val_steps=10,
        gpus=1 if device == "cuda" else 0,
        precision=16 if device == "cuda" else 32,
        default_root_dir=logs_root_dir,
        accumulate_grad_batches=batch_size_mult,
        gradient_clip_val=0.2,
    )

    trainer.fit(model, train_dataloader)

    return model


def load_state_encodings(directory):
    return {
        k: load_concat_pickle_files_from_directory(os.path.join(directory, k))
        for k in filter(
            lambda x: os.path.isdir(os.path.join(directory, x)), os.listdir(directory)
        )
    }


def save_state_encodings(state_encodings_dict, directory):
    for key, encodings in state_encodings_dict.items():
        path = os.path.join(directory, key)
        os.makedirs(path, exist_ok=True)

        for i, batch in enumerate(batched(encodings, 10000)):
            with open(os.path.join(path, f"{i}.pb"), "wb") as f:
                pickle.dump(np.stack(batch), f)


def generate_state_encodings(
    state_autoencoder_transformer,
    train_demonstrations,
    valid_demonstrations_dict,
    batch_size,
    device="cpu",
):
    dataloaders = {
        split: DataLoader(
            PaddingDataset(
                MapDataset(demos, lambda x: (x[-1],)),
                ((36, 7),),
                (0,),
            ),
            batch_size=batch_size,
            pin_memory=True,
        )
        for split, demos in zip(
            itertools.chain.from_iterable(
                [valid_demonstrations_dict.keys(), ["train"]]
            ),
            itertools.chain.from_iterable(
                [valid_demonstrations_dict.values(), [train_demonstrations]]
            ),
        )
    }

    state_autoencoder_transformer.to(device)
    state_autoencoder_transformer.eval()

    with torch.inference_mode():
        encoded_states = {
            split: np.concatenate(
                list(
                    map(
                        lambda x: state_autoencoder_transformer.encode_to_vector(
                            x[0].long().to(device)
                        )
                        .detach()
                        .cpu()
                        .numpy(),
                        tqdm(dl),
                    )
                )
            )
            for split, dl in tqdm(dataloaders.items())
        }

    return encoded_states


def premultiply_balance_dimensions(vectors, dim_other):
    return vectors * (dim_other / vectors.shape[-1]) * (1 / 8)


def shift_bit_length(x):
    return 1 << ((x - 1).bit_length() - 1)


def lower_pow2(num):
    return shift_bit_length(num)


def compute_state_pca(train_demonstrations, state_component_max_len, pca_dim):
    train_state_vectors = vectorize_all_example_situations(
        tqdm(train_demonstrations, desc="Vectorizing examples"),
        state_component_max_len
    )

    train_state_vectors_fit_perm = np.random.permutation(train_state_vectors.shape[0])[:8192]
    selected_train_state_vectors = train_state_vectors[train_state_vectors_fit_perm]

    n_components = min(
        lower_pow2(selected_train_state_vectors.shape[-1]),
        pca_dim
    )
    print(f"Fitting PCA {(selected_train_state_vectors.shape[-1], n_components)}")
    state_pca = make_pipeline(
        StandardScaler(),
        PCA(n_components=n_components)
    )
    state_pca.fit(selected_train_state_vectors)

    print(f"Applying PCA to train state vectors")
    pca_train_state_vectors = np.concatenate([
        state_pca.transform(train_state_vectors[i * 1024:(i + 1) * 1024].astype(np.float32))
        for i in trange(train_state_vectors.shape[0] // 1024 + 1, desc="Applying PCA")
    ], axis=0)

    # Sanity check, how well do we reconstruct the original layouts
    print(f"Sanity check: computing reconstruction error")
    reconstruction_error_sample = np.random.permutation(pca_train_state_vectors.shape[0])[:8192]
    reconstruction_error = ((
        state_pca.inverse_transform(pca_train_state_vectors[reconstruction_error_sample]) -
        train_state_vectors[reconstruction_error_sample].astype(np.float32)
    ) ** 2).mean()

    print(f"PCA reconstruction error {reconstruction_error}")

    return state_pca, n_components


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-data", type=str, required=True)
    parser.add_argument("--dictionary", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-transformer-model", type=str)
    parser.add_argument("--load-transformer-model", type=str)
    parser.add_argument("--transformer-iterations", type=int, default=150000)
    parser.add_argument(
        "--state-autoencoder-transformer-iterations", type=int, default=50000
    )
    parser.add_argument("--load-state-autoencoder-transformer", type=str)
    parser.add_argument("--save-state-autoencoder-transformer", type=str)
    parser.add_argument("--load-state-encodings", type=str)
    parser.add_argument("--save-state-encodings", type=str)
    parser.add_argument("--data-output-directory", type=str, required=True)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--only-splits", nargs="*", help="Which splits to include")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--offset", type=float, default=0)
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument("--limit-load", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--decode-to", type=int, default=256)
    parser.add_argument("--gen-sample-n", type=int, default=256)
    parser.add_argument("--include-state", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retrieval-sentence-state-tradeoff", type=float, default=(4 / 3))
    parser.add_argument("--retrieval-state-pca-dim", type=int, default=1024)
    args = parser.parse_args()

    (
        dictionaries,
        (train_demonstrations, valid_demonstrations_dict),
    ) = load_data_directories(
        args.training_data,
        args.dictionary,
        only_splits=list(set(["train"]) | set(args.only_splits)) if args.only_splits else None,
        limit_load=args.limit_load
    )
    resume_points = compute_resume_points(args.data_output_directory) if args.resume else {}
    datasets_with_resume_points = {
        k: (v, resume_points.get(k, 0)) for k, v in {
            "train": train_demonstrations,
            **valid_demonstrations_dict
        }.items()
    }

    WORD2IDX = dictionaries[0]
    ACTION2IDX = dictionaries[1]

    print(WORD2IDX, ACTION2IDX)

    pad_action = ACTION2IDX["[pad]"]
    pad_word = WORD2IDX["[pad]"]

    IDX2WORD = {i: w for w, i in WORD2IDX.items()}

    os.makedirs(os.path.join(args.data_output_directory), exist_ok=True)

    state_component_max_len, state_feat_len = determine_state_profile(train_demonstrations, {
        k: v
        for k, v in valid_demonstrations_dict.items()
    })
    pad_instructions_to, pad_actions_to, pad_state_to = determine_padding(train_demonstrations)

    print(f"Paddings instr: {pad_instructions_to} act: {pad_actions_to} state: {pad_state_to}")

    state_pca, state_pca_n_components = compute_state_pca(
        train_demonstrations,
        state_component_max_len,
        args.retrieval_state_pca_dim
    )

    torch.set_float32_matmul_precision("medium")

    transformer_validation_datasets = [
        Subset(
            PaddingDataset(data, (pad_instructions_to, pad_actions_to, (pad_state_to, state_feat_len)), (pad_word, pad_action, 0)),
            np.random.permutation(min(512, len(data))),
        )
        for data in valid_demonstrations_dict.values()
    ]

    transformer_model = train_transformer(
        args.load_transformer_model,
        PaddingDataset(
            train_demonstrations,
            (pad_instructions_to, pad_actions_to, (pad_state_to, state_feat_len)),
            (
                pad_word,
                pad_action,
                0,
            ),
        ),
        transformer_validation_datasets,
        args.seed,
        args.transformer_iterations if args.save_transformer_model else 0,
        args.batch_size,
        state_feat_len,
        len(WORD2IDX),
        len(ACTION2IDX),
        pad_word,
        pad_action,
        ACTION2IDX["[sos]"],
        ACTION2IDX["[eos]"],
        hidden_size=args.hidden_size,
        nlayers=args.num_layers,
        nhead=args.nhead,
        device=args.device,
    )

    if args.save_transformer_model:
        torch.save(transformer_model.state_dict(), args.save_transformer_model)

    transformer_model_trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else 0,
        devices=1 if torch.cuda.is_available() else 0,
        precision=(
            "bf16" if (
                torch.cuda.is_bf16_supported()
            ) else (
                "16-mixed" if torch.cuda.is_available()
                else (
                    "32"
                )
            )
        ),
    )

    transformer_model_trainer.validate(
        transformer_model,
        [
            DataLoader(
                ds,
                batch_size=16,
                pin_memory=True,
            )
            for ds in transformer_validation_datasets
        ],
    )

    print(args.offset, args.offset + (0 if args.limit is None else args.limit))

    # Make a lookup table of train sentences to indices - this will allow
    # us to just encode the unique sentences and save some memory.
    train_sentences_index_dict = defaultdict(list)
    for i, example in enumerate(tqdm(train_demonstrations)):
        train_sentences_index_dict[" ".join([IDX2WORD[w] for w in example[0] if w != pad_word])].append(i)
    train_sentences_unique = sorted(list(train_sentences_index_dict.keys()))
    train_sentences_to_unique_index = {
        t: i for i, t in enumerate(train_sentences_unique)
    }
    train_sentences_unique_list_lookup = np.zeros(len(train_demonstrations), dtype=np.int32)
    for t, indices in train_sentences_index_dict.items():
        for i in indices:
            train_sentences_unique_list_lookup[i] = train_sentences_to_unique_index[t]

    model = SentenceTransformer('all-mpnet-base-v2')
    train_sentences_unique_all_token_encodings = list(map(lambda x: x.cpu().numpy(), model.encode(
        train_sentences_unique,
        output_value='token_embeddings',
        normalize_embeddings=True
    )))
    train_sentences_unique_all_token_encodings = [
        v / (np.linalg.norm(v, axis=-1)[:, None] + 1e-7)
        for v in train_sentences_unique_all_token_encodings
    ]
    train_sentences_unique_all_sentence_encodings = model.encode(
        train_sentences_unique,
        normalize_embeddings=True
    )

    # Make an index from the training data
    np.random.seed(args.seed)

    count_matrix = to_count_matrix(
        [
            actions
            for instruction, actions, state in train_demonstrations
        ],
        len(ACTION2IDX),
    )
    tfidf_transformer = TfidfTransformer()
    tfidf_transformer.fit(count_matrix)
    actions_count_matrix_tfidf = to_tfidf(
        tfidf_transformer,
        count_matrix
    )
    non_action_component = balance_dims(
        train_sentences_unique_all_sentence_encodings[train_sentences_unique_list_lookup],
        state_pca.transform(vectorize_all_example_situations(
            tqdm(train_demonstrations, desc="Vectorizing examples"),
            state_component_max_len
        ).astype(np.float32)),
        factors=[args.retrieval_sentence_state_tradeoff]
    )
    unscaled_vectors = (
        balance_dims(
            actions_count_matrix_tfidf,
            non_action_component,
            factors=[(1 / 4)]
        )
    )

    scaled_vectors = normalize(unscaled_vectors, axis=1)
    index = faiss.IndexFlatIP(scaled_vectors.shape[1])
    index.add(scaled_vectors)

    dataloader_splits = {
        split: DataLoader(
            PaddingDataset(
                Subset(
                    demos,
                    np.arange(
                        min(
                            math.floor(args.offset * len(demos)) + resume_point_offset,
                            len(demos),
                        ),
                        min(
                            math.floor(args.offset * len(demos))
                            + (
                                len(demos)
                                if not args.limit
                                else math.floor(args.limit * len(demos))
                            ),
                            len(demos),
                        ),
                    ),
                ),
                (pad_instructions_to, pad_actions_to, (pad_state_to, state_feat_len)),
                (pad_word, pad_action, 0),
            ),
            batch_size=args.batch_size,
            pin_memory=True,
        )
        for split, (demos, resume_point_offset) in datasets_with_resume_points.items()
        if not args.only_splits or split in args.only_splits
    }

    save_offset = (args.offset or 0) // 10000

    for split, dataloader in tqdm(dataloader_splits.items()):
        # Note we still make always make the directory,
        # this is to ensure that the dataloader indices align correctly
        os.makedirs(f"{args.data_output_directory}/{split}", exist_ok=True)

        if len(dataloader.dataset) == 0:
            print(f"Skip {split} as it is empty")
            continue

        for i, batch in enumerate(
            batched(
                gandr_like_search(
                    transformer_model,
                    index,
                    train_demonstrations,
                    train_sentences_unique_list_lookup,
                    train_sentences_unique_all_token_encodings,
                    tfidf_transformer,
                    state_pca,
                    state_component_max_len,
                    model,
                    tqdm(dataloader),
                    args.gen_sample_n,
                    decode_len=args.decode_to,
                    pad_word_idx=pad_word,
                    pad_action_idx=pad_action,
                    IDX2WORD=IDX2WORD,
                    word_vocab_size=len(WORD2IDX),
                    action_vocab_size=len(ACTION2IDX),
                    device=args.device,
                ),
                10000,
            )
        ):
            save_path = os.path.join(args.data_output_directory, split, f"{i + save_offset}.pb")
            with open(save_path, "wb") as f:
                print(f"Save {save_path}")
                pickle.dump(batch, f)


if __name__ == "__main__":
    main()
