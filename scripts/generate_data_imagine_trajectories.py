import argparse
import itertools
import operator
import os
import math
import sys
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, Subset
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from collections import Counter, defaultdict

from gscan_metaseq2seq.models.embedding import BOWEmbedding
from gscan_metaseq2seq.util.dataset import (
    PaddingDataset,
    ReshuffleOnIndexZeroDataset,
    MapDataset,
    SampleSentencesByWordWeights
)
from gscan_metaseq2seq.util.load_data import load_data_directories
from gscan_metaseq2seq.util.logging import LoadableCSVLogger
from gscan_metaseq2seq.util.scheduler import transformer_optimizer_config
from gscan_metaseq2seq.models.enc_dec_transformer.enc_dec_transformer_model import (
    SequenceTransformerLearner,
    TransformerLearner,
    autoregressive_model_unroll_predictions,
)
from gscan_metaseq2seq.models.instruction_gen.masked_language_model import (
    sample_from_mlm,
    train_mlm,
)
from gscan_metaseq2seq.models.instruction_gen.clip_ranking import (
    train_clip,
)
from generate_data_retrieval import compute_sorted_bsr
from train_transformer import determine_padding, determine_state_profile
from train_bart import (
    EncoderDecoderLanguageModel,
    StateEncoderDecoderLanguageModel,
    train_encoder_decoder,
    train_state_encoder_decoder
)

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


def make_gscan_instruction_gen_closure(
    instruction_gen_model, pad_target_idx, eos_tgt_idx, device="cpu", noise_level=0.2
):
    instruction_gen_model.to(device)
    instruction_gen_model.eval()

    def generate_instruction(inputs, sample_n):
        state, instruction = inputs
        (
            result_instrs,
            result_samples,
            result_samples_mask,
        ) = sample_from_state_encoder_decoder_model(
            instruction_gen_model,
            instruction,
            state,
            sample_n,
            pad_target_idx,
            eos_tgt_idx,
            noise_level=noise_level,
            device=device,
        )

        is_same_mask = (result_samples == instruction[:, None]).all(dim=-1)

        return [
            s[~m].reshape(-1, s.shape[-1]) for s, m in zip(result_samples, is_same_mask)
        ]

    return generate_instruction


def make_gscan_clip_ranking_closure(clip_ranking_model, pad_word_idx, device="cpu"):
    clip_ranking_model.to(device)
    clip_ranking_model.eval()

    clip_ranking_model.positional_encoding.cached_penc = None

    def compute_scores(instructions, inputs):
        states, query_instructions = inputs

        states = states.to(device)
        instructions = instructions.to(device)

        instruction_pad = instructions == pad_word_idx
        state_pad = torch.zeros_like(states[..., 0])
        state_pad = state_pad.to(torch.bool)

        with torch.inference_mode(), torch.autocast(
            device_type=device, dtype=torch.float16, enabled=True
        ):
            encoded_state = clip_ranking_model.state_encoder(states)
            projected_state = clip_ranking_model.state_encoder_projection(encoded_state)
            encoded_instruction = clip_ranking_model.embedding_instructions(
                instructions
            )
            encoded_instruction = (
                encoded_instruction
                + clip_ranking_model.positional_encoding(encoded_instruction)
            )

            decoded_instruction = clip_ranking_model.transformer_encoder_instruction(
                encoded_instruction, instruction_pad
            )
            decoded_state = clip_ranking_model.transformer_encoder_state(
                projected_state, state_pad
            )

            # Take the componentwise product
            scores = (decoded_instruction * decoded_state).sum(dim=-1)

            return scores

    return compute_scores


def make_gscan_seq2seq_ranking_closure(model, pad_word_idx, sos_word_idx, device="cpu"):
    model.eval()

    def compute_scores(instructions, inputs):
        states, query_instructions = inputs

        states = states.to(device)
        instructions = instructions.to(device)

        instruction_pad = instructions == pad_word_idx
        query_instruction_pad = query_instructions == pad_word_idx
        state_pad = (states == 0).all(dim=-1).bool()

        with torch.inference_mode(), torch.autocast(
            device_type=device, dtype=torch.float16, enabled=True
        ):
            logits = model(
                query_instructions,
                states,
                torch.cat([query_instruction_pad, state_pad], dim=-1),
                query_instruction_pad,
                torch.cat(
                    [
                        torch.ones_like(instructions[:, :1]) * sos_word_idx,
                        instructions[:, :-1],
                    ],
                    dim=-1,
                ),
            ).log_softmax(dim=-1)
            selected_logits = logits.gather(-1, instructions.unsqueeze(-1)).squeeze(-1)
            selected_logits[instruction_pad] = 0.0
            scores = selected_logits.sum(dim=-1) / (~instruction_pad).float().sum(
                dim=-1
            )

            return scores

    return compute_scores


def make_gscan_max_bsr_ranking_closure(model, pad_word_idx, sos_word_idx, device="cpu"):
    model.eval()

    def compute_scores(instructions, inputs):
        states, query_instructions = inputs

        states = states.to(device)
        instructions = instructions.to(device)

        instruction_pad = instructions == pad_word_idx
        query_instruction_pad = query_instructions == pad_word_idx
        state_pad = (states == 0).all(dim=-1).bool()

        with torch.inference_mode(), torch.autocast(
            device_type=device, dtype=torch.float16, enabled=True
        ):
            # Encode the query instructions and state
            # using the model's encoder and get the representations
            query_encodings = model.encode(
                query_instructions,
                states,
                torch.cat([query_instruction_pad, state_pad], dim=-1),
                query_instruction_pad
            )
            select_query_instructions_mask = torch.cat([
                torch.ones_like(query_instruction_pad),
                torch.zeros_like(state_pad)
            ], dim=-1)
            query_encodings[~select_query_instructions_mask] = 0
            query_encodings = F.normalize(query_encodings, dim=-1)
            instruction_encodings = model.encode(
                instructions,
                states,
                torch.cat([instruction_pad, state_pad], dim=-1),
                instruction_pad
            )
            select_instructions_mask = torch.cat([
                torch.ones_like(instruction_pad),
                torch.zeros_like(state_pad)
            ], dim=-1)
            instruction_encodings[select_instructions_mask] = 0
            instruction_encodings = F.normalize(instruction_encodings, dim=-1)

        # Now that we have them, compute the max-bsr ordering. This is descending-order
        indices_descending_bsr = np.array(compute_sorted_bsr(
            query_encodings.cpu().numpy(),
            instruction_encodings.cpu().numpy(),
            target_limit=instruction_encodings.shape[0]
        ))
        scores = np.ones(instruction_encodings.shape[0]) * float('-inf')
        scores[indices_descending_bsr] = np.arange(indices_descending_bsr.shape[0])[::-1]

        return torch.tensor(scores)

    return compute_scores


def make_gscan_generate_targets_closure(
    transformer_model, pad_word_idx, pad_action_idx, device="cpu"
):
    transformer_model.to(device)
    transformer_model.eval()

    transformer_model.encoder.pos_encoding.cached_penc = None
    transformer_model.decoder.pos_encoding.cached_penc = None

    def compute_targets(instructions, inputs, decode_len):
        states, query_instructions = inputs

        predicted_targets, logits = transformer_predict(
            transformer_model,
            states,
            instructions,
            decode_len,
        )

        return predicted_targets

    return compute_targets


def make_gscan_format_output_closure(WORD2IDX, ACTION2IDX, IDX2WORD, IDX2ACTION):
    def format_output(inputs, targets, sample_scores):
        (generated_instructions, generated_targets, scores) = list(zip(*sample_scores))
        query_state, query_instruction = inputs

        if True:
            import pprint
            print([IDX2WORD[w] for w in query_instruction.cpu().numpy()])
            pad_word = WORD2IDX["[pad]"]
            pad_act = ACTION2IDX["[pad]"]
            pprint.pprint([
                (" ".join([
                    IDX2WORD[w] for w in s if w != pad_word
                ]), " ".join([
                    IDX2ACTION[w] for w in a if w != pad_act
                ]))
                for s, a in zip(generated_instructions, generated_targets)
            ])

        return (
            query_instruction.numpy(),
            targets.numpy(),
            query_state.numpy(),
            query_state.numpy(),
            generated_instructions,
            generated_targets,
            scores,
        )

    return format_output


class SamplingError(Exception):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "SamplingError"


def try_gen_instructions(
    inputs,
    targets,
    instruction_gen_closure,
    instruction_ranking_closure,
    target_gen_closure,
    format_output_closure,
    sample_n,
    batch_size,
    decode_len,
    device="cpu",
    no_query_overlap=False,
    allow_targets_overlap=False,
    deduplicate_by_output=False,
    no_sort=False,
    demonstrations_limit=None
):
    sampled_instructions = instruction_gen_closure(inputs, sample_n)

    # Now for each element in the batch, we take
    # the set (involves a conversion back to tuples
    # but its fine)
    sampled_instructions_sets = [
        set([tuple(x.tolist()) for x in sampled_instruction])
        for sampled_instruction in sampled_instructions
    ]
    sampled_instruction_set_ids = np.concatenate(
        [np.array([i] * len(s)) for i, s in enumerate(sampled_instructions_sets)]
    )
    # We want batches of inputs, so we transpose here
    sampled_instruction_set_inputs = [
        np.concatenate(
            [
                np.repeat(
                    input_element[None].numpy(),
                    len(sampled_instruction_set),
                    axis=0,
                )
                for input_element, sampled_instruction_set in zip(
                    input_elements_batch, sampled_instructions_sets
                )
            ],
            axis=0,
        )
        for input_elements_batch in inputs
    ]
    try:
        concat_sampled_instruction_set_inputs = np.concatenate(
            [
                np.stack([np.array(x) for x in sampled_instruction_set])
                for sampled_instruction_set in sampled_instructions_sets
            ]
        )
    except ValueError:
        raise SamplingError()

    # Note: we are kind of assuming the gscan dataset here
    original_pred_targets = target_gen_closure(inputs[1], inputs, max(targets.shape[-1], decode_len)).cpu()

    per_id_results = defaultdict(list)

    for i in trange(
        concat_sampled_instruction_set_inputs.shape[0] // batch_size + 1,
        desc="Score/pred batch",
        disable=True
    ):
        first_index = i * batch_size
        last_index = (i + 1) * batch_size

        # Skip if this batch would have a size of zero
        if first_index >= concat_sampled_instruction_set_inputs.shape[0]:
            continue

        sampled_instruction_set_ids_batch = torch.from_numpy(
            sampled_instruction_set_ids[first_index:last_index]
        ).to(device)
        sampled_instruction_set_inputs_batch = tuple(
            [
                torch.from_numpy(
                    sampled_instruction_set_inputs_elements[first_index:last_index]
                ).to(device)
                for sampled_instruction_set_inputs_elements in sampled_instruction_set_inputs
            ]
        )
        sampled_instruction_set_batch = torch.from_numpy(
            concat_sampled_instruction_set_inputs[first_index:last_index]
        ).to(device)
        sampled_instruction_set_scores = instruction_ranking_closure(
            sampled_instruction_set_batch, sampled_instruction_set_inputs_batch
        )
        sampled_instruction_set_targets = target_gen_closure(
            sampled_instruction_set_batch,
            sampled_instruction_set_inputs_batch,
            targets.shape[-1],
        )

        # Now we populate per_id_results with the score
        # and the result_set_ids
        for batch_id, sample_result, predicted_target, score in zip(
            sampled_instruction_set_ids_batch.cpu(),
            sampled_instruction_set_batch.cpu(),
            sampled_instruction_set_targets.cpu(),
            sampled_instruction_set_scores.cpu(),
        ):
            batch_id = batch_id.item()

            # Filter out demonstrations that do not give any new information
            if not allow_targets_overlap and (targets[batch_id] == predicted_target).all(axis=-1):
                continue

            # Also filter out demonstrations which match what we would have predicted
            # under the original instruction.
            #
            # In practice, this does not seem to help performance very much
            if no_query_overlap and (original_pred_targets[batch_id] == predicted_target).all(axis=-1):
                continue

            per_id_results[batch_id].append(
                (sample_result.numpy(), predicted_target.numpy(), score.item())
            )

    if any([not v for v in per_id_results.items()]):
        raise SamplingError()

    if deduplicate_by_output:
        # Keep only unique trajectories. We do this by sorting by the predicted
        # target (as a string, lexicographically) as the primary sort key, then the
        # negated score as the secondary sort key. Then we can take groups and
        # take only the first item from each group. This is to handle the case where
        # multiple generated instructions end up demonstrating the same thing - this
        # duplication does not really help us, so we just eliminate it
        #
        # In practice, this does not seem to help performance very much
        per_id_results = {
            i: [list(g)[0] for k, g in itertools.groupby(
                sorted(results, key=lambda x: ("".join(list(map(str, x[1]))), -x[-1])),
                key=lambda x: "".join(list(map(str, x[1])))
            )]
            for i, results in per_id_results.items()
        }

    # Now we sort the per_id_results descending by score
    # (eg, a higher score is better)
    if not no_sort:
        per_id_results = {
            i: sorted(results, key=lambda x: -x[-1])[:demonstrations_limit]
            for i, results in per_id_results.items()
        }

    # Now yield per_id, the state, the query instruction and all supports
    # and their scores
    return [
        format_output_closure(
            tuple([i[batch_id] for i in inputs]), targets[batch_id], sample_scores
        )
        for batch_id, sample_scores in sorted(
            per_id_results.items(), key=lambda x: x[0]
        )
    ]


def generate_instructions_and_rank(
    instruction_gen_closure,
    instruction_ranking_closure,
    target_gen_closure,
    format_output_closure,
    dataloader,
    sample_n,
    batch_size,
    decode_len,
    device="cpu",
    no_query_overlap=False,
    allow_targets_overlap=False,
    deduplicate_by_output=False,
    no_sort=False,
    demonstrations_limit=None
):
    for batch in dataloader:
        inputs, targets = batch

        while True:
            try:
                yield from try_gen_instructions(
                    inputs,
                    targets,
                    instruction_gen_closure,
                    instruction_ranking_closure,
                    target_gen_closure,
                    format_output_closure,
                    sample_n,
                    batch_size,
                    decode_len,
                    device=device,
                    no_query_overlap=no_query_overlap,
                    allow_targets_overlap=allow_targets_overlap,
                    deduplicate_by_output=deduplicate_by_output,
                    no_sort=no_sort,
                    demonstrations_limit=demonstrations_limit
                )
                break
            except SamplingError:
                print("Sampling error")
                continue


def sample_from_state_encoder_decoder_model_with_mask(
    model,
    expanded_instruction,
    expanded_state,
    all_mask,
    instruction_mask,
    pad_tgt_idx,
    eos_tgt_idx,
    noise_level=0.2,
    device="cpu",
    deterministic=False,
):
    model.eval()
    model.to(device)

    unroll_length = expanded_instruction.shape[1]

    with torch.inference_mode(), torch.autocast(
        device_type=device, dtype=torch.float16, enabled=True
    ):
        decoded_instruction = (
            torch.ones_like(expanded_instruction[:, :1]) * model.sos_word_idx
        )

        # We always keep the very last pad bit unset, so that the entire
        # input is not padded
        instruction_mask[:, -1] = False

        encodings = model.encode(
            expanded_instruction, expanded_state, all_mask, instruction_mask
        )

        for i in trange(unroll_length, desc="Gen instrs", disable=True):
            stopped_mask = (decoded_instruction == pad_tgt_idx).any(dim=-1)
            still_going_mask = ~stopped_mask
            still_going_indices = torch.nonzero(still_going_mask).flatten()

            if still_going_mask.any(dim=-1):
                decoded_instruction_still_going = decoded_instruction[still_going_mask]
                encodings_still_going = encodings[still_going_mask]
                key_padding_mask_still_going = all_mask[still_going_mask]

                current_logits = model.decode(
                    encodings_still_going,
                    key_padding_mask_still_going,
                    decoded_instruction_still_going,
                )
                if deterministic:
                    samples = current_logits[:, -1].argmax(dim=-1)
                else:
                    samples = torch.distributions.Categorical(
                        logits=current_logits[:, -1] + 10
                    ).sample()

                scatter_target = torch.zeros_like(samples[0, None].expand(encodings.shape[0]))
                scatter_target.scatter_(
                    0,
                    still_going_indices,
                    samples
                )
                decoded_instruction = torch.cat(
                    [decoded_instruction, scatter_target[:, None]], dim=1
                )
            else:
                decoded_instruction = torch.cat(
                    [decoded_instruction, torch.ones_like(decoded_instruction)[:, :1] * pad_tgt_idx], dim=1
                )

        # Set the decoded-so-far instruction to be pad_tgt_idx if we hit pad_tgt_idx once
        decoded_instruction[
            (decoded_instruction == pad_tgt_idx).cumsum(dim=-1).bool()
        ] = pad_tgt_idx

        return (
            expanded_instruction.cpu(),
            decoded_instruction[:, 1:].view(expanded_instruction.shape[0], -1).cpu(),
            instruction_mask.view(expanded_instruction.shape[0], -1).cpu(),
        )


def sample_from_state_encoder_decoder_model(
    model,
    instruction,
    state,
    sample_n,
    pad_target_idx,
    eos_target_idx,
    noise_level=0.2,
    device="cpu",
    deterministic=False,
):
    model.eval()
    model.to(device)

    unroll_length = instruction.shape[1]

    expanded_instruction = (
        instruction[:, None].expand(-1, sample_n, -1).flatten(0, 1).clone()
    )
    expanded_instruction = expanded_instruction.to(device)

    expanded_state = state[:, None].expand(-1, sample_n, -1, -1).flatten(0, 1).clone()
    expanded_state = expanded_state.to(device)

    instruction_state_padding = torch.cat(
        [expanded_instruction == pad_target_idx, (expanded_state == 0).all(dim=-1)],
        dim=-1,
    )

    all_mask = torch.logical_or(
        torch.rand(instruction_state_padding.shape, device=device) < noise_level,
        instruction_state_padding,
    )
    instruction_only_mask = all_mask[:, : expanded_instruction.shape[1]]

    # We always keep the very last pad bit unset, so that the entire
    # input is not padded
    all_mask[:, -1] = False
    instruction_only_mask[:, -1] = False

    (
        expanded_instruction,
        decoded_instruction,
        instruction_mask,
    ) = sample_from_state_encoder_decoder_model_with_mask(
        model,
        expanded_instruction,
        expanded_state,
        all_mask,
        instruction_only_mask,
        pad_target_idx,
        eos_target_idx,
        noise_level=noise_level,
        device=device,
        deterministic=deterministic,
    )

    return (
        instruction,
        decoded_instruction.view(instruction.shape[0], sample_n, -1),
        instruction_mask.view(instruction.shape[0], sample_n, -1),
    )


def make_inv_counts_dist(counts_dictionary):
    counts_array = np.zeros(max(counts_dictionary.keys()) + 1)
    for word, count in counts_dictionary.items():
        counts_array[word] = count

    inv_counts = 1 / counts_array
    inv_counts[counts_array == 0] = 0
    inv_counts_dist = inv_counts / inv_counts.sum()

    return inv_counts_dist


GSCAN_RANKING_FUNCS = {
    "clip": make_gscan_clip_ranking_closure,
    "seq2seq": make_gscan_seq2seq_ranking_closure,
    "max_bsr": make_gscan_max_bsr_ranking_closure
}


def gscan_make_closures(args, dictionaries, datasets, extra_data):
    WORD2IDX, ACTION2IDX = dictionaries

    pad_action = ACTION2IDX["[pad]"]
    pad_word = WORD2IDX["[pad]"]

    IDX2WORD = {i: w for w, i in WORD2IDX.items()}
    IDX2ACTION = {i: w for w, i in ACTION2IDX.items()}

    # Punching through the abstraction a bit, we reach
    # through the MapDataset and PaddingDataset to get the underlying
    # demonstrations without any padding
    train_demonstrations = datasets["train"].dataset.dataset

    training_data_indices_by_command = {}
    for i in range(len(train_demonstrations)):
        # A workaround in the gSCAN case to not include stuff from split G
        if WORD2IDX.get("cautiously", -1) in train_demonstrations[i][0]:
            continue

        cmd = " ".join(map(lambda x: IDX2WORD[x], train_demonstrations[i][0]))
        if cmd not in training_data_indices_by_command:
            training_data_indices_by_command[cmd] = []
        training_data_indices_by_command[cmd].append(i)

    min_len = min(
        [(cmd, len(x)) for cmd, x in training_data_indices_by_command.items()],
        key=lambda x: x[-1],
    )[-1]

    sentence2idx = {s: i for i, s in enumerate(training_data_indices_by_command)}
    idx2sentence = [s for s in sentence2idx]

    balanced_training_data_subset = SampleSentencesByWordWeights(
        {sentence2idx[s]: v for s, v in training_data_indices_by_command.items()},
        np.ones(len(sentence2idx)) / len(sentence2idx),
        MapDataset(datasets["train"], lambda x: (x[0][1], x[0][0])),
    )

    state_component_max_len, state_feat_len = determine_state_profile(train_demonstrations, {
        k: v.dataset.dataset
        for k, v in datasets.items()
    })

    model = train_state_encoder_decoder(
        balanced_training_data_subset,
        args.seed,
        0 if args.load_mlm_model else args.mlm_train_iterations,
        pad_word,
        WORD2IDX["[sos]"],
        state_feat_len,
        len(WORD2IDX),
        args.batch_size,
        device=args.device,
        load=args.load_mlm_model,
    )
    print(model)

    if args.save_mlm_model:
        torch.save(model.state_dict(), args.save_mlm_model)

    os.makedirs(os.path.join(args.data_output_directory), exist_ok=True)

    transformer_model_weights = torch.load(args.load_transformer_model)
    transformer_model_state_dict = transformer_model_weights["state_dict"] if "state_dict" in transformer_model_weights else transformer_model_weights
    loaded_hparams = transformer_model_weights["hyper_parameters"] if "hyper_parameters" in transformer_model_weights else None
    print(state_feat_len, state_component_max_len)
    print(loaded_hparams)
    transformer_model_hparams = loaded_hparams or dict(
        n_state_components=state_feat_len,
        x_categories=len(WORD2IDX),
        y_categories=len(ACTION2IDX),
        embed_dim=512,
        dropout_p=0.0,
        nlayers=12,
        nhead=8,
        pad_word_idx=WORD2IDX['[pad]'],
        pad_action_idx=ACTION2IDX['[pad]'],
        sos_action_idx=ACTION2IDX['[sos]'],
        eos_action_idx=ACTION2IDX['[eos]'],
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

    transformer_model_trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else None,
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
                MapDataset(
                    Subset(data, np.random.permutation(min(len(data), 512))),
                    lambda x: (x[0][1], x[1], x[0][0]),
                ),
                batch_size=args.batch_size,
                pin_memory=True,
            )
            for data in {
                k: v for k, v in datasets.items()
                if k != "train"
            }.values()
        ],
    )
    print(transformer_model)

    return (
        make_gscan_instruction_gen_closure(
            model, pad_word, None, device=args.device, noise_level=args.noise_level
        ),
        GSCAN_RANKING_FUNCS[args.ranking_func](
            model, pad_word, WORD2IDX["[sos]"], device=args.device
        ),
        make_gscan_generate_targets_closure(
            transformer_model, pad_word, pad_action, device=args.device
        ),
        make_gscan_format_output_closure(WORD2IDX, ACTION2IDX, IDX2WORD, IDX2ACTION),
    )


def gscan_load_data(args):
    (
        dictionaries,
        (train_demonstrations, valid_demonstrations_dict),
    ) = load_data_directories(
        args.data_directory, args.dictionary, limit_load=args.limit_load
    )

    WORD2IDX = dictionaries[0]
    ACTION2IDX = dictionaries[1]

    instruction_padding, action_padding, state_padding = determine_padding(train_demonstrations)
    state_component_max_len, state_feat_len = determine_state_profile(train_demonstrations, {
        k: v
        for k, v in {
            **valid_demonstrations_dict,
            "train": train_demonstrations
        }.items()
    })

    dataset_splits = {
        split: MapDataset(
            PaddingDataset(
                demos,
                (instruction_padding, action_padding, (state_padding, state_feat_len)),
                (WORD2IDX["[pad]"], ACTION2IDX["[pad]"], 0),
            ),
            lambda x: ((x[2], x[0]), x[1]),
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

    return ((WORD2IDX, ACTION2IDX), dataset_splits, dictionaries[2:])


def sample_from_encoder_decoder_model_with_mask(
    model,
    expanded_instruction,
    instruction_mask,
    eos_target_idx,
    pad_target_idx,
    noise_level=0.2,
    device="cpu",
    deterministic=False,
):
    model.eval()
    model.to(device)

    unroll_length = expanded_instruction.shape[1]

    with torch.inference_mode(), torch.autocast(
        device_type=device, dtype=torch.float16, enabled=True
    ):
        decoded_instruction = (
            torch.ones_like(expanded_instruction[:, :1]) * model.sos_word_idx
        )

        # We always keep the very last pad bit unset, so that the entire
        # input is not padded
        instruction_mask[:, -1] = False

        for i in range(unroll_length):
            logits = model(expanded_instruction, instruction_mask, decoded_instruction)
            if deterministic:
                samples = logits[:, -1].argmax(dim=-1)
            else:
                samples = torch.distributions.Categorical(
                    logits=logits[:, -1] + 10
                ).sample()
            decoded_instruction = torch.cat(
                [decoded_instruction, samples[:, None]], dim=1
            )

        # Set anything past EOS to be a PAD token
        decoded_eq_mask = (
            (decoded_instruction == eos_target_idx).int().cumsum(dim=-1).bool()[:, :-1]
        )
        decoded_eq_mask = torch.cat(
            [torch.zeros_like(decoded_eq_mask[:, :1]), decoded_eq_mask], dim=-1
        )
        decoded_instruction
        decoded_instruction[decoded_eq_mask] = pad_target_idx

        return (
            expanded_instruction.cpu(),
            decoded_instruction[:, 1:].view(expanded_instruction.shape[0], -1).cpu(),
            instruction_mask.view(expanded_instruction.shape[0], -1).cpu(),
        )


def sample_from_encoder_decoder_model(
    model,
    instruction,
    eos_target_idx,
    pad_target_idx,
    sample_n,
    noise_level=0.2,
    device="cpu",
    deterministic=False,
):
    unroll_length = instruction.shape[1]

    expanded_instruction = instruction[:, None].expand(-1, sample_n, -1).flatten(0, 1)
    expanded_instruction = expanded_instruction.to(device)

    instruction_padding = expanded_instruction == pad_target_idx

    instruction_mask = torch.logical_or(
        torch.rand(instruction_padding.shape, device=device) < noise_level,
        instruction_padding,
    )

    (
        expanded_instruction,
        decoded_instruction,
        instruction_mask,
    ) = sample_from_encoder_decoder_model_with_mask(
        model,
        expanded_instruction,
        instruction_mask,
        eos_target_idx,
        pad_target_idx,
        noise_level=noise_level,
        device=device,
        deterministic=deterministic,
    )

    return (
        instruction,
        decoded_instruction.view(instruction.shape[0], sample_n, -1),
        instruction_mask.view(instruction.shape[0], sample_n, -1),
        [
            [i[m] for i, m in zip(i_batch, m_batch)]
            for i_batch, m_batch in zip(
                expanded_instruction.view(instruction.shape[0], sample_n, -1),
                (~instruction_mask).view(instruction.shape[0], sample_n, -1),
            )
        ],
    )


def sample_from_encoder_decoder_model_deterministic(
    model,
    instruction,
    eos_target_idx,
    pad_target_idx,
    noise_level=0.2,
    device="cpu",
    deterministic=False,
):
    unroll_length = instruction.shape[1]

    expanded_instruction = (
        instruction[:, None].expand(-1, instruction.shape[-1], -1).flatten(0, 1)
    )
    expanded_instruction = expanded_instruction.to(device)

    # We'll make a mask of 1-grams, 2-grams
    mask_lowerbound = (
        torch.arange(expanded_instruction.shape[0], device=device)
        % (expanded_instruction.shape[-1] + 1)
    ) - 1
    mask_upperbound = mask_lowerbound + 1
    instruction_mask_indices = torch.arange(
        expanded_instruction.shape[1], device=device
    )[None].expand(expanded_instruction.shape[0], -1)
    instruction_mask = ~torch.bitwise_and(
        instruction_mask_indices >= mask_lowerbound[:, None],
        instruction_mask_indices < mask_upperbound[:, None],
    )

    (
        expanded_instruction,
        decoded_instruction,
        instruction_mask,
    ) = sample_from_encoder_decoder_model_with_mask(
        model,
        expanded_instruction,
        instruction_mask,
        eos_target_idx,
        pad_target_idx,
        noise_level=noise_level,
        device=device,
        deterministic=deterministic,
    )

    return (
        instruction,
        decoded_instruction.view(instruction.shape[0], instruction.shape[-1], -1),
        instruction_mask.view(instruction.shape[0], instruction.shape[-1], -1),
        [
            [i[m] for i, m in zip(i_batch, m_batch)]
            for i_batch, m_batch in zip(
                expanded_instruction.view(
                    instruction.shape[0], instruction.shape[-1], -1
                ),
                (~instruction_mask).view(
                    instruction.shape[0], instruction.shape[-1], -1
                ),
            )
        ],
    )


def make_cogs_instruction_gen_closure(
    encoder_decoder_model, eos_target_idx, pad_target_idx, device="cpu", noise_level=0.2
):
    encoder_decoder_model.to(device)
    encoder_decoder_model.eval()

    def generate_instruction(inputs, sample_n):
        (query_instructions,) = inputs
        (
            result_instrs,
            result_samples,
            result_samples_mask,
            result_masked_instruction,
        ) = sample_from_encoder_decoder_model(
            encoder_decoder_model,
            query_instructions,
            eos_target_idx,
            pad_target_idx,
            sample_n,
            noise_level=noise_level,
            device="cuda",
            deterministic=False,
        )

        is_same_mask = (result_samples == query_instructions[:, None]).all(dim=-1)

        return [
            s[~m].reshape(-1, s.shape[-1]) for s, m in zip(result_samples, is_same_mask)
        ]

    return generate_instruction


def compute_ll_from_encoder_decoder_model(model, query, samples, device="cuda"):
    samples = samples.to(device)
    query = query.to(device)

    flat_query = query
    flat_samples = samples
    encoder_mask = flat_query == model.pad_word_idx
    decoded_instruction = torch.cat(
        [torch.ones_like(flat_samples[:, :1]) * model.sos_word_idx, flat_samples],
        dim=-1,
    )[:, :-1]

    with torch.inference_mode():
        logits = model(flat_query, encoder_mask, decoded_instruction)
        logprobs = torch.gather(logits, -1, flat_samples[..., None]).squeeze(-1)
        total_logprobs = logprobs.sum(dim=-1)

    return total_logprobs.detach().cpu()


def make_cogs_instruction_ranking_closure(encoder_decoder_model, device="cpu"):
    encoder_decoder_model.to(device)
    encoder_decoder_model.eval()

    def score_instructions(sampled, inputs):
        generated_instructions = sampled
        (query_instructions,) = inputs
        return compute_ll_from_encoder_decoder_model(
            encoder_decoder_model,
            query_instructions,
            generated_instructions,
            device=device,
        )

    return score_instructions


def make_cogs_target_generation_closure(
    transformer_encoder_decoder_model, device="cpu"
):
    transformer_encoder_decoder_model.to(device)
    transformer_encoder_decoder_model.eval()

    transformer_encoder_decoder_model.encoder.pos_encoding.cached_penc = None
    transformer_encoder_decoder_model.decoder.pos_encoding.cached_penc = None

    def compute_targets(instructions, inputs, decode_len):
        (query_instructions,) = inputs

        dummy_targets = torch.zeros(
            query_instructions.shape[0], decode_len, dtype=torch.long, device=device
        )
        instructions = instructions.to(device)

        decoded, logits, exacts, _ = autoregressive_model_unroll_predictions(
            transformer_encoder_decoder_model,
            (instructions,),
            dummy_targets,
            transformer_encoder_decoder_model.sos_action_idx,
            transformer_encoder_decoder_model.eos_action_idx,
            transformer_encoder_decoder_model.pad_action_idx,
        )

        return decoded

    return compute_targets


def make_cogs_format_output_closure():
    def format_output(inputs, targets, sample_scores):
        (generated_instructions, generated_targets, scores) = list(zip(*sample_scores))
        (query_instruction,) = inputs

        return (
            query_instruction.numpy(),
            targets.numpy(),
            generated_instructions,
            generated_targets,
            scores,
        )

    return format_output


def train_encoder_decoder_transformer(
    dataset,
    valid_dataset_dict,
    seed,
    iterations,
    pad_word,
    pad_action,
    sos_action,
    eos_action,
    query_vocab_size,
    output_vocab_size,
    batch_size,
    device="cpu",
):
    nlayers = 4
    nhead = 8
    hidden_size = 128
    dropout_p = 0.1
    train_batch_size = batch_size
    batch_size_mult = 1
    dataset_name = "cogs"
    check_val_every = 8000

    exp_name = "cogs_enc_dec_transformer"
    model_name = f"transformer_l_{nlayers}_h_{nhead}_d_{hidden_size}"
    dataset_name = dataset_name
    effective_batch_size = train_batch_size * batch_size_mult
    exp_name = f"{exp_name}_s_{seed}_m_{model_name}_it_{iterations}_b_{effective_batch_size}_d_{dataset_name}_drop_{dropout_p}"
    model_dir = f"models/{exp_name}/{model_name}"
    model_path = f"{model_dir}/{exp_name}.pt"
    print(model_path)
    print(
        f"Batch size {train_batch_size}, mult {batch_size_mult}, total {train_batch_size * batch_size_mult}"
    )

    train_dataloader = DataLoader(
        dataset, batch_size=train_batch_size, pin_memory=True, shuffle=True
    )
    valid_dataloaders = [
        DataLoader(
            Subset(ds, np.random.permutation(len(ds))[:128]),
            batch_size=train_batch_size,
            pin_memory=True,
        )
        for ds in valid_dataset_dict.values()
    ]

    logs_root_dir = f"logs/{exp_name}/{model_name}/{dataset_name}/{seed}"

    model = SequenceTransformerLearner(
        query_vocab_size,
        output_vocab_size,
        hidden_size,
        dropout_p,
        nlayers,
        nhead,
        pad_word,
        pad_action,
        sos_action,
        eos_action,
        norm_first=True,
        lr=1e-4,
        decay_power=-1,
        warmup_proportion=0.1,
    )

    trainer = pl.Trainer(
        logger=[
            TensorBoardLogger(logs_root_dir),
            LoadableCSVLogger(logs_root_dir, flush_logs_every_n_steps=10),
        ],
        callbacks=[pl.callbacks.LearningRateMonitor()],
        max_steps=iterations,
        num_sanity_val_steps=10,
        gpus=1 if device == "cuda" else 0,
        precision=16 if device == "cuda" else 32,
        default_root_dir=logs_root_dir,
        accumulate_grad_batches=batch_size_mult,
        gradient_clip_val=0.2,
        check_val_every_n_epoch=int(len(train_dataloader) / 5000) + 1,
    )

    trainer.fit(model, train_dataloader, valid_dataloaders)

    return model


def cogs_make_closures(args, dictionaries, datasets, extra_data):
    in_word2idx, out_word2idx = dictionaries

    train_inputs = MapDataset(datasets["train"], lambda x: x[0])
    train_data_indices_by_word_idx = defaultdict(list)

    for i, train_sentence in enumerate(train_inputs):
        for word in train_sentence[0][train_sentence[0] != in_word2idx["[pad]"]]:
            train_data_indices_by_word_idx[word].append(i)

    inv_counts = make_inv_counts_dist(
        {k: len(v) for k, v in train_data_indices_by_word_idx.items()}
    )

    sample_dataset = SampleSentencesByWordWeights(
        train_data_indices_by_word_idx, inv_counts, train_inputs
    )

    encoder_decoder_lm = train_encoder_decoder(
        sample_dataset,
        args.seed,
        args.mlm_train_iterations if not args.load_mlm_model else 0,
        in_word2idx["[pad]"],
        in_word2idx["[sos]"],
        in_word2idx["[eos]"],
        len(in_word2idx),
        args.batch_size,
        device=args.device,
    )

    if args.load_mlm_model:
        encoder_decoder_lm.load_state_dict(torch.load(args.load_mlm_model))

    if args.save_mlm_model:
        torch.save(encoder_decoder_lm.state_dict(), args.save_mlm_model)

    encoder_decoder_transformer = train_encoder_decoder_transformer(
        MapDataset(datasets["train"], lambda x: (x[0][0], x[1])),
        {
            k: MapDataset(v, lambda x: (x[0][0], x[1]))
            for k, v in datasets.items()
            if k != "train"
        },
        args.seed,
        args.transformer_train_iterations if not args.load_transformer_model else 0,
        in_word2idx["[pad]"],
        out_word2idx["[pad]"],
        out_word2idx["[sos]"],
        out_word2idx["[eos]"],
        len(in_word2idx),
        len(out_word2idx),
        args.batch_size,
        device=args.device,
    )

    if args.load_transformer_model:
        encoder_decoder_transformer.load_state_dict(
            torch.load(args.load_transformer_model)
        )

    if args.save_transformer_model:
        torch.save(
            encoder_decoder_transformer.state_dict(), args.save_transformer_model
        )

    return (
        make_cogs_instruction_gen_closure(
            encoder_decoder_lm,
            in_word2idx["[eos]"],
            in_word2idx["[pad]"],
            device=args.device,
            noise_level=0.2,
        ),
        make_cogs_instruction_ranking_closure(encoder_decoder_lm, device=args.device),
        make_cogs_target_generation_closure(
            encoder_decoder_transformer, device=args.device
        ),
        make_cogs_format_output_closure(),
    )


def cogs_parse_task(task):
    input_statement, output_statement, category = task

    return (input_statement.split(), output_statement.split(), category)


def cogs_make_vocab(examples):
    idx2word = sorted(list(set(itertools.chain.from_iterable(examples))))
    idx2word.append("[sos]")
    idx2word.append("[eos]")
    idx2word.append("[pad]")
    word2idx = {w: i for i, w in enumerate(idx2word)}

    return idx2word, word2idx


def cogs_load_data(args, filter_input_length=32, filter_output_length=128):
    train_df = pd.read_csv(
        os.path.join(args.data_directory, "train_100.tsv"), sep="\t", header=None
    )
    test_df = pd.concat(
        [
            pd.read_csv(
                os.path.join(args.data_directory, "test.tsv"), sep="\t", header=None
            ),
            pd.read_csv(
                os.path.join(args.data_directory, "gen.tsv"), sep="\t", header=None
            ),
        ],
        axis=0,
    )

    train_df = train_df.sort_values(2, axis=0)
    test_df = test_df.sort_values(2, axis=0)

    cogs_train = list(map(cogs_parse_task, train_df.values))
    cogs_test = list(map(cogs_parse_task, test_df.values))

    in_idx2word, in_word2idx = cogs_make_vocab(
        map(
            operator.itemgetter(0),
            itertools.chain.from_iterable([cogs_train, cogs_test]),
        )
    )
    out_idx2word, out_word2idx = cogs_make_vocab(
        map(
            operator.itemgetter(1),
            itertools.chain.from_iterable([cogs_train, cogs_test]),
        )
    )

    cogs_train_idx_inputs, cogs_train_idx_outputs = list(
        zip(
            *[
                (
                    [in_word2idx[w] for w in in_sentence] + [in_word2idx["[eos]"]],
                    [out_word2idx[w] for w in out_sentence] + [out_word2idx["[eos]"]],
                )
                for in_sentence, out_sentence, category in cogs_train
            ]
        )
    )

    cogs_test_idx_inputs, cogs_test_idx_outputs = list(
        zip(
            *[
                (
                    [in_word2idx[w] for w in in_sentence] + [in_word2idx["[eos]"]],
                    [out_word2idx[w] for w in out_sentence] + [out_word2idx["[eos]"]],
                )
                for in_sentence, out_sentence, category in cogs_test
            ]
        )
    )

    cogs_grouped_train_data = {
        "train": list(zip(cogs_train_idx_inputs, cogs_train_idx_outputs))
    }
    cogs_grouped_test_data = {
        k: list(map(lambda x: (x[1], x[2]), v))
        for k, v in itertools.groupby(
            zip(cogs_test, cogs_test_idx_inputs, cogs_test_idx_outputs),
            key=lambda x: x[0][-1],
        )
    }

    cogs_grouped_train_datasets = {
        k: MapDataset(
            PaddingDataset(
                MapDataset(
                    list(
                        filter(
                            lambda x: len(x[1]) <= filter_output_length
                            or len(x[0]) <= filter_input_length,
                            v,
                        )
                    ),
                    lambda x: (np.array(x[0]), np.array(x[1])),
                ),
                (filter_input_length, filter_output_length),
                (in_word2idx["[pad]"], out_word2idx["[pad]"]),
            ),
            lambda x: ((x[0],), x[1]),
        )
        for k, v in itertools.chain.from_iterable(
            [cogs_grouped_train_data.items(), cogs_grouped_test_data.items()]
        )
    }

    return ((in_word2idx, out_word2idx), cogs_grouped_train_datasets, None)


def gscan_add_subparser(subparsers):
    gscan_parser = subparsers.add_parser("gscan", help="gscan generation help")
    gscan_parser.add_argument("--dictionary", type=str, required=True)
    gscan_parser.add_argument("--mlm-train-iterations", type=int, default=100000)
    gscan_parser.add_argument("--clip-train-iterations", type=int, default=100000)
    gscan_parser.add_argument("--load-mlm-model", type=str)
    gscan_parser.add_argument("--save-mlm-model", type=str)
    gscan_parser.add_argument("--load-transformer-model", type=str, required=True)
    gscan_parser.add_argument("--limit-load", type=int, default=None)
    gscan_parser.add_argument("--ranking-func", type=str, choices=list(GSCAN_RANKING_FUNCS.keys()), default="seq2seq")
    gscan_parser.add_argument("--noise-level", type=float, default=0.2)


def cogs_add_subparser(subparsers):
    gscan_parser = subparsers.add_parser("cogs", help="cogs generation help")
    gscan_parser.add_argument("--mlm-train-iterations", type=int, default=100000)
    gscan_parser.add_argument(
        "--transformer-train-iterations", type=int, default=100000
    )
    gscan_parser.add_argument("--load-mlm-model", type=str)
    gscan_parser.add_argument("--save-mlm-model", type=str)
    gscan_parser.add_argument("--load-transformer-model", type=str)
    gscan_parser.add_argument("--save-transformer-model", type=str)
    gscan_parser.add_argument("--limit-load", type=int, default=None)


def load_and_count_file(file_path):
    with open(file_path, "rb") as f:
        object_len = len(pickle.load(f))
        tqdm.write(f"{file_path} - found {object_len} items\n")
        return object_len


def running_count_sum_till_exception(split_path, filenames):
    running_sum = 0
    try:
        for fname in filenames:
            running_sum += load_and_count_file(os.path.join(split_path, fname))
    except:
        pass

    return running_sum


def load_and_count_from_split_dir(split_path):
    filenames = os.listdir(split_path)
    filenames = sorted(filenames, key=lambda x: int(os.path.splitext(x)[0]))

    return running_count_sum_till_exception(
        split_path, list(map(lambda fname: os.path.join(split_path, fname), filenames))
    )


def compute_resume_points(data_output_directory):
    try:
        return {
            dirname: load_and_count_from_split_dir(
                os.path.join(data_output_directory, dirname)
            )
            for dirname in tqdm(
                os.listdir(data_output_directory),
                desc="Computing resume points"
            )
            if os.path.isdir(os.path.join(data_output_directory, dirname))
        }
    except IOError:
        import traceback
        traceback.print_exc()
        return {}


DATASET_CONFIGS = {
    "gscan": {
        "add_subparser": gscan_add_subparser,
        "load_data": gscan_load_data,
        "make_closures": gscan_make_closures,
    },
    "cogs": {
        "add_subparser": cogs_add_subparser,
        "load_data": cogs_load_data,
        "make_closures": cogs_make_closures,
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-directory", type=str, required=True)
    parser.add_argument("--data-output-directory", type=str, required=True)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gen-batch-size", type=int, default=16)
    parser.add_argument("--gen-sample-n", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only-splits", nargs="*", help="Which splits to include")
    parser.add_argument("--offset", type=float, default=0)
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-targets-overlap", action="store_true")
    parser.add_argument("--no-query-overlap", action="store_true")
    parser.add_argument("--deduplicate-by-outputs", action="store_true")
    parser.add_argument("--no-sort", action="store_true")
    parser.add_argument("--decode-to", type=int, default=256)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--demonstrations-limit", type=int, default=128)
    subparsers = parser.add_subparsers(dest="dataset")

    for config_name, config_values in DATASET_CONFIGS.items():
        config_values["add_subparser"](subparsers)

    args = parser.parse_args()

    dictionaries, datasets, extra_data = DATASET_CONFIGS[args.dataset]["load_data"](
        args
    )
    resume_points = compute_resume_points(args.data_output_directory) if args.resume else {}
    datasets_with_resume_points = {
        k: (v, resume_points.get(k, 0)) for k, v in datasets.items()
    }

    torch.set_float32_matmul_precision("medium")
    print("Flash attention:", torch.backends.cuda.flash_sdp_enabled())

    print(args.offset, args.offset + (args.limit or 0))
    dataset_offsets_string = '\n'.join([f'{k}: {v[1]}/{(args.limit or 1) * len(v[0])}' for k, v in datasets_with_resume_points.items()])
    print(f"Dataset offsets {dataset_offsets_string}")

    dataloader_offsets = {
        split: (
            min(
                math.floor(args.offset * len(dataset)) + resume_point_offset,
                len(dataset),
            ),
            min(
                math.floor(args.offset * len(dataset))
                + (
                    len(dataset)
                    if not args.limit
                    else math.floor(args.limit * len(dataset))
                ),
                len(dataset),
            )
        )
        for split, (dataset, resume_point_offset) in datasets_with_resume_points.items()
        if not args.only_splits or split in args.only_splits
    }
    dataloader_offsets_string = '\n'.join([
        f'{split}: {offset}'
        for split, offset in dataloader_offsets.items()
    ])
    print(dataloader_offsets_string)

    dataloader_splits = {
        split: DataLoader(
            Subset(
                dataset,
                np.arange(
                    min(
                        math.floor(args.offset * len(dataset)) + resume_point_offset,
                        len(dataset),
                    ),
                    min(
                        math.floor(args.offset * len(dataset))
                        + (
                            len(dataset)
                            if not args.limit
                            else math.floor(args.limit * len(dataset))
                        ),
                        len(dataset),
                    ),
                ),
            ),
            batch_size=args.gen_batch_size,
            pin_memory=True,
        )
        for split, (dataset, resume_point_offset) in datasets_with_resume_points.items()
        if not args.only_splits or split in args.only_splits
    }
    (
        instruction_gen_closure,
        ranking_closure,
        generate_targets_closure,
        format_output_closure,
    ) = DATASET_CONFIGS[args.dataset]["make_closures"](
        args, dictionaries, datasets, extra_data
    )

    os.makedirs(os.path.join(args.data_output_directory), exist_ok=True)
    with open(os.path.join(args.data_output_directory, "dictionary.pb"), "wb") as f:
        pickle.dump(dictionaries, f)

    file_batch_size = 10000

    for (split, dataloader) in tqdm(dataloader_splits.items(), desc="Splits"):
        # Note we still make always make the directory,
        # this is to ensure that the dataloader indices align correctly
        os.makedirs(f"{args.data_output_directory}/{split}", exist_ok=True)

        if len(dataloader.dataset) == 0:
            print(f"Skip {split} as it is empty")
            continue

        known_offset = resume_points.get(split, 0)
        i_offset = known_offset // file_batch_size

        for i, batch in enumerate(
            batched(
                generate_instructions_and_rank(
                    instruction_gen_closure,
                    ranking_closure,
                    generate_targets_closure,
                    format_output_closure,
                    tqdm(dataloader, f"Split ({split}) examples"),
                    args.gen_sample_n,
                    batch_size=args.batch_size,
                    decode_len=args.decode_to,
                    device=args.device,
                    no_query_overlap=args.no_query_overlap,
                    allow_targets_overlap=args.allow_targets_overlap,
                    deduplicate_by_output=args.deduplicate_by_outputs,
                    no_sort=args.no_sort,
                    demonstrations_limit=args.demonstrations_limit
                ),
                file_batch_size,
            )
        ):
            try:
                with open(os.path.join(args.data_output_directory, split, f"{i + i_offset}.pb"), "rb") as f:
                    existing_examples = pickle.load(f)
            except:
                existing_examples = []

            if not args.dry_run:
                with open(
                    os.path.join(args.data_output_directory, split, f"{i + i_offset}.pb"), "wb"
                ) as f:
                    pickle.dump(existing_examples + batch, f)


if __name__ == "__main__":
    main()
