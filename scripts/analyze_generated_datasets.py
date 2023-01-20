import argparse
import os
import itertools
import pandas as pd
import numpy as np
import re
import pickle

from gscan_metaseq2seq.util.load_data import load_data_directories


def segment_instruction(query_instruction, word2idx, colors, nouns):
    verb_words = [
        [word2idx[w] for w in v] for v in [["walk", "to"], ["push"], ["pull"]]
    ]
    adverb_words = [
        [word2idx[w] for w in v]
        for v in [
            ["while spinning"],
            ["while zigzagging"],
            ["hesitantly"],
            ["cautiously"],
        ]
    ]
    size_words = [word2idx[w] for w in ["small", "big"]]
    color_words = [word2idx[w] for w in list(colors)]
    noun_words = [word2idx[w] for w in list(nouns) if w in word2idx]

    query_verb_words = [
        v for v in verb_words if all([w in query_instruction for w in v])
    ]
    query_adverb_words = [
        v for v in adverb_words if all([w in query_instruction for w in v])
    ]
    query_size_words = [v for v in size_words if v in query_instruction]
    query_color_words = [v for v in color_words if v in query_instruction]
    query_noun_words = [v for v in noun_words if v in query_instruction]

    return (
        query_verb_words,
        query_adverb_words,
        query_size_words,
        query_color_words,
        query_noun_words,
    )


def find_agent_position(state):
    return [s for s in state if s[3] != 0][0]


def find_target_object(state, size, color, noun, idx2word, idx2color, idx2noun):
    color_word = [idx2word[c] for c in color]
    noun_word = [idx2word[c] for c in noun]
    size_word = [idx2word[c] for c in size]

    # Find any state elements with a matching noun, then
    # filter by matching color
    states_with_matching_noun = [
        s for s in state if s[2] and idx2noun[s[2] - 1] in noun_word
    ]
    states_with_matching_color = [
        s
        for s in states_with_matching_noun
        if s[1] and idx2color[s[1] - 1] in color_word or not color_word
    ]
    sorted_by_size = sorted(states_with_matching_color, key=lambda x: x[0])

    if not sorted_by_size:
        return None

    if size_word and size_word[0] == "small":
        return sorted_by_size[0]

    if size_word and size_word[0] == "big":
        return sorted_by_size[-1]

    return sorted_by_size[0]


def compute_statistics_for_example(example, word2idx, colors, nouns):
    idx2word = [w for w in word2idx]
    idx2color = [c for c in colors]
    idx2noun = [n for n in nouns]

    (
        query,
        target,
        state,
        support_state,
        support_query,
        support_target,
        ranking,
    ) = example
    query_verb, query_adverb, query_size, query_color, query_noun = segment_instruction(
        query, word2idx, colors, nouns
    )
    segmented_support_queries = [
        segment_instruction(support_instruction, word2idx, colors, nouns)
        for support_instruction in support_query
    ]
    support_state = (
        [support_state] * len(support_query)
        if isinstance(support_state[0], np.ndarray)
        else support_state
    )

    query_agent_position = find_agent_position(state)
    query_target_object = find_target_object(
        state, query_size, query_color, query_noun, idx2word, idx2color, idx2noun
    )
    support_agent_positions = [find_agent_position(state) for state in support_state]
    support_target_objects = [
        find_target_object(
            state,
            support_size,
            support_color,
            support_noun,
            idx2word,
            idx2color,
            idx2noun,
        )
        for state, (
            support_verb,
            support_adverb,
            support_size,
            support_color,
            support_noun,
        ) in zip(support_state, segmented_support_queries)
    ]

    matches = np.array(
        [
            (
                query_verb == support_verb,
                query_adverb == support_adverb,
                query_size + query_color + query_noun
                == support_size + support_color + support_noun,
                (query_agent_position[-2:] == support_agent_pos[-2:]).all(),
                support_target_object is not None
                and (query_target_object[-2:] == support_target_object[-2:]).all(),
                support_target_object is not None
                and (
                    (query_target_object[-2:] - query_agent_position[-2:])
                    == (support_target_object[-2:] - support_agent_pos[-2:])
                ).all(),
                support_target_object is not None
                and (query_target_object[:3] == support_target_object[:3]).all(),
                support_target_object is not None,
            )
            for (
                support_verb,
                support_adverb,
                support_size,
                support_color,
                support_noun,
            ), support_agent_pos, support_target_object in zip(
                segmented_support_queries,
                support_agent_positions,
                support_target_objects,
            )
        ]
    )

    return matches


def summarize_by_dividing_out_count(sum_summaries):
    return np.nan_to_num(
        sum_summaries[..., :-1].sum(axis=0) / sum_summaries[..., -1].sum(), 0.0
    )


def load_data_and_make_hit_results(data_directory, limit_load=None):
    (
        (
            WORD2IDX,
            ACTION2IDX,
            color_dictionary,
            noun_dictionary,
        ),
        (meta_train_demonstrations, meta_valid_demonstrations_dict),
    ) = load_data_directories(
        data_directory,
        os.path.join(data_directory, "dictionary.pb"),
        limit_load=limit_load,
    )

    color_dictionary = sorted(color_dictionary)
    noun_dictionary = sorted(noun_dictionary)

    return {
        split: summarize_by_dividing_out_count(
            np.stack(
                [
                    summarize_hits(
                        compute_statistics_for_example(
                            example, WORD2IDX, color_dictionary, noun_dictionary
                        )
                    )
                    for example in tqdm(examples, desc=f"Split {split}")
                ]
            )
        )
        for split, examples in tqdm(
            itertools.chain.from_iterable(
                [
                    [["train", meta_train_demonstrations]],
                    meta_valid_demonstrations_dict.items(),
                ]
            ),
            total=len(meta_valid_demonstrations_dict) + 1,
        )
    }


NAME_MAP = {
    "i2g": "DemoGen",
    "metalearn_allow_any": "Expert",
    "metalearn_find_matching_instruction_demos_allow_any": "Retrieval",
    "metalearn_random_instructions_same_layout_allow_any": "Random",
}


def extract_split_to_table(dataset_summaries, split):
    df = pd.DataFrame.from_dict(
        {k: v[split] for k, v in dataset_summaries.items()}, orient="columns"
    ).drop([5, 9, 10], axis="index")
    df.index = INDEX_COLS
    df.columns = [NAME_MAP.get(n, n) for n in df.columns]

    return df


INDEX_COLS = [
    f"\\footnotesize{{({i + 1}) {s}}}"
    for i, s in enumerate(
        [
            "Desc. Obj.",
            "Agent Pos.",
            "Tgt. Pos.",
            "Same Diff.",
            "Tgt. Obj.",
            "Verb \\& (5)",
            "Advb \\& (5)",
            "(6) \\& (7)",
            "(4) \\& (8)",
        ]
    )
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-directory", required=True)
    parser.add_argument("--limit-load", type=int, default=None)
    parser.add_argument("--load-analyzed", type=str)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument(
        "--splits", nargs="+", default=["a", "b", "c", "d", "e", "f", "g", "h"]
    )
    args = parser.parse_args()

    if args.load_analyzed:
        with open(args.load_analyzed, "rb") as f:
            dataset_summaries = pickle.load(f)
    else:
        dataset_summaries = {
            dataset: load_data_and_make_hit_results(
                os.path.join(args.data_directory, dataset), limit_load=None
            )
            for dataset in args.datasets
        }

    print("\\begin{table*}[ht]")
    print("\\centering")
    for i, split in enumerate(args.splits):
        print("% {split}")
        print("\\subfloat[]{Split " + split.upper() + "}{")
        print("\\centering")
        print("\\resizebox{0.4\\linewidth}{!}{")
        print(
            extract_split_to_table(dataset_summaries, split)[
                [NAME_MAP.get(n, n) for n in args.datasets]
            ]
            .loc[INDEX_COLS]
            .to_latex(float_format="%.3f", escape=False)
        )
        print("}")
        print("}")
        print("\\qquad" if i % 2 == 0 else "\\vskip 10mm")
    print("\\caption{Property statistics on all gSCAN test splits}")
    print("\\label{tab:gscan_split_properties}")
    print("\\end{table*}")


if __name__ == "__main__":
    main()
