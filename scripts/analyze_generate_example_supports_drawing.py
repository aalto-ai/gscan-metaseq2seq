import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from gscan_metaseq2seq.util.load_data import load_data_directories

from gscan_metaseq2seq.util.solver import (
    create_vocabulary,
    create_world,
    demonstrate_command_oracle,
    state_to_situation,
    reinitialize_world,
)

def instruction_is_correct(
    encoded_instruction,
    encoded_state,
    encoded_targets,
    word2idx,
    action2idx,
    color_dictionary,
    noun_dictionary,
    world,
    vocabulary,
):
    encoded_instruction = encoded_instruction[encoded_instruction != word2idx["[pad]"]]
    encoded_targets = encoded_targets[encoded_targets != action2idx["[pad]"]]
    encoded_state = [s for s in encoded_state if not (s == 0).all()]

    instr, situation = state_to_situation(
        encoded_instruction, encoded_state, word2idx, color_dictionary, noun_dictionary
    )

    oracle_actions = [
        action2idx[a]
        for a in demonstrate_command_oracle(
            world,
            vocabulary,
            color_dictionary,
            noun_dictionary,
            instr,
            situation.target_object,
            situation,
        )
        + ["[eos]"]
    ]
    dataset_actions = encoded_targets.tolist()

    return oracle_actions == dataset_actions, situation.target_object is not None


REMAP_ACTIONS = {
    "pull": "PULL",
    "push": "PUSH",
    "walk": "WALK",
    "turn left": "LTURN",
    "turn right": "RTURN",
    "stay": "STAY",
    "[eos]": "",
}


def compress_tokens(tokens):
    count = 0
    current_token = None
    for token in tokens:
        if current_token is not None and current_token != token:
            yield f"{current_token}" + ("" if count == 1 else f"({count})")
            count = 0

        current_token = token
        count += 1

    if current_token is not None and current_token != token:
        yield f"{current_token}" + ("" if count == 1 else f"({count})")


def which_color_box(correct, valid, relevant):
    if not valid:
        return "black"

    if relevant and correct:
        return "green"

    if not relevant and correct:
        return "yellow"

    if relevant and not correct:
        return "orange"

    return "red"


def compare_objects(left, right):
    print(left, right)
    return (left.color == right.color) and (left.shape == right.shape)


def plot_at_index(
    examples,
    dataset_name,
    index,
    output_dir,
    word2idx,
    action2idx,
    color_dictionary,
    noun_dictionary,
    analyze_if_correct=False,
    render_supports=False
):
    idx2word = [w for w in word2idx]
    idx2action = [w for w in action2idx]
    vocabulary = create_vocabulary()
    world = create_world(vocabulary)

    instr, situation = state_to_situation(
        examples[index][0],
        examples[index][2],
        word2idx,
        color_dictionary,
        noun_dictionary,
        need_target=analyze_if_correct
    )
    query_actions = [idx2action[a] for a in examples[index][1]]
    query_target_object = situation.target_object

    world = reinitialize_world(world, situation, vocabulary)

    plt.imshow(world.render(mode="rgb_array"))
    plt.axis("off")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_{index}.pdf"))
    plt.clf()

    print(index)
    print(" ".join([w for w in instr if w != "[pad]"]))
    print(" ".join([w for w in query_actions if w != "[pad]"]))

    support_states = examples[index][3]
    support_states = (
        [support_states] * len(examples[index][-3])
        if isinstance(support_states[0], np.ndarray)
        else support_states
    )

    corrects_and_valids = [
        instruction_is_correct(
            demo_instr,
            demo_state,
            demo_acts,
            word2idx,
            action2idx,
            color_dictionary,
            noun_dictionary,
            world,
            vocabulary,
        ) if analyze_if_correct else (False, False)
        for demo_instr, demo_acts, demo_state in zip(
            examples[index][-3], examples[index][-2], support_states
        )
    ]
    relevants = [
        compare_objects(
            query_target_object.object,
            state_to_situation(
                demo_instr, demo_state, word2idx, color_dictionary, noun_dictionary
            )[1].target_object.object,
        ) if analyze_if_correct else False
        for demo_instr, demo_state in zip(examples[index][-3], support_states)
    ]

    sorted_instrs = [
        ((demo_instr, demo_acts, demo_state, demo_score), (correct, valid), relevant)
        for (demo_instr, demo_acts, demo_state, demo_score), (correct, valid), relevant in sorted(
            list(
                zip(
                    zip(examples[index][-3], examples[index][-2], support_states, examples[index][-1]),
                    corrects_and_valids,
                    relevants,
                )
            ),
            key=lambda x: (~x[-1], ~x[-2][1], ~x[-2][0], -x[-3][-1]),
        )
    ]

    for (
        (demo_instr, demo_acts, demo_state, demo_score),
        (correct, valid),
        relevant,
    ) in sorted_instrs:
        print(
            " ".join([idx2word[w] for w in demo_instr if w != word2idx["[pad]"]])
            + " "
            + " ".join(map(str, [relevant, valid, correct]))
            + str(demo_score)
        )
        print(
            " -- "
            + " ".join(
                compress_tokens(
                    [
                        REMAP_ACTIONS[idx2action[w]]
                        for w in demo_acts
                        if w != action2idx["[pad]"]
                    ]
                )
            )
        )

    print(
        r"""\begin{figure*}[ht]
    \resizebox{\textwidth}{!}{
    \begin{tikzpicture}[
        title/.style={font=\fontsize{6}{6}\color{black!50}\ttfamily},
        node distance = 10mm, 
    ]
    \node [fill=black!10,rounded corners, inner sep=3pt] (query) {
        \begin{tikzpicture}[node distance=0mm]
            \node [anchor=west] (title) {\footnotesize{Query}};
            \node [anchor=north, below=of title.south] (state)
        {\includegraphics[width=.35\textwidth,keepaspectratio=1,trim={4cm 1cm, 3.5cm 1cm},clip]{img/figures/"""
        + f"{dataset_name}_{index}.pdf"
        + r"""}};
            \node[inner sep=0pt, below=of state.south] (query)
        {\footnotesize{$I^Q$ = ``"""
        + " ".join([w for w in instr if w != "[pad]"])
        + r"""\"}};
        \end{tikzpicture}
    };
    \node [fill=black!10, rounded corners, inner sep = 5pt, right=of query.east, minimum width=50mm, anchor=south, rotate=90] (ig) {Instruction Generator};
    """
        + (
            r"\node [fill="
            + which_color_box(*sorted_instrs[0][1], sorted_instrs[0][2])
            + r"!10, rounded corners, inner sep = 5pt, right= 7mm of ig.east, anchor=west, yshift=7mm] (i1) {\footnotesize{$I_1$ = ``"
            + " ".join(
                [idx2word[w] for w in sorted_instrs[0][0][0] if w != word2idx["[pad]"]]
            )
            +
            " ({:0.2f}) ".format(np.exp(sorted_instrs[0][-1]))
            +
            "}};\n"
        ) + (
            "\n".join(
                [
                    r"\node [fill="
                    + which_color_box(correct, valid, relevant)
                    + r"!10, rounded corners, inner sep = 5pt, below=7mm of i"
                    + str(i + 1)
                    + r".west, anchor=west] (i"
                    + str(i + 2)
                    + r") {\footnotesize{$I_{"
                    + str(i + 2) + "}"
                    + "$ = ``"
                    + (
                        " ".join(
                            [idx2word[w] for w in demo_instr if w != word2idx["[pad]"]]
                        )
                        +
                        " ({:0.2f}) ".format(np.exp(demo_score))
                        +
                        "}"
                    ) + "};"
                    for i, (
                        (demo_instr, demo_acts, demo_state, demo_score),
                        (correct, valid),
                        relevant,
                    ) in enumerate(sorted_instrs[1:])
                ]
            )
        )
        + r"""
    \node [fill=black!10, rounded corners, inner sep = 5pt, right=9.5cm of query.east, minimum width=50mm, anchor=south, rotate=90] (at) {Transformer};
    """
        + (
            r"\node [fill="
            + which_color_box(*sorted_instrs[0][1], sorted_instrs[0][2])
            + r"!10, rounded corners, inner sep = 5pt, right= 7mm of at.east, anchor=west, yshift=7mm] (a1) {\footnotesize{$A_1$ = ``"
            + " ".join(
                compress_tokens(
                    [
                        REMAP_ACTIONS[idx2action[w]]
                        for w in sorted_instrs[0][0][1]
                        if w != action2idx["[pad]"]
                    ]
                )
            )
            + '"}};\n'
        )
        + (
            "\n".join(
                [
                    r"\node [fill="
                    + which_color_box(correct, valid, relevant)
                    + r"!10, rounded corners, inner sep=5pt, below=7mm of a"
                    + str(i + 1)
                    + r".west, anchor=west] (a"
                    + str(i + 2)
                    + r") {\footnotesize{$A_{"
                    + str(i + 2) + "}"
                    + "$ = ``"
                    + (
                        " ".join(
                            compress_tokens(
                                [
                                    REMAP_ACTIONS[idx2action[w]]
                                    for w in demo_acts
                                    if w != action2idx["[pad]"]
                                ]
                            )
                        )
                        + "}};"
                    )
                    for i, (
                        (demo_instr, demo_acts, demo_state, demo_score),
                        (correct, valid),
                        relevant,
                    ) in enumerate(sorted_instrs[1:])
                ]
            )
        )
        + r"""
    \end{tikzpicture}
    }
    \end{figure*}"""
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--data-directory", required=True)
    parser.add_argument("--dictionary", required=True)
    parser.add_argument("--img-output-directory", required=True)
    parser.add_argument("--limit-load", type=int)
    parser.add_argument("--split", required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--analyze-if-correct", action="store_true")
    args = parser.parse_args()

    (
        (
            WORD2IDX,
            ACTION2IDX,
            color_dictionary,
            noun_dictionary,
        ),
        (meta_train_demonstrations, meta_valid_demonstrations_dict),
    ) = load_data_directories(
        args.data_directory, args.dictionary, limit_load=args.limit_load, only_splits=[args.split]
    )

    color_dictionary = sorted(color_dictionary)
    noun_dictionary = sorted(noun_dictionary)

    plot_at_index(
        meta_valid_demonstrations_dict[args.split],
        args.dataset_name,
        args.index,
        args.img_output_directory,
        WORD2IDX,
        ACTION2IDX,
        color_dictionary,
        noun_dictionary,
        analyze_if_correct=args.analyze_if_correct
    )


if __name__ == "__main__":
    main()
