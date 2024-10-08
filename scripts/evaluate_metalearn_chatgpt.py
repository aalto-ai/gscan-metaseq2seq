import argparse
import openai
from openai import OpenAI
import json
import itertools
import functools
import tqdm
import re
import numpy as np
import pprint
import os
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential
)
from gscan_metaseq2seq.util.load_data import load_data_directories

@retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(3))
def completion_with_backoff(client, **kwargs):
    return client.chat.completions.create(**kwargs)


def encode_individual_state(s, idx2color, idx2object):
    size, color, obj, agent, direction, x, y = s

    if agent == 1:
        return f"agent d: {direction} x: {x} y: {y}"

    return f"{idx2color[color - 1]} {idx2object[obj - 1]} s: {size} x: {x} y: {y}"


def encode_state(state, idx2color, idx2object):
    return ", ".join(map(lambda s: encode_individual_state(s, idx2color, idx2object),
                         filter(lambda s: s[-1] != 0, state)))


def encode_instr(instr, idx2word, word2idx):
    return " ".join([idx2word[w] for w in instr if w != word2idx['[pad]']])


def encode_targets(targets, idx2action, action2idx):
    return " ".join([idx2action[w] for w in targets if w != action2idx['[pad]']])


def convert_to_text_representation(example, idx2word, idx2action, word2idx, action2idx, idx2color, idx2object, max_examples=16, reverse=False):
    query_instr, query_targets, query_state, support_states, support_instrs, support_targets, scores = example

    scores = np.array(scores)
    rev_stride = -1 if reverse else 1

    # highest scores go last
    support_instrs = [support_instrs[i] for i in scores.argsort()[::rev_stride]]
    support_targets = [support_targets[i] for i in scores.argsort()[::rev_stride]]

    if isinstance(support_states[0], np.ndarray):
        state_line = f"State: {encode_state(query_state, idx2color, idx2object)}\n"
        return ("Complete based on the following. Base the answer on Inputs Output pairs that are relevant to the Query Input:\n" + "\n".join([
            f"Input: {encode_instr(instr, idx2word, word2idx)}\nOutput: {encode_targets(targets, idx2action, action2idx)}"
            for instr, targets in zip(
                support_instrs[-max_examples:],
                support_targets[-max_examples:]
            )
        ]) + f"\nQuery Input: {encode_instr(query_instr, idx2word, word2idx)}\nOutput:", encode_targets(query_targets, idx2action, action2idx))

    support_states = [support_states[i] for i in scores.argsort()[::rev_stride]]

    return ("Complete based on the following. Base the answer on Input Output pairs that are relevant to the Query Input:\n" + "\n".join([
        f"State: {encode_state(state, idx2color, idx2object)}\nInput: {encode_instr(instr, idx2word, word2idx)}\nOutput: {encode_targets(targets, idx2action, action2idx)}"
        for state, instr, targets in zip(
            support_states[-max_examples:],
            support_instrs[-max_examples:],
            support_targets[-max_examples:]
        )
    ]) + f"\nState: {encode_state(query_state, idx2color, idx2object)}\nQuery Input: {encode_instr(query_instr, idx2word, word2idx)}\nOutput:", encode_targets(query_targets, idx2action, action2idx))


def print_through(val):
    tqdm.tqdm.write(pprint.pformat(val))
    return val


def stream_json_list_to_file(iterable, f, append=False):
    if not append:
        f.write("[\n  ")
    for i, element in enumerate(iterable):
        if i != 0 or append:
            f.write(",\n  ")
        json.dump(element, f)
    f.write("]\n")


def try_json_parse(line):
    try:
        return json.loads(line.strip().rstrip(","))
    except json.decoder.JSONDecodeError as e:
        import pdb
        pdb.set_trace()
        return None


def check_already_done(path):
    try:
        with open(path, "r") as f:
            file_lines = f.readlines()
    except FileNotFoundError:
        return []

    import pdb
    pdb.set_trace()

    return map(lambda obj: obj["orig_instruction"],
               filter(lambda x: x,
                      map(lambda line: try_json_parse(line),
                          file_lines)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", type=str, required=True, help="OpenAI API key")
    parser.add_argument("--dataset", type=str, required=True, help="List of input sentences")
    parser.add_argument("--output-responses", type=str, required=True, help="Where to write chatgpt responses")
    parser.add_argument("--only-splits", nargs="*")
    parser.add_argument("--limit-load", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model", choices=("gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"), default="gpt-3.5-turbo")
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--limit-examples", type=int, default=8)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()

    client = openai.OpenAI(api_key=args.api_key)

    (
        dictionaries,
        (train_demonstrations, valid_demonstrations_dict),
    ) = load_data_directories(
        args.dataset,
        os.path.join(args.dataset, "dictionary.pb"),
        limit_load=args.limit_load,
        only_splits=args.only_splits
    )

    word2idx, action2idx, idx2color, idx2object = dictionaries
    idx2word = list(word2idx.keys())
    idx2action = list(action2idx.keys())
    idx2action = ['pull', 'push', 'stay', 'lturn', 'rturn', 'walk', '[pad]', '[sos]', '[eos]']
    #idx2action = list(map(str, range(len(idx2action))))
    #idx2action[action2idx['[pad]']] = '[pad]'
    #action2idx = {a: i for i, a in enumerate(idx2action)} # 

    with open(args.output_responses, "w") as f:
        stream_json_list_to_file(
            itertools.chain.from_iterable(
                map(
                    lambda split_examples: map(
                        lambda x: print_through(
                            (
                                split_examples[0],
                                [
                                    m.message.content
                                    for m in completion_with_backoff(
                                        client,
                                        model=args.model,
                                        messages=[{"role": "user", "content": print_through(x[0])}],
                                        temperature=0,
                                        max_tokens=args.max_output_tokens,
                                        n=1
                                    ).choices
                                ],
                                x[1]
                            )
                        ),
                        map(lambda e: convert_to_text_representation(e, idx2word, idx2action, word2idx, action2idx, idx2color, idx2object, args.limit_examples, args.reverse), split_examples[1][:args.limit])
                    ),
                    valid_demonstrations_dict.items()
                )
            ),
            f
        )


if __name__ == "__main__":
    main()