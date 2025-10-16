from typing import Optional, Union, Dict, List, Tuple

from gscan_metaseq2seq.gscan.world import (
    Situation,
    ObjectVocabulary,
    World,
    INT_TO_DIR,
    Object,
    PositionedObject,
    Position,
)
from gscan_metaseq2seq.gscan.grammar import Derivation
from gscan_metaseq2seq.gscan.vocabulary import Vocabulary

GRID_SIZE = 6
MIN_OTHER_OBJECTS = 0
MAX_OBJECTS = 2
MIN_OBJECT_SIZE = 1
MAX_OBJECT_SIZE = 4
OTHER_OBJECTS_SAMPLE_PERCENTAGE = 0.5

TYPE_GRAMMAR = "adverb"
INTRANSITIVE_VERBS = "walk"
TRANSITIVE_VERBS = "pull,push"
ADVERBS = "cautiously,while spinning,hesitantly,while zigzagging"
NOUNS = "square,cylinder,circle"
COLOR_ADJECTIVES = "red,green,yellow,blue"
SIZE_ADJECTIVES = "big,small"


def create_vocabulary(colors=None, sizes=None, nouns=None):
    vocabulary = Vocabulary.initialize(
        intransitive_verbs=(INTRANSITIVE_VERBS).split(","),
        transitive_verbs=(TRANSITIVE_VERBS).split(","),
        adverbs=(ADVERBS).split(","),
        nouns=(nouns or NOUNS).split(","),
        color_adjectives=(colors or COLOR_ADJECTIVES).split(","),
        size_adjectives=(sizes or SIZE_ADJECTIVES).split(","),
    )
    return vocabulary


def create_world(
    vocabulary,
    grid_size: Optional[int] = None,
    min_object_size: Optional[int] = None,
    max_object_size: Optional[int] = None,
    type_grammar: Optional[str] = None,
):

    # Object vocabulary.
    object_vocabulary = ObjectVocabulary(
        shapes=vocabulary.get_semantic_shapes(),
        colors=vocabulary.get_semantic_colors(),
        min_size=min_object_size or MIN_OBJECT_SIZE,
        max_size=max_object_size or MAX_OBJECT_SIZE,
    )

    # Initialize the world.
    return World(
        grid_size=grid_size or GRID_SIZE,
        colors=vocabulary.get_semantic_colors(),
        object_vocabulary=object_vocabulary,
        shapes=vocabulary.get_semantic_shapes(),
        save_directory=None,
    )


def reinitialize_world(
    world,
    situation: Situation,
    vocabulary,
    mission="",
    manner=None,
    verb=None,
    end_pos=None,
    required_push=0,
    required_pull=0,
    num_instructions=0,
):
    objects = []
    for positioned_object in situation.placed_objects:
        objects.append((positioned_object.object, positioned_object.position))
    world.initialize(
        objects,
        agent_position=situation.agent_pos,
        agent_direction=situation.agent_direction,
        target_object=situation.target_object,
        carrying=situation.carrying,
    )
    if mission:
        is_transitive = False
        if verb in vocabulary.get_transitive_verbs():
            is_transitive = True
        world.set_mission(
            mission,
            manner=manner,
            verb=verb,
            is_transitive=is_transitive,
            end_pos=end_pos,
            required_push=required_push,
            required_pull=required_pull,
            num_instructions=num_instructions,
        )

    return world


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
    color_words = [word2idx[w] for w in list(colors) if w is not None]
    noun_words = [word2idx[w] for w in list(nouns) if w in word2idx if w is not None]

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


def state_to_situation(
    query_instruction, state, word2idx, colors, nouns, need_target=True
):
    idx2word = [w for w in word2idx if w != word2idx["[pad]"]]

    if need_target:
        verb, adverb, size, color, noun = segment_instruction(
            query_instruction, word2idx, colors, nouns
        )
        target_object = find_target_object(
            state, size, color, noun, idx2word, colors[1:], nouns[1:]
        )
    else:
        target_object = None

    agent = find_agent_position(state)
    return (
        [idx2word[w] for w in query_instruction],
        Situation(
            grid_size=6,
            agent_position=Position(agent[-1], agent[-2]),
            agent_direction=INT_TO_DIR[agent[-3] - 1],
            target_object=None
            if target_object is None
            else PositionedObject(
                object=Object(
                    shape=nouns[target_object[2]],
                    color=colors[target_object[1]],
                    size=target_object[0],
                ),
                position=Position(target_object[-1], target_object[-2]),
                vector=[],
            ),
            placed_objects=[
                PositionedObject(
                    object=Object(
                        shape=nouns[o[2]], color=colors[o[1]], size=o[0]
                    ),
                    position=Position(o[-1], o[-2]),
                    vector=[],
                )
                for o in state[1:]
                if not (o[:3] == 0).all()
            ],
        ),
    )


def demonstrate_command_oracle(
    world,
    vocabulary,
    vocabulary_colors,
    vocabulary_nouns,
    command,
    target_object,
    initial_situation,
):
    """
    Demonstrate a command derivation and situation pair. Done by extracting the events from the logical form
    of the command derivation, extracting the arguments of each event. The argument of the event gets located in the
    situation of the world and the path to that target gets calculated. Based on whether the verb in the command is
    transitive or not, the agent interacts with the object.
    :param derivation:
    :param initial_situation:
    :returns
    """
    action_words = []
    article_words = []
    description_words = []
    adverb_words = []

    if not target_object:
        return []

    for w in command:
        if w in ["walk", "to", "push", "pull"]:
            action_words.append(w)

        if w in ["a"]:
            article_words.append(w)

        if w in vocabulary_colors + vocabulary_nouns + ["big", "small"]:
            description_words.append(w)

        if w in ["while spinning", "while zigzagging", "hesitantly", "cautiously"]:
            adverb_words.append(w)

    # Initialize the world based on the initial situation and the command.
    reinitialize_world(world, initial_situation, vocabulary, mission=command)

    # Our commands are quite simple
    manner = adverb_words[0] if adverb_words else ""
    world.go_to_position(
        position=target_object.position,
        manner=manner,
        primitive_command="walk",
    )

    # Then if the primitive command is push or pull, we have to move the object to the wall
    if action_words == ["pull"] or action_words == ["push"]:
        world.move_object_to_wall(action=action_words[0], manner=manner)

    # Done
    target_commands, _ = world.get_current_observations()
    return target_commands
