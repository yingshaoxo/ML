import random

def get_similarity_score_of_two_sentence_by_position_match(sentence1: str, sentence2: str) -> float:
    sentence1_length = len(sentence1)
    sentence2_length = len(sentence2)
    base_sentence = None
    another_sentence = None
    if sentence1_length <= sentence2_length:
        base_sentence = sentence1
        another_sentence = sentence2
    else:
        base_sentence = sentence2
        another_sentence = sentence1

    counting = 0
    for index, char in enumerate(base_sentence):
        another_sentence_char = another_sentence[index]
        if char == another_sentence_char:
            counting += 1

    return counting / len(base_sentence)


def get_random_number_string(length: int) -> str:
    return ''.join(random.choice("0123456789") for _ in range(length))


def get_regex_version_of_keys_and_values(a_dict: dict[str, str]) -> dict[str, str]:
    """
    a_dict = {
        "That is steven, my uncle.": "I see, steven is your uncle.",
        "That is wind_god, my uncle.": "I see, wind_god is your uncle.",
    }

    it will return:
        {
            "That is (?P<name>.*?), my uncle.": "I see, {name} is your uncle.",
        }
    """
    items = list(a_dict.items())
    items.sort(key=lambda x: len(x[0]))

    base_sentence = items[0]
    another_sentence = items[1]

    if base_sentence[0] == another_sentence[0]:
        return {base_sentence[0]: base_sentence[1]}

    start_index = 0
    for index, char in enumerate(base_sentence[0]):
        another_sentence_char = another_sentence[0][index]
        if char == another_sentence_char:
            pass
        else:
            start_index = index
            break

    end_index = len(base_sentence[0])
    for index, char in enumerate(reversed(another_sentence[0])):
        another_sentence_char = base_sentence[0][-index-1]
        if char == another_sentence_char:
            pass
        else:
            index = len(base_sentence[0]) - index
            end_index = index
            break

    the_fuzz_keyword = base_sentence[0][start_index: end_index]
    if len(the_fuzz_keyword) == 0:
        return a_dict

    random_string = get_random_number_string(4)
    return {
        base_sentence[0][:start_index] + f"(?P<{random_string}>.*?)" + base_sentence[0][end_index:] :
            base_sentence[1].replace(base_sentence[0][start_index: end_index], f"{{{random_string}}}")
    }

#result = get_regex_version_of_keys_and_values({
#    "That is steven, my uncle.": "I see, steven is your uncle.",
#    "That is wind_god, my uncle.": "I see, wind_god is your uncle.",
#})
result = get_regex_version_of_keys_and_values({
"Did you see ?": "I see AA.",
"Did you see akj?": "I see CC.",
})
print(result)
exit()

"""
We will use char level operation to get unknown keywords regex from "multiple key -> one value" data pairs

Hi AA -> Hi you.
Hi BB -> Hi you.
Hi CC -> Hi you.

We need to get "Hi (.*?) -> Hi you." from above data automatically.



Did you see AA? => I see AA.
Did you see BB? => I see BB.

We need to get "Did you see (?P<someone>.*?)? -> I see {someone}." from above data automatically.



That is steven, my uncle. => I see, steven is your uncle.
That is wind_god, my uncle. => I see, wind_god is your uncle.

We need to get "That is (?P<name>.*?), my uncle. => I see, {name} is your uncle." from above data automatically.
"""
