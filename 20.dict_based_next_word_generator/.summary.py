from auto_everything.ml import ML
from auto_everything.disk import Disk
ml = ML()
disk = Disk()

text_generator = ml.Yingshaoxo_Text_Generator()
text_preprocessor = ml.Yingshaoxo_Text_Preprocessor()
#text = text_generator.get_source_text_data_by_using_yingshaoxo_method(input_txt_folder_path="../18.fake_ai_asistant/input_txt_files", type_limiter=[".txt"])
#text = text.replace("\n\n\n__**__**__yingshaoxo_is_the_top_one__**__**__\n\n\n", "")


def get_word_list_from_string(text: str, include_punctuation: bool = False) -> list[str]:
    final_list = []

    sentence_segment_list = text_preprocessor.split_string_into_list_by_punctuations(text, not_include_punctuations="\n' _")
    for segment in sentence_segment_list:
        if segment["language"] == "punctuation":
            if include_punctuation == False:
                continue
            else:
                final_list += list(segment["text"])
                continue

        segment = segment["text"].strip()
        if segment == "":
            continue

        segment_list = []
        if text_preprocessor.is_english_string(segment):
            segment_list += segment.split(" ")
        else:
            segment_list += list(segment)

        segment_list = [one.strip() for one in segment_list]
        final_list += segment_list

    return final_list


def get_meaning_group_source_list_in_char_level_low_speed_version(source_text: str, frequency_level: int = 2, return_raw_list: bool = False, return_raw_dict: bool = False, raw_dict: dict|None = None) -> list[str] | dict[str, int]:
    if raw_dict == None:
        raw_dict = {}

    for length in range(1, len(source_text)+1):
        for index, _ in enumerate(source_text):
            if index + length > len(source_text):
                break
            part = source_text[index: index + length]
            if part in raw_dict.keys():
                raw_dict[part] += 1
            else:
                raw_dict[part] = 1

    killed_list = sorted(raw_dict.items(), key=lambda x: x[1], reverse=True)
    for one in killed_list:
        if one[1] < frequency_level:
            del raw_dict[one[0]]

    if return_raw_dict == True:
        return raw_dict
    else:
        return_list = []
        return_set = set()
        for length in range(1, len(source_text)+1):
            for index, _ in enumerate(source_text):
                if index + length > len(source_text):
                    break
                part = source_text[index: index + length]
                if part in raw_dict.keys():
                    if part not in return_set:
                        return_set.add(part)
                        if return_raw_list == True:
                            return_list.append([part, raw_dict[part]])
                        else:
                            return_list.append(part)
        return return_list


def split_meaning_group_from_text(input_text: str, meaning_group_source_list: list[str]) -> list[str]:
    meaning_group_source_list.sort(key=len, reverse=True)
    pass


text = """
I'm yingshaoxo.
I'm yingshaoxo.
"""
print(get_meaning_group_source_list_in_char_level_low_speed_version(text[:3000], frequency_level=2, return_raw_list=True))
exit()




"""
To achive following result, you have to have a global meaning dict first, so that you could seperate text in a more accurate way


For one sentence summary:
1. first define a dict = {longest_sub_string: ['a': importance_rate, 'b': importance_rate], second_long_sub_string: ...}. the importance rate depends on if you can find that word in the following text. (The sub string is meaning group words) (the importance rate here is global version, has to get from all dataset)
2. for a input_text, "based on what you say", seperate it into meaning group or sentence_segment, if you find that segment in dict, you count its word importance rate, if the rate below than average rate of those meaning_group in that sentence, you remove it
"""


"""
Text Summary:

Remain those meaning_group_words appears most times, especially for those appears in following text, because they are important for the sentence flow to go on
"""


"""
Text Classification:

You have (input_text, target_tag) dataset, you split input_text into meaning_group, save it in a list[['meaning_group1', 'meaning_group1'], target_tag]

If you meet new text that happen to have meaning group that could match one of the key in sequence (if that is the longest key), then it is the target_tag.

Or you could use sentence similarity to do the classification.
"""


"""
Process logic:
Article -> paragraph(new line seperated) -> sentence -> sentence_fragment -> meaning group -> words -> char
"""


'''
def suffix_array(s):
    suffixes = [(s[i:], i) for i in range(len(s))]
    suffixes.sort()
    return [suffix[1] for suffix in suffixes]

def lcp_array(s, suffix_array):
    rank = [0] * len(s)
    for i in range(len(s)):
        rank[suffix_array[i]] = i

    lcp = [0] * (len(s) - 1)
    k = 0
    for i in range(len(s)):
        if rank[i] == len(s) - 1:
            k = 0
            continue
        j = suffix_array[rank[i] + 1]
        while i + k < len(s) and j + k < len(s) and s[i+k] == s[j+k]:
            k += 1
        lcp[rank[i]] = k
        if k > 0:
            k -= 1
    return lcp

def repeated_substrings(text, repeat_count):
    full_text = text + "$"
    sa = suffix_array(full_text)
    lcp = lcp_array(full_text, sa)

    result = set()
    for i in range(len(text)):
        if sa[i] < len(text) and (lcp[i] >= len(text) - sa[i] or lcp[i] >= len(text) - sa[i + 1]):
            length = lcp[i]
            repeated_substring = text[sa[i]:sa[i]+length]
            if repeated_substring not in result and text.count(repeated_substring) == repeat_count:
                result.add(repeated_substring)

    return result

repeat_count = 2
repeated_substrings_found = repeated_substrings(text, repeat_count)
print(f"Repeated substrings repeated {repeat_count} times in the text: {repeated_substrings_found}")
exit()
'''
