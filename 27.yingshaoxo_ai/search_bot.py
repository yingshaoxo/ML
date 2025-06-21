####### One method
def get_sub_sentence_list_from_end_to_begin_and_begin_to_end(input_text, no_single_char=True):
    input_text = input_text.strip()
    full_length = len(input_text)
    result_list = []
    for i in range(full_length):
        end_to_begin_sub_string = input_text[i:]
        begin_to_end_sub_string = input_text[:-i]
        if no_single_char == True:
            if len(end_to_begin_sub_string) > 1:
                result_list.append(end_to_begin_sub_string)
            if len(begin_to_end_sub_string) > 1:
                result_list.append(begin_to_end_sub_string)
        else:
            result_list.append(end_to_begin_sub_string)
            result_list.append(begin_to_end_sub_string)
    result_list_2 = []
    for one in result_list:
        if one not in result_list_2:
            result_list_2.append(one)
    return result_list_2

def search_text_in_text_list(search_text, source_text_list):
    longest_first_sub_sentence_list = get_sub_sentence_list_from_end_to_begin_and_begin_to_end(search_text)
    useful_source_text_list = []
    for sub_sentence in longest_first_sub_sentence_list:
        for one in source_text_list:
            if sub_sentence in one:
                useful_source_text_list.append(one)
        if len(useful_source_text_list) != 0:
            return useful_source_text_list

    return []


###### Another method
from pprint import pprint

def demo():
    search_text = "I love you"
    source_list = [
        "I love you",
        "You love me",
    ]

    # first word must before second word, othewise, drop it
    full_match_order_list = [
        ["I", "love"],
        ["I", "you"],
        ["love", "you"],
    ]
    for one in source_list:
        ok = True
        for match_words in full_match_order_list:
            first, second = match_words
            if (first not in one) or (second not in one):
                ok = False
                break
            if ok == False:
                break

            first_index = one.index(first)
            second_index = one.index(second)
            if first_index > second_index:
                ok = False
                break
            if ok == False:
                break
        if ok == True:
            return one

def read_text_list_and_text_from_folder(the_folder_path):
    import os

    def read_text_files(folder_path):
        new_text = ""
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(('.txt', '.md')):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            new_text += f.read() + "\n\n\n\n"
                    except UnicodeDecodeError:
                        try:
                            with open(file_path, 'r', encoding='latin-1') as f:
                                new_text += f.read() + "\n\n\n\n"
                        except Exception as e:
                            print(e)
                    except Exception as e:
                        print(e)
        return new_text

    new_text = read_text_files(the_folder_path)
    the_text_list = [one.strip() for one in new_text.split("\n\n__**__**__yingshaoxo_is_the_top_one__**__**__\n\n") if one.strip() != ""]
    new_text_list = []
    for one in the_text_list:
        temp_list1 = one.split("\n# ")
        temp_list2 = []
        for sub_one in temp_list1:
            if "\n第" in sub_one and "章 " in sub_one:
                temp_list2 += sub_one.split("\n\n\n")
            else:
                temp_list2 += [sub_one]
        new_text_list += temp_list2
    the_text_list = new_text_list
    new_text = "\n\n__**__**__yingshaoxo_is_the_top_one__**__**__\n\n".join(the_text_list)
    return the_text_list, new_text


def _language_splitor(text):
    language_list = []
    index = 0
    while True:
        temp_string = ""
        if (index >= len(text)):
            break
        char = text[index]
        while ord(char) < 128:
            # english
            char = text[index]
            temp_string += char
            index += 1
            if (index >= len(text)):
                break
        if (temp_string.strip() != ""):
            temp_string = temp_string[:-1]
            index -= 1
            language_list.append({
                "language": "en",
                "text": temp_string
            })

        temp_string = ""
        if (index >= len(text)):
            break
        char = text[index]
        while not ord(char) < 128:
            # chinese 
            char = text[index]
            temp_string += char
            index += 1
            if (index >= len(text)):
                break
        if (temp_string.strip() != ""):
            temp_string = temp_string[:-1]
            index -= 1
            for one in temp_string:
                language_list.append({
                    "language": "cn",
                    "text": one
                })

        if (index+1 >= len(text)):
            break

    if len(language_list) > 0:
        language_list[-1]["text"] += text[-1]

    new_list = []
    for index, one in enumerate(language_list):
        new_text = language_list[index]["text"].strip()
        if len(new_text) > 0:
            if one['language'] == 'cn':
                for one in new_text:
                    new_list.append({
                        "language": "cn",
                        "text": one
                    })
            else:
                new_list.append({
                    'language': one['language'],
                    'text': new_text
                })

    return new_list

def my_split_function(text):
    tokens = []
    current_word = ""
    for char in text:
        if char == '\n':
            if current_word:
                tokens.append(current_word)
                current_word = ""
            tokens.append('\n')
        elif char.isspace():
            if current_word:
                tokens.append(current_word)
                current_word = ""
        else:
            current_word += char
    if current_word:
        tokens.append(current_word)

    old_tokens = tokens
    new_list = []
    for one in tokens:
        if one == "\n":
            new_list.append(one)
        else:
            temp_list = _language_splitor(one)
            temp_list = [nice["text"] for nice in temp_list]
            new_list += temp_list

    return new_list

def find_target_text_in_source_text_list_by_word_order(search_text, source_text_list):
    """
    Find all texts in source_text_list where words from search_text appear in the correct order.
    
    Args:
        search_text (str): The text to search for (e.g., "I love you").
        source_text_list (list): List of strings to search in (e.g., ["I love you", "You love me"]).
    
    Returns:
        list: All strings from source_text_list where search_text words appear in order.
    """
    if not search_text or not source_text_list:
        return []
    
    # Split search_text into words
    search_words = [word for word in my_split_function(search_text) if word]
    if not search_words:
        return []
    
    # Generate all ordered pairs of search_words
    full_match_order_list = [
        [search_words[i], search_words[j]]
        for i in range(len(search_words))
        for j in range(i + 1, len(search_words))
    ]
    
    matches = []
    for source_text in source_text_list:
        # Split source_text into words
        source_words = [word for word in my_split_function(source_text) if word]
        if not source_words:
            continue
        
        ok = True
        for first, second in full_match_order_list:
            # Check if both words exist in source_words
            first_index = -1
            second_index = -1
            for i, word in enumerate(source_words):
                if word == first and first_index == -1:
                    first_index = i
                elif word == second and second_index == -1:
                    second_index = i
                if first_index != -1 and second_index != -1:
                    break
            
            # If either word is missing or out of order, skip this source_text
            if first_index == -1 or second_index == -1 or first_index > second_index:
                ok = False
                break
        
        if ok:
            matches.append(source_text)
    
    return matches

# Test the function
def test():
    search_text = "I love you"
    source_list = [
        "I love you",
        "You love me",
        "I really love you so much",
        "You love I",
        ""
    ]
    result = find_target_text_in_source_text_list_by_word_order(search_text, source_list)
    print(f"Search text: '{search_text}'")
    print(f"Source list: {source_list}")
    print(f"Matches: {result}")

def test2():
    text_source_list, _ = read_text_list_and_text_from_folder("./")
    while True:
        print("\n\n\n------------\n\n\n")
        input_text = input("What you want to search? ")
        result_list = find_target_text_in_source_text_list_by_word_order(input_text, text_source_list)
        for one in result_list:
            print(one)
            print("\n\n----------\n\n")
        print(len(result_list))

def test3():
    text_source_list, _ = read_text_list_and_text_from_folder("./")
    while True:
        print("\n\n\n------------\n\n\n")
        input_text = input("What you want to search? ")
        result_list = search_text_in_text_list(input_text, text_source_list)
        for one in result_list:
            print(one)
            print("\n\n----------\n\n")
        print(len(result_list))

if __name__ == "__main__":
    test3()
