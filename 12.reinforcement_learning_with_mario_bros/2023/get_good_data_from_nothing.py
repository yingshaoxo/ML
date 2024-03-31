"""
Add recent cache for not update dict search
"""
"""
CPU based reinforcement learning method:

You should let the bot run as far as you can. In the process, you save traning data in a list.

If that bot die, you record current distance as max_distance, then take the first 2/3(0.6 times max_distance) data for traning.

Based on that data, you run it again, but when you reach 2/3 max_distance long, you let bot use random actions.

If the bot run further than previous one, you still take 2/3(0.6 times max_distance) long path data for traning.

Otherwise, you set the max_distance = max_x_position * 0.9. So that you could collect more base data for early stage or unstable period.

#reinforcement_learning #algorithm #yingshaoxo
"""
from typing import Any
import random
import sys
import select
import termios
import tty
from time import sleep
import json
import functools

import cv2
import numpy as np
from PIL import Image

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
env = gym_super_mario_bros.make('SuperMarioBros-v2', apply_api_compatibility=True, render_mode="human")
MY_MARIO_MOVEMENT = [
    ['NOOP'],
    ['right', 'B'],
    #['right'],
    ['A'],
    ['left'],
]
env = JoypadSpace(env, MY_MARIO_MOVEMENT)

from auto_everything.disk import Disk
disk = Disk()

from auto_everything.string_ import String
string_ = String()

from auto_everything.time import Time
time_ = Time()

from auto_everything.database import Database_Of_Yingshaoxo
hash_database = Database_Of_Yingshaoxo(database_name="cpu_quick_mode", use_sqlite=False)
hash_database.refactor_database()

from auto_everything.ml import Yingshaoxo_Computer_Vision
yingshaoxo_computer_vision = Yingshaoxo_Computer_Vision()


images_dict = {}
image_keys = []
def one_row_dict_handler(data: dict[str, Any]) -> (dict[str, Any] | None):
    return data
def update_image_dict():
    global images_dict, image_keys
    images_dict = {image_data["id_"]:image_data for image_data in hash_database.search(one_row_dict_handler=one_row_dict_handler)}
    image_keys = list(images_dict.keys())
    for key in image_keys:
        images_dict[key]["data"] = images_dict[key]["data"]
    hash_database.refactor_database()
update_image_dict()


previous_frame = None
previous_frame_loaded = False
def image_process_backup(frame):
    global previous_frame, previous_frame_loaded
    side_length = 225
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (side_length, side_length)).astype(np.uint8)
    else:
        frame = np.zeros((3, side_length, side_length)) #may it is not a good idea to put 0 here, may cause other weights get into 0 quickly

    width, height = frame.shape[1], frame.shape[0]
    frame = frame[80:height, 0:width]

    #frame = cv2.resize(frame, (8, 8)).astype(np.uint8)
    frame = cv2.resize(frame, (16, 16)).astype(np.uint8)

    # only get the changed part
    #if previous_frame_loaded == False:
    #    previous_frame = frame
    #    previous_frame_loaded = True
    #empty_array = np.zeros_like(frame)
    #np.putmask(empty_array, frame != previous_frame, frame)

    #return empty_array

def image_process(frame):
    global previous_frame, previous_frame_loaded
    side_length = 225
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (side_length, side_length)).astype(np.uint8)
    else:
        frame = np.zeros((3, side_length, side_length)) #may it is not a good idea to put 0 here, may cause other weights get into 0 quickly

    width, height = frame.shape[1], frame.shape[0]
    frame = frame[80:height, 0:width]

    #frame = cv2.resize(frame, (8, 8)).astype(np.uint8)
    frame = cv2.resize(frame, (16, 16)).astype(np.uint8)

    return frame

def get_image_feature_vector_data(image) -> str:
    global sequence_length
    text = str(image.tolist())
    #hash_code = string_.get_fuzz_hash(text, level=512)
    #hash_code = string_.get_fuzz_hash(text, level=sequence_length)
    hash_code = string_.get_fuzz_hash(text, level=5, seperator="")
    return hash_code

def get_merged_feature(image, other_data) -> str:
    image_feature = get_image_feature_vector_data(image)
    #other_data = json.dumps(other_data)
    return other_data + "+" + image_feature
    #return image_feature

def get_similarity_between_two_numpy_array(array_a, array_b):
    # norm_a = np.linalg.norm(array_a)
    # norm_b = np.linalg.norm(array_b)
    # dot = sum(a * b for a, b in zip(array_a, array_b))
    # return (dot / (norm_a * norm_b))
    return string_.get_similarity_score_of_two_sentence_by_position_match(array_a, array_b)

def check_if_image_is_unique(image, other_data_dict) -> tuple[bool, str|None]:
    """
    Maybe you could have a global min and max value, then do a compare based on it
    """
    global images_dict
    original_image_data = get_merged_feature(image, other_data_dict)

    for image_data in images_dict.values():
        temp_data = image_data["data"]
        #print(temp_data)
        #print(original_image_data)
        similarity = string_.get_similarity_score_of_two_sentence_by_position_match(temp_data, original_image_data)
        #print("similarity: ", similarity)

        if similarity > 0.999:
            return False, image_data["id_"]

    return True, None

def get_char_input(use_timeout: bool = False) -> tuple[str, int]:
    #https://www.physics.udel.edu/~watson/scen103/ascii.html
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        if use_timeout:
            if select.select([sys.stdin,],[],[],0.1)[0]:
                char = sys.stdin.read(1)
            else:
                return "", -1
        else:
            char = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return char, ord(char)

@functools.lru_cache(maxsize=10000, typed=False)
def _get_most_possible_action(last_state_feature) -> tuple[str, str]:
    global images_dict, image_keys
    last_used_image_path = ""
    similarity_list = []
    for image_data_id in image_keys:
        image = images_dict[image_data_id]["data"]
        last_used_image_path = images_dict[image_data_id]["id_"]

        similarity = get_similarity_between_two_numpy_array(image, last_state_feature)
        #similarity = abs(int(image) - int(last_state_feature))
        action = images_dict[image_data_id]["action"]

        similarity_list.append([similarity, action])
    similarity_list.sort(key=lambda item: item[0], reverse=True)
    #print(similarity_list[:3])
    the_most_possible_one = similarity_list[0]
    #the_most_possible_one = random.choice(similarity_list[:3])
    #print("similarity:", str(the_most_possible_one[0])[:7])
    action = the_most_possible_one[1]
    return action, last_used_image_path

def get_most_possible_action_from_database(input_image, other_data_dict) -> tuple[str, str]:
    last_state_feature = get_merged_feature(input_image, other_data_dict)
    return _get_most_possible_action(last_state_feature)

def save_data_to_database(id_, image, additional_data, action):
    image_data = {
        "id_": id_,
        "data": get_merged_feature(image, additional_data),
        "action": action,
    }
    hash_database.add(image_data)
    #print(image_data)
    print("save")

def update_data_in_database(old_id, id_, image, additional_data, action):
    def one_row_dict_filter(item: dict[str, Any]) -> bool:
        if item["id_"] == old_id:
            return True
        return False
    hash_database.delete(one_row_dict_filter=one_row_dict_filter)

    image_data = {
        "id_": id_,
        "data": get_merged_feature(image, additional_data),
        "action": action,
    }
    hash_database.add(image_data)
    #print(image_data)
    print("update")

state, info = env.reset()
state = image_process(state)
continues_steps = 10

last_used_image_path = None

sequence_length = 5
continuse_states = [state] * sequence_length

def get_last_state():
    global continuse_states
    return np.concatenate(continuse_states, axis=0)

def generate_real_action_queue(action_list: list[int]):
    action_queue = []
    for one in action_list:
        action_queue += [int(one)] * continues_steps
    return action_queue


human_handle = True
action_queue = []
action = None
speed = 0
x_position_list = [0,0]
y_position_list = [0,0]
good_data_list = []
max_x_position = 10
processed_data_id_set = set()
while True:
    last_state = get_last_state()
    #additional_data = {"speed": speed, "y_position": "".join(y_position_list)} #, "x_position": "".join([str(one) for one in x_position_list])
    #str_speed = str(speed)[:3].rjust(3, '0') # padding for the right string
    #str_y_position = str(y_position_list[-1]).rjust(3, '0')
    #additional_data = str_speed + "+" + str_y_position
    additional_data = ""
    action = None

    """
    input_char, input_char_id = get_char_input(use_timeout=not human_handle)
    if input_char == '0':
        action = 0
        action_queue += [action] * continues_steps
    elif input_char == '1':
        action = 1
        action_queue += [action] * continues_steps
    elif input_char == '2':
        action = 2
        action_queue += [action] * continues_steps
    elif input_char == "6":
        # press 6 to let the bot run mario
        action = None
        human_handle = not human_handle
        if human_handle:
            sleep(1)
        continue
    elif input_char == "q":
        exit()
    elif input_char == "r":
        state, info = env.reset()
        continue
    else:
        action = None
    """

    if action == None:
        if len(image_keys) == 0:
            action = env.action_space.sample()
        else:
            if x_position_list[-1] < max_x_position*(0.7):
                if random.randint(0, 50) == 0:
                    action = env.action_space.sample()
                else:
                    action, last_used_image_path = get_most_possible_action_from_database(last_state, additional_data)
            else:
                if random.randint(0, 100) == 0:
                    action, last_used_image_path = get_most_possible_action_from_database(last_state, additional_data)
                else:
                    action = env.action_space.sample()

    action_queue = generate_real_action_queue([action])

    the_state = None
    the_info = {}
    for index, the_action in enumerate(action_queue):
        state, reward, terminated, truncated, info = env.step(the_action) # type: ignore
        done = terminated or truncated or (info["time"] == 0)
        if done:
            state, info = env.reset()
            if "life" in info:
                if info["life"] == 3:
                    # train
                    print("learned")
                    for one in good_data_list:
                        if one[0] in processed_data_id_set:
                            continue
                        else:
                            processed_data_id_set.add(one[0])
                        is_unique, similar_image_data_id = check_if_image_is_unique(one[1], one[2])
                        if is_unique:
                            save_data_to_database(*one)
                        else:
                            update_data_in_database(similar_image_data_id, *one)
                    update_image_dict()
                else:
                    os.system("clear")
                    print("died, clean")
                    good_data_list = []
                    processed_data_id_set = set()

                    max_x_position = max_x_position * 0.9
        if index == 0:
            state = image_process(state)
            the_state = state
            the_info = info

    if "score" in the_info:
        if the_info["score"] > 0:
            new_image_data_id = str(time_.get_current_timestamp_in_10_digits_format())
            good_data_list.append([new_image_data_id, last_state, additional_data, action])
    #new_image_data_id = str(time_.get_current_timestamp_in_10_digits_format())
    #good_data_list.append([new_image_data_id, last_state, additional_data, action])
    """
    if human_handle == True:
        is_unique, similar_image_data_id = check_if_image_is_unique(last_state, additional_data)
        new_image_data_id = str(time_.get_current_timestamp_in_10_digits_format())
        if is_unique:
            #save_data_to_database(new_image_data_id, last_state, additional_data, action)
            good_data_list.append(new_image_data_id, last_state, additional_data, action)
            print("save")
        else:
            #update_data_in_database(similar_image_data_id, new_image_data_id, last_state, additional_data, action)
            print("update")
        update_image_dict()
        print()
        #print("learned")
    """

    # handle data adding for continue data input
    continuse_states.append(the_state)
    if len(continuse_states) > sequence_length:
        continuse_states = continuse_states[-sequence_length:]

    if "x_pos" in the_info:
        x_position_list.append(the_info["x_pos"])
    else:
        x_position_list.append(0)
    if len(x_position_list) > sequence_length:
        x_position_list = x_position_list[-sequence_length:]
    speed = (x_position_list[-1] - x_position_list[0]) / sequence_length
    if speed >= 0:
        speed = "+" + str(speed)
    else:
        speed = str(speed)
    #print("speed:", speed)
    if "y_pos" in the_info:
        y_position_list.append(str(the_info["y_pos"]))
    else:
        y_position_list.append(str(0))
    if len(y_position_list) > sequence_length:
        y_position_list = y_position_list[-sequence_length:]

    if "x_pos" in the_info:
        old_x_position = the_info["x_pos"]
        if old_x_position > max_x_position:
            max_x_position = old_x_position
            #for one in good_data_list[:int(len(good_data_list)*0.5)]:
            for one in good_data_list[:int(len(good_data_list)*0.7)]:
                if one[0] in processed_data_id_set:
                    continue
                else:
                    processed_data_id_set.add(one[0])
                is_unique, similar_image_data_id = check_if_image_is_unique(one[1], one[2])
                if is_unique:
                    save_data_to_database(*one)
                else:
                    update_data_in_database(similar_image_data_id, *one)
            update_image_dict()

    #cv2.imshow("real input", last_state)
    #cv2.waitKey(1)

    env.render()

env.close()
