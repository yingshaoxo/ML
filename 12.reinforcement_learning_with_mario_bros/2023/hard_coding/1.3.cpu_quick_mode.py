import random
import sys
import termios
import tty
from typing import Any

import cv2
import numpy as np
from PIL import Image

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
env = gym_super_mario_bros.make('SuperMarioBros-v2', apply_api_compatibility=True, render_mode="human")
MY_MARIO_MOVEMENT = [
    ['right', 'B'],
    #['right'],
    ['A'],
    ['left'],
]
env = JoypadSpace(env, MY_MARIO_MOVEMENT)

from auto_everything.disk import Disk
disk = Disk()

from auto_everything.cryptography import Password_Generator
password_generator = Password_Generator(base_secret_string="yingshaoxo is the strongest person in this world.")

from auto_everything.database import Database_Of_Yingshaoxo
mobile_net_image_database = Database_Of_Yingshaoxo(database_name="cpu_quick_mode", use_sqlite=False)
mobile_net_image_database.refactor_database()

from auto_everything.ml import Yingshaoxo_Computer_Vision
yingshaoxo_computer_vision = Yingshaoxo_Computer_Vision()

preset_action_dict = {
    "jump_pipeline": [2, 2, 2, 2, 0, 0, 0, 1, 1, 1, 1]
}


current_folder = disk.get_directory_path(__file__)
source_image_folder = disk.join_paths(current_folder, "raw_images")


images_dict = {}
image_keys = []
def one_row_dict_handler(data: dict[str, Any]) -> (dict[str, Any] | None):
    return data
def update_image_dict():
    global images_dict, image_keys
    images_dict = {image_data["path"]:image_data for image_data in mobile_net_image_database.search(one_row_dict_handler=one_row_dict_handler)}
    image_keys = list(images_dict.keys())
    for key in image_keys:
        images_dict[key]["data"] = np.array(images_dict[key]["data"])
update_image_dict()


def image_process(frame):
    side_length = 225
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (side_length, side_length)).astype(np.uint8)
    else:
        frame = np.zeros((3, side_length, side_length)) #may it is not a good idea to put 0 here, may cause other weights get into 0 quickly
    
    width, height = frame.shape[1], frame.shape[0]
    frame = frame[40:height, 0:width]

    frame = cv2.resize(frame, (8, 8)).astype(np.uint8)
    
    return frame

def get_image_feature_vector_data(image):
    return image

def get_similarity_between_two_numpy_array(array_a, array_b):
    # norm_a = np.linalg.norm(array_a)
    # norm_b = np.linalg.norm(array_b)
    # dot = sum(a * b for a, b in zip(array_a, array_b))
    # return (dot / (norm_a * norm_b))
    return yingshaoxo_computer_vision.get_similarity_of_two_images(array_a, array_b)

def check_if_image_is_unique(image) -> tuple[bool, str|None]:
    global images_dict
    original_image_data = get_image_feature_vector_data(image)

    for image_data in images_dict.values():
        temp_data = image_data["data"]
        similarity = get_similarity_between_two_numpy_array(temp_data, original_image_data)

        if similarity > 0.999:
            return False, image_data["path"]

    return True, None

def get_char_input() -> tuple[str, int]:
    #https://www.physics.udel.edu/~watson/scen103/ascii.html
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        char = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return char, ord(char)

def get_most_possible_action_from_database(input_image) -> tuple[str, str]:
    global images_dict, image_keys
    last_used_image_path = ""
    similarity_list = []
    last_state_feature = get_image_feature_vector_data(input_image)
    for image_path in random.sample(image_keys, len(image_keys)):
        image = images_dict[image_path]["data"]
        last_used_image_path = images_dict[image_path]["path"]

        similarity = get_similarity_between_two_numpy_array(image, last_state_feature)
        action = image_path.split("/")[-2]

        similarity_list.append([similarity, action])
    similarity_list.sort(key=lambda item: item[0], reverse=True)
    the_most_possible_one = similarity_list[0]
    #print(the_most_possible_one)
    action = the_most_possible_one[1]
    return action, last_used_image_path

def save_numpy_image_to_database(image, path):
    image_data = {
        "data": get_image_feature_vector_data(image=image).tolist(),
        "path": path,
    }
    mobile_net_image_database.add(image_data)

def update_numpy_image_in_database(old_path, image, path):
    def one_row_dict_filter(item: dict[str, Any]) -> bool:
        if item["path"] == old_path:
            return True
        return False
    mobile_net_image_database.delete(one_row_dict_filter=one_row_dict_filter)

    image_data = {
        "data": get_image_feature_vector_data(image=image).tolist(),
        "path": path,
    }
    mobile_net_image_database.add(image_data)

i = 0
state, info = env.reset()
state = image_process(state)
action_queue = []
continues_steps = 10

last_used_image_path = None

sequence_length = 32
continuse_states = [state] * sequence_length

last_state = np.concatenate(continuse_states, axis=0)

def add_string_action_to_real_action_queue(action_string: str):
    global preset_action_dict, action_queue
    action_sequence = preset_action_dict[action_string]
    for one in action_sequence:
        action_queue += [int(one)] * continues_steps

while True:
    try:
        human_handled = False

        action = None
        if len(action_queue) == 0:
            input_char, input_char_id = get_char_input()
            if input_char == '0':
                action = 0
                action_queue += [action] * continues_steps
            elif input_char == '1':
                action = 1
                action_queue += [action] * continues_steps
            elif input_char == '2':
                action = 2
                action_queue += [action] * continues_steps
            elif input_char == "j":
                action = "jump_pipeline"
                add_string_action_to_real_action_queue(action)
            elif input_char == "6":
                # press 6 to let the bot run mario
                action = None
            else:
                action = None

            if action == None:
                if len(image_keys) == 0:
                    action = env.action_space.sample()
                    action_queue += [int(action)] * continues_steps
                else:
                    action, last_used_image_path = get_most_possible_action_from_database(last_state)

                    if len(action) == 1:
                        action_queue += [int(action)] * continues_steps
                    else:
                        add_string_action_to_real_action_queue(action)

                human_handled = False
            else:
                human_handled = True
        
        real_action = action

        action = action_queue[0]
        action_queue = action_queue[1:]

        if real_action == None:
            real_action = action

        state, reward, terminated, truncated, info = env.step(action) # type: ignore
        state = image_process(state)
        done = terminated or truncated or (info["time"] == 0)

        if human_handled == True:
            is_unique, similar_image_path = check_if_image_is_unique(last_state)
            new_image_path = disk.join_paths(source_image_folder, f"{real_action}", f"{password_generator.get_random_password(length=12)}.png")
            if is_unique:
                save_numpy_image_to_database(last_state, new_image_path)
            else:
                update_numpy_image_in_database(similar_image_path, last_state, new_image_path)
            update_image_dict()
            print()
            print("learned")

        if done:
            state, info = env.reset()
            state = image_process(state)

        continuse_states.append(state)
        if len(continuse_states) > sequence_length:
            continuse_states = continuse_states[-sequence_length:]
        last_state = np.concatenate(continuse_states, axis=0)
        # cv2.imshow("input", last_state)
        # cv2.waitKey(1)

        env.render()

        i += 1
    except Exception as e:
        print(e)

env.close()