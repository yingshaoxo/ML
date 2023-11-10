import random
import sys
import termios
import tty

import cv2
import numpy as np
from PIL import Image

from sewar.full_ref import msssim, uqi, rmse

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

from auto_everything.cryptography import Encryption_And_Decryption, Password_Generator
password_generator = Password_Generator(base_secret_string="yingshaoxo is the strongest person in this world.")

current_folder = disk.get_directory_path(__file__)
source_image_folder = disk.join_paths(current_folder, "raw_images")
disk.create_a_folder(source_image_folder)

images = disk.get_files(folder=source_image_folder)
images_dict = {image_path:cv2.imread(image_path) for image_path in images}
image_keys = list(images_dict.keys())

preset_action_dict = {
    "jump_pipeline": [2, 2, 2, 2, 0, 0, 0, 1, 1, 1, 1]
}

def update_image_dict():
    global images, images_dict, image_keys
    images = disk.get_files(folder=source_image_folder)
    images_dict = {image_path:cv2.imread(image_path) for image_path in images}
    image_keys = list(images_dict.keys())

def check_if_image_is_unique(image):
    global images, images_dict, image_keys
    save_it = True
    for image_path in image_keys:
        temp_image = image_process(images_dict[image_path])

        similarity = uqi(image, temp_image)

        if similarity > 0.92:
            save_it = False
            break
    return save_it

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

def image_process(frame):
    side_length = 225
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (side_length, side_length)).astype(np.uint8)
    else:
        frame = np.zeros((3, side_length, side_length)) #may it is not a good idea to put 0 here, may cause other weights get into 0 quickly
    
    width, height = frame.shape[1], frame.shape[0]
    frame = frame[40:height, 0:width]
    # frame = frame / 255
    
    return frame

def save_numpy_image(image, path):
    disk.create_a_folder(disk.get_directory_path(path)) 
    cv2.imwrite(path, image)

i = 0
state,_ = env.reset()
state = image_process(state)
last_state = state
action_queue = []
continues_steps = 10

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

            if action == None:
                if len(image_keys) == 0:
                    action = env.action_space.sample()
                    action_queue += [int(action)] * continues_steps
                else:
                    similarity_list = []
                    for image_path in random.sample(image_keys, len(image_keys)):
                        image = image_process(images_dict[image_path])

                        similarity = uqi(image, last_state)
                        action = image_path.split("/")[-2]

                        similarity_list.append([similarity, action])
                    similarity_list.sort(key=lambda item: item[0], reverse=True)
                    the_most_possible_one = similarity_list[0]
                    #print(the_most_possible_one)
                    action = the_most_possible_one[1]

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
        done = terminated or truncated

        if human_handled == True:
            if check_if_image_is_unique(last_state):
                save_numpy_image(last_state, disk.join_paths(current_folder, "raw_images", f"{real_action}", f"{password_generator.get_random_password(length=12)}.png"))
                update_image_dict()
                print()
                print("learned")

        if done:
            state, _ = env.reset()
            state = image_process(state)

        last_state = np.copy(state)

        env.render()

        i += 1
    except Exception as e:
        print(e)

env.close()