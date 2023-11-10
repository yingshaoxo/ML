import sys
import termios
import tty

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

from auto_everything.cryptography import Encryption_And_Decryption, Password_Generator
password_generator = Password_Generator(base_secret_string="yingshaoxo is the strongest person in this world.")
from auto_everything.disk import Disk
disk = Disk()

current_folder = disk.get_directory_path(__file__)

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

def crop_image_to_pieces(numpy_image, height, width):
    im = Image.fromarray(numpy_image)
    imgwidth, imgheight = im.size
    for i in range(imgheight//height):
        for j in range(imgwidth//width):
            box = (j*width, i*height, (j+1)*width, (i+1)*height)
            yield np.array(im.crop(box))

def save_numpy_image(image, path):
    disk.create_a_folder(disk.get_directory_path(path)) 
    cv2.imwrite(path, image)

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

i = 0
state,_ = env.reset()
state = image_process(state)
last_state = state
while True:
    try:
        input_char, input_char_id = get_char_input()
        action = env.action_space.sample()
        if input_char == '0':
            action = 0
        elif input_char == '1':
            action = 1
        elif input_char == '2':
            action = 2
        # elif input_char == '3':
        #     action = 3

        state, reward, terminated, truncated, info = env.step(action) # type: ignore
        # state, reward, done, info = env.step(env.action_space.sample()) # type: ignore
        state = image_process(state)
        done = terminated or truncated

        # height, width, _ = state.shape
        # sub_height = height // (16//2)
        # sub_width = width // (16//2)
        # sub_image_list = crop_image_to_pieces(state, sub_height, sub_width)

        save_numpy_image(last_state, disk.join_paths(current_folder, "raw_images", f"{action}", f"{password_generator.get_random_password(length=12)}.jpg"))
        # save_numpy_image(state, disk.join_paths(current_folder, "raw_images", f"{i}.jpg"))
        # for index, sub_image in enumerate(sub_image_list):
        #     save_numpy_image(sub_image, disk.join_paths(current_folder, "raw_seperate_images", f"{i}_{index}.jpg"))

        if done:
            state, _ = env.reset()
            state = image_process(state)

        last_state = np.copy(state)

        env.render()

        i += 1
    except Exception as e:
        print(e)

env.close()