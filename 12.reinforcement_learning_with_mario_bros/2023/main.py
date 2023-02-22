from datetime import datetime
import cv2
import time

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

env = gym_super_mario_bros.make("SuperMarioBros-v2", apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

from random import randint
import numpy as np
import torch

from collections import deque
from model import Trainer

def get_broken_timestamp():
    return str(time.time()).split(".")[1]

def image_process(frame):
    if frame is not None:
        frame = cv2.resize(frame, (84, 84))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.uint8)
        # frame = cv2.Canny(frame, 100, 200)
    else:
        frame = np.zeros((1, 84, 84)) #may it is not a good idea to put 0 here, may cause other weights get into 0 quickly

    frame = np.where(frame > 127, 255, 0).astype(np.uint8)

    frame_for_pytorch = np.where(frame > 127, 1.0, 0.0)
    frame_for_pytorch = np.expand_dims(frame_for_pytorch, 0)
    frame_for_pytorch = torch.from_numpy(frame_for_pytorch).float()
    return frame, frame_for_pytorch

# model_file_path = "./nn_model"
# if os.path.exists(model_file_path):
#     model = tf.keras.models.load_model(model_file_path)
# else:
#     img_rows, img_cols = 240, 256
#     model = generate_model((img_rows, img_cols, 3), env.action_space.n)


trainer = Trainer()
epoch_index = 0
max_experience_length = 300 #slowly go up
experience_list = deque(maxlen=max_experience_length)
minimum_speed = 2


"""
env.action_space.sample() = numbers, for example, 0,1,2,3...
state = RGB of raw picture; is a numpy array with shape (240, 256, 3), but we can convert it to greyscale picture with shape (240, 256, 1)
reward = int; for example, 0, 1 ,2, ...
done = False or True
info = {'coins': 0, 'flag_get': False, 'life': 3, 'score': 0, 'stage': 1, 'status': 'small', 'time': 400, 'world': 1, 'x_pos': 40}
"""
done = True
last_state = None
last_time = datetime.now()
last_x_position = 0
last_y_position = 0

last_observe_data_as_input = None

the_farest_x_position = 0 #slowly go down

jump_height = 0
speed = 0

stay_in_same_position_count = 0
while 1:
    for step_count in range(10000):
        if done:
            state = env.reset()
            # experience_list.clear()
            done = False

            last_state = None
            # last_time = datetime.now()
            last_x_position = 0
            last_y_position = 0
            last_observe_data_as_input = None

            # print(the_farest_x_position)
            the_farest_x_position *= 0.9
            if the_farest_x_position < 0:
                the_farest_x_position = 0
        
        # if last_observe_data_as_input != None and last_observe_data_as_input[0] != None:
        #     action = trainer.predict(last_observe_data_as_input)
        # else:
        #     action = env.action_space.sample() # this is an integer between [1, 7]
        #     # print(len(SIMPLE_MOVEMENT))

        is_random_action = True
        if len(experience_list) < max_experience_length or speed < minimum_speed:
            action = env.action_space.sample() # this is an integer between [1, 7]
            # print(len(SIMPLE_MOVEMENT))
        else:
            if last_observe_data_as_input != None and last_observe_data_as_input[0] != None:
                action = trainer.predict(last_observe_data_as_input)
                print(f"                                , go by prediction")
                is_random_action = False
            else:
                action = env.action_space.sample() # this is an integer between [1, 7]

        state, reward, terminated, truncated, info = env.step(action) # type: ignore
        state, state_for_pytorch = image_process(state)
        done = terminated or truncated
        x_position = info.get("x_pos")
        x_position = 0 if x_position == None else int(x_position)
        y_position = info.get("y_pos")
        y_position = 0 if y_position == None else int(y_position)
        game_level = info.get("world")
        if (game_level == 2):
            exit()

        now = datetime.now()
        time_difference_in_milliseconds = (now - last_time).microseconds / 1000
        if time_difference_in_milliseconds >= 100:
            x_position_distance = x_position - last_x_position  
            speed = x_position_distance

            y_position_distance = y_position - last_y_position 
            jump_height = y_position_distance

            if speed == 0:
                stay_in_same_position_count += 1
                second_has_passed = stay_in_same_position_count // 10
                if second_has_passed >= 3:
                    stay_in_same_position_count = 0
                    done = True

            last_time = now

        last_observe_data_as_input = [last_state, speed, game_level]
        last_state = state_for_pytorch
        last_x_position = x_position
        last_y_position = y_position
        # print(f"reward: {reward}; info: {info}")

        # if reward > 0:
        #     trainer.train(data=last_observe_data_as_input, target_data=action)
        # if (speed > 0 and jump_height > 0):
        #     trainer.train(data=last_observe_data_as_input, target_data=action)
        # print(len(experience_list))
        # print(speed, jump_height)

        experience_list.append((last_observe_data_as_input, action)) # type: ignore
        if speed >= minimum_speed:
            if x_position > the_farest_x_position:
                the_farest_x_position = x_position
                if len(experience_list) == max_experience_length:
                    print(f"learn from data where ai still alive, {get_broken_timestamp()}")
                    one_item = experience_list.popleft()
                    trainer.train(data=one_item[0], target_data=one_item[1])
            else:
                if len(experience_list) < max_experience_length:
                    print(f"learn after fail, {get_broken_timestamp()}")
                    trainer.train(data=last_observe_data_as_input, target_data=action)

        env.render()
        cv2.imshow('graycsale mario', state)

    trainer.print_progress_info(epoch_index)
    epoch_index += 1
    print("saving...")
    trainer.save_model()

env.close()
cv2.destroyAllWindows()