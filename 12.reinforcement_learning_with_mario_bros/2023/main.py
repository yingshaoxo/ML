from datetime import datetime
import cv2

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

env = gym_super_mario_bros.make("SuperMarioBros-v2", apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

from random import randint
import numpy as np
import torch

from model import Trainer


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

jump_height = 0
speed = 0
while 1:
    for step in range(100000):
        if done:
            state = env.reset()
            done = False

        if last_observe_data_as_input != None and last_observe_data_as_input[0] != None and randint(0, 10) != 0:
            action = trainer.predict(last_observe_data_as_input)
        else:
            action = env.action_space.sample() # this is an integer between [1, 7]
            # print(len(SIMPLE_MOVEMENT))

        state, reward, terminated, truncated, info = env.step(action) # type: ignore
        state, state_for_pytorch = image_process(state)
        done = terminated or truncated
        x_position = info.get("x_pos")
        x_position = 0 if x_position == None else int(x_position)
        y_position = info.get("y_pos")
        y_position = 0 if y_position == None else int(y_position)
        game_level = info.get("world")

        now = datetime.now()
        time_difference_in_milliseconds = (now - last_time).microseconds / 1000
        if time_difference_in_milliseconds >= 500:
            x_position_distance = x_position - last_x_position  
            speed = x_position_distance

            y_position_distance = y_position - last_y_position 
            jump_height = y_position_distance

            last_time = now

        last_observe_data_as_input = [last_state, speed, game_level]
        last_state = state_for_pytorch
        last_x_position = x_position
        last_y_position = y_position
        # print(f"reward: {reward}; info: {info}")

        if reward > 0:
            trainer.train(data=last_observe_data_as_input, target_data=action)
        if speed > 0 and jump_height > 0:
            trainer.train(data=last_observe_data_as_input, target_data=action)

        env.render()
        cv2.imshow('graycsale mario', state)

    trainer.print_progress_info(epoch_index)
    epoch_index += 1
    print("saving...")
    trainer.save_model()

env.close()
cv2.destroyAllWindows()