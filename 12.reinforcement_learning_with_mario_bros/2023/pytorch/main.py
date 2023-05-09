from datetime import datetime
import cv2
import time
import os

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
# from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

env = gym_super_mario_bros.make("SuperMarioBros-v2", apply_api_compatibility=True, render_mode="human")
MY_MARIO_MOVEMENT = [
    ['right', 'B'],
    ['A'],
    ['left'],
]
env = JoypadSpace(env, MY_MARIO_MOVEMENT)

from random import randint
import numpy as np
import torch
from PIL import Image

import torchvision
import torchvision.models as models

from collections import deque
from model import Trainer

def get_broken_timestamp():
    return str(time.time()).split(".")[1]

# transforms = torchvision.transforms.Compose([
#     torchvision.transforms.Resize(256),
#     torchvision.transforms.CenterCrop(224),
#     torchvision.transforms.ToTensor(),
#     torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
# # Load the pretrained model
# model = models.resnet18(pretrained=True)
# # Use the model object to select the desired layer
# layer = model._modules.get('avgpool')
# # Set model to evaluation mode
# model.eval()

# def get_image_vector(image):
#     image = Image.fromarray(image)

#     # Create a PyTorch tensor with the transformed image
#     t_img = transforms(image)
#     # Create a vector of zeros that will hold our feature vector
#     # The 'avgpool' layer has an output size of 512
#     my_embedding = torch.zeros(512)

#     # Define a function that will copy the output of a layer
#     def copy_data(m, i, o):
#         my_embedding.copy_(o.flatten())                 # <-- flatten

#     # Attach that function to our selected layer
#     h = layer.register_forward_hook(copy_data)  # type: ignore
#     # Run the model on our transformed image
#     with torch.no_grad():                               # <-- no_grad context
#         model(t_img.unsqueeze(0))                       # <-- unsqueeze
#     # Detach our copy function from the layer
#     h.remove()
#     # Return the feature vector
#     return my_embedding

def image_process(frame):
    side_length = 225
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (side_length, side_length)).astype(np.uint8)
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.uint8)
        # frame = cv2.Canny(frame, 100, 200)
    else:
        frame = np.zeros((3, side_length, side_length)) #may it is not a good idea to put 0 here, may cause other weights get into 0 quickly
    
    width, height = frame.shape[1], frame.shape[0]
    frame = frame[40:height, 0:width]
    frame = frame / 255
    
    return frame#, get_image_vector(frame)


# model_file_path = "./nn_model"
# if os.path.exists(model_file_path):
#     model = tf.keras.models.load_model(model_file_path)
# else:
#     img_rows, img_cols = 240, 256
#     model = generate_model((img_rows, img_cols, 3), env.action_space.n)


trainer = Trainer()
if os.path.exists(trainer.model_saving_path):
    trainer.load_model()
epoch_index = 0
max_experience_length = 150 #slowly go up
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
speed = 1

stay_in_same_position_count = 0
while 1:
    for step_count in range(3000):
        if done:
            state = env.reset()
            experience_list.clear()
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
        
        is_random_action = True
        if len(experience_list) < max_experience_length or speed < minimum_speed:
            action = env.action_space.sample() # this is an integer between [1, 7]
            # print(len(SIMPLE_MOVEMENT))
        else:
            if last_observe_data_as_input != None and last_observe_data_as_input[0] != None:
                action = trainer.predict(last_observe_data_as_input)
                print(f"                                , go by prediction, {get_broken_timestamp()}")
                is_random_action = False
            else:
                action = env.action_space.sample() # this is an integer between [1, 7]

        state, reward, terminated, truncated, info = env.step(action) # type: ignore
        state = image_process(state)
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

        last_observe_data_as_input = [last_state, game_level]
        last_state = state
        last_x_position = x_position
        last_y_position = y_position
        # print(f"reward: {reward}; info: {info}")

        experience_list.append((last_observe_data_as_input, action)) # type: ignore
        if len(experience_list) == max_experience_length:
            print(f"learn from data where ai still alive, {get_broken_timestamp()}")
            one_item = experience_list.popleft()
            trainer.train(data=one_item[0], target_data=one_item[1])
        else:
            if reward > 2 or speed > 0:
                print(f"pre-learning, {get_broken_timestamp()}")
                trainer.train(data=last_observe_data_as_input, target_data=action)

        env.render()
        # cv2.imshow('my mario', state)

    trainer.print_progress_info(epoch_index)
    epoch_index += 1
    print("saving...")
    trainer.save_model()

env.close()
cv2.destroyAllWindows()