# https://pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/#:~:text=Let%E2%80%99s%20go%20ahead%20and%20compile%2C%20train%2C%20and%20evaluate%20our%20newly%20formed
# https://pyimagesearch.com/2018/06/04/keras-multiple-outputs-and-multiple-losses/#:~:text=Let%E2%80%99s%20build%20the%20network%2C%20define%20our%20independent%20losses%2C%20and%20compile%20our%20model

from collections import deque
from datetime import datetime
from operator import mod
import time
import random

import tensorflow
import tensorflow_hub
from transformers import BartTokenizer, BartModel

keras = tensorflow.keras
keras.utils.disable_interactive_logging()

import cv2
import numpy

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT



def get_broken_timestamp():
    return str(time.time()).split(".")[1]

def make_sure_a_number_is_in_a_range(number, range):
    start, end = range
    if number < start:
        return start
    if number > end:
        return end
    return number


class Trainer:
    image_one_side_length = 225

    # env = gym_super_mario_bros.make("SuperMarioBros-v2", apply_api_compatibility=True, render_mode="rgb_array") #just the data
    env = gym_super_mario_bros.make("SuperMarioBros-v2", apply_api_compatibility=True, render_mode="human") #include the graph
    MY_MARIO_MOVEMENT = [
        ['NOOP'],
        ['A', 'right', 'B'],
        # ['B', 'left', 'A'],
    ]
    the_action_numbers = len(MY_MARIO_MOVEMENT)
    # actions_by_using_matrix_representation = numpy.identity(the_action_numbers) # for quickly get a hot vector, like 0001000000000000

    mario_model = None
    action_score_model = None

    mobilenet_v3_small_image_model =  tensorflow_hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v2_075_224/feature_vector/5", trainable=False)

    bart_base_tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    bart_base_model = BartModel.from_pretrained('facebook/bart-base')

    steps_range = (10, 70)

    last_state = None
    last_life = 2
    reward_sum_before_dying = 0

    x_diff = 0

    history_action_length = 4 
    action_history = list([0 for i in range(5)])

    temp_life_experience = []
    global_positive_life_experience = []
    max_storage_for_life_experience = 30

    def __init__(self) -> None:
        self.mario_model = self.get_mario_model()
        self.action_score_model = self.get_action_score_model()
    
    def numpy_to_tf_tensor(self, array, add_one_dimention=False):
        result = tensorflow.convert_to_tensor(array, dtype=tensorflow.float32)
        if add_one_dimention:
            return result
        else:
            return tensorflow.expand_dims(result, axis=0)
    
    def it_is_not_none(self, array):
        if 'NoneType' in str(type(array)):
            return False
        else:
            return True

    def get_image_vector(self, image):
        #return shape: (1, 1280)
        # image = image / 255
        image = cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
        image = tensorflow.image.convert_image_dtype(image, tensorflow.float32)
        results = self.mobilenet_v3_small_image_model(tensorflow.expand_dims(image, axis=0))
        return results
    
    def get_text_vector(self, text):
        # output: (1, 46080)
        inputs = self.bart_base_tokenizer(text, return_tensors="pt")
        outputs = self.bart_base_model(**inputs) # type: ignore
        last_hidden_states = outputs.last_hidden_state

        np_tensor = last_hidden_states.detach().numpy()
        tf_tensor = tensorflow.convert_to_tensor(np_tensor)

        return tensorflow.reshape(tf_tensor, [1, -1])

    def image_process(self, frame: numpy.ndarray):
        side_length = 225
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (side_length, side_length)).astype(numpy.uint8)
        else:
            frame = numpy.zeros((3, side_length, side_length)) #may it is not a good idea to put 0 here, may cause other weights get into 0 quickly
        
        width, height = frame.shape[1], frame.shape[0]
        frame = frame[40:height, 0:width]
        
        return self.get_image_vector(frame)
    
    def get_mario_model(self):
        # handle image part
        # image_input = keras.Input(shape=(32, 5, 5, 1280), name="img")
        # x = keras.layers.Conv2D(32, 3, strides=(2,2), padding="same", activation="relu")(image_input)
        # x = keras.layers.Conv2D(32, 3, strides=(2,2), padding="same", activation="relu")(x)
        # x = keras.layers.Flatten()(x)
        action_image_input = keras.Input(shape=(1280), name="action_image_input")
        x = keras.layers.Dense(64, activation='relu')(action_image_input)
        # x = keras.layers.Dense(512, activation="relu")(x)
        action_image_output = keras.layers.Dense(64, activation='relu')(x)

        history_action_input = keras.Input(shape=(self.history_action_length), name="history_action")
        x = keras.layers.Dense(32, activation='relu')(history_action_input)
        x = keras.layers.Dense(32, activation='relu')(x)
        history_action_output = keras.layers.Dense(32, activation='relu')(x)

        # handle the prediction of `which action to use` part
        x = keras.layers.concatenate([action_image_output, history_action_output])
        x = keras.layers.Dense(32, activation="relu")(x)
        x = keras.layers.Dense(32, activation="relu")(x)
        action_output = keras.layers.Dense(self.the_action_numbers, activation="relu", name="action_output")(x)

        steps_image_input = keras.Input(shape=(1280), name="steps_image_input")
        x = keras.layers.Dense(32, activation='relu')(action_image_input)
        x = keras.layers.Dense(32, activation="relu")(x)
        steps_image_output = keras.layers.Dense(16, activation='relu')(x)

        # handle the prediction of `how many steps for one action` part
        x = keras.layers.concatenate([steps_image_output, history_action_output, action_output])
        x = keras.layers.Dense(128, activation="relu")(x)
        x = keras.layers.Dense(64, activation="relu")(x)
        step_output = keras.layers.Dense(1, activation="relu", name="steps_output")(x)

        # # handle the prediction of `how many reward we'll get` part
        # x = keras.layers.concatenate([image_output, action_output, step_output])
        # # x = keras.layers.Dense(32, activation="relu")(x)
        # x = keras.layers.Dense(64, activation="relu")(x)
        # reward_output = keras.layers.Dense(1, activation="relu", name="reward_output")(x)

        model = keras.Model([action_image_input, steps_image_input, history_action_input], [action_output, step_output], name="yingshaoxo_and_mario")
        #print(model.summary())

        model.compile(
            optimizer=keras.optimizers.Adam(0.001),
            loss={
                "action_output": "sparse_categorical_crossentropy",
                "steps_output": "mean_squared_error",
                # "reward_output": "mean_squared_error",
            },
            loss_weights={
                "action_output": 1,
                "steps_output": 1,
                # "reward_output": 0,
            },
            metrics=["accuracy"],
        )

        return model
    
    def get_action_score_model(self):
        # model = keras.Sequential([
        #         keras.layers.Conv2D(input_shape=(32, 5, 5, 1280), filters=32, kernel_size=3, strides=(2,2), padding="same", activation='relu'),
        #         keras.layers.Conv2D(32, 3, strides=(2,2), activation='relu'),
        #         keras.layers.Conv2D(32, 3, strides=(2,2), activation='relu'),
        #         keras.layers.Flatten(),
        #         keras.layers.Dense(512, activation='relu'),
        #         keras.layers.Dense(1, activation='relu'),
        # ])

        image_input = keras.Input(shape=(1280), name="image_input")
        x = keras.layers.Dense(512, activation='relu')(image_input)
        x = keras.layers.Dense(64, activation='relu')(x)
        image_output = keras.layers.Dense(16, activation='relu')(x)

        action_input = keras.Input(shape=(1), name="action_input")

        merged_layout = keras.layers.concatenate([image_output, action_input])
        x = keras.layers.Dense(16, activation='relu')(merged_layout)
        reward_output = keras.layers.Dense(1, activation='relu', name="reward_output")(x)

        model = keras.Model([image_input, action_input], [reward_output], name="guess_mario_reward_model")

        model.compile(optimizer='sgd', loss='mse')

        return model
    
    def perform_n_steps_with_one_action(self, n, action):
        state, reward, done, info = numpy.array([]), 0, False, {}
        old_info = None
        OK = True
        for i in range(1, n+1):
            state, reward, terminated, truncated, info = self.env.step(action) # type: ignore
            done = terminated or truncated
            if old_info == None:
                old_info = info
            if done:
                OK = False
                break
        x_diff = info['x_pos'] - old_info['x_pos']
        return OK, x_diff, (state, reward, done, info)
    
    def loop(self):
        self.env = JoypadSpace(self.env, self.MY_MARIO_MOVEMENT)

        done = True
        while True:
            if self.last_life != 2:
                done = True

            if done:
                self.reward_sum_before_dying = 0
                self.last_life = 3
                self.last_state = None
                self.last_x_position = 0

                state = self.env.reset()
                self.action_history = [0 for i in range(self.history_action_length)]

                self.temp_life_experience.sort(key=lambda x: x[0])
                if len(self.temp_life_experience) > 0:
                    the_temp_max_value = self.temp_life_experience[-1][0]
                    self.global_positive_life_experience.append([the_temp_max_value, self.temp_life_experience[:len(self.temp_life_experience)//2].copy()])
                self.temp_life_experience = []

                self.global_positive_life_experience.sort(key=lambda x: x[0], reverse=True)
                if len(self.global_positive_life_experience) >= self.max_storage_for_life_experience:
                    self.global_positive_life_experience = self.global_positive_life_experience[:self.max_storage_for_life_experience]

                print(f"{get_broken_timestamp()} in tranning...")
                for part in self.global_positive_life_experience[:len(self.global_positive_life_experience)//2]:
                    for one in part[1]:
                        self.mario_model.fit(
                            x=one[1], 
                            y=one[2],
                        )

            if random.randint(0, 2) == 0 or (not self.it_is_not_none(self.last_state)):
                action = self.env.action_space.sample()
                steps = random.randint(self.steps_range[0], self.steps_range[1])
                print(f"{get_broken_timestamp()},                                       , {steps} by random choose...")
            else:
                # result1 = self.action_score_model.predict(
                #     {
                #         "image_input": self.last_state,
                #         "action_input": tensorflow.expand_dims(0, axis=0),
                #     }
                # )
                # result1 = result1[0][0]
                # result2 = self.action_score_model.predict(
                #     {
                #         "image_input": self.last_state,
                #         "action_input": tensorflow.expand_dims(1, axis=0),
                #     }
                # )
                # result2 = result2[0][0]

                # if result1 > result2:
                #     action = 0
                # else:
                #     action = 1

                result = self.mario_model.predict(
                            {
                                "action_image_input": self.last_state, 
                                "steps_image_input": self.last_state, 
                                "history_action":  tensorflow.convert_to_tensor([self.action_history], dtype=tensorflow.float32)
                            }
                        )
                if (str(result[1][0][0]) == "nan"):
                    print("nan error...")
                    continue
                action = numpy.argmax(result[0])
                steps = make_sure_a_number_is_in_a_range(int(result[1][0][0]), self.steps_range)
                self.action_history += [int(action)]
                self.action_history = self.action_history[-self.history_action_length:]
                print(f"{get_broken_timestamp()},             , {steps} by prediction...")

            ok, x_diff, (state, reward, done, info) = self.perform_n_steps_with_one_action(steps, action)
            if (ok == False):
                continue
            self.x_diff = x_diff
            state = self.image_process(state)
            life = info['life']

            # add data to database
            if self.it_is_not_none(self.last_state):
                the_x = {
                        "action_image_input": self.last_state, 
                        "steps_image_input": self.last_state, 
                        "history_action": tensorflow.convert_to_tensor([self.action_history], dtype=tensorflow.float32) 
                    } 
                the_y = {
                        "action_output": tensorflow.expand_dims(action, axis=0),
                        "steps_output": tensorflow.expand_dims(steps, axis=0),
                        # "reward_output": tensorflow.expand_dims(reward, axis=0),
                }
                self.temp_life_experience.append((info['x_pos'], the_x, the_y))

                # self.action_score_model.fit(
                #     x={
                #         "image_input": self.last_state,
                #         "action_input": tensorflow.expand_dims(action, axis=0),
                #     },
                #     y={
                #         "reward_output": tensorflow.expand_dims(reward+15, axis=0)
                #     }
                # )

            self.env.render()

            self.last_state = state
            self.reward_sum_before_dying += reward
            self.last_life = life
                

        self.env.close()


trainer = Trainer()
trainer.loop()