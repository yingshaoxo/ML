import numpy as np
import tensorflow as tf
# tf.keras.utils.disable_interactive_logging()
from tensorflow import keras
import keras.backend as keras_backend

from random import randint
import os
import cv2

from pymongo import MongoClient


class Trainer:
    def __init__(self):
        from nes_py.wrappers import JoypadSpace
        import gym_super_mario_bros

        self.env = gym_super_mario_bros.make('SuperMarioBros-v2', apply_api_compatibility=True, render_mode="human")
        MY_MARIO_MOVEMENT = [
            ['right'],
            ['right', 'A'],
            # ['NOOP'],
            # ['A', 'right', 'B'],
            # # ['A', 'left', 'B'],
            # ['A', 'right'],
            # # ['A', 'left'],
            # ['right'],
            # # ['left'],
        ]
        self.number_of_actions = len(MY_MARIO_MOVEMENT)
        self.env = JoypadSpace(self.env, MY_MARIO_MOVEMENT)

        self.img_rows , self.img_cols = 240, 256
        self.continuous_image_number = 3
        self.time_jump_number = 50
        # self.use_how_many_steps_later_reward = 100

        self.reward_model_file_path = './reward_nn_keras_model'
        self.action_model_file_path = './action_nn_keras_model'

        self.reward_model = self.generate_reward_model()
        self.action_model = self.generate_action_model()

        self.mongo_client = MongoClient('mongodb://localhost:27017/')
        self.training_data_collection = self.mongo_client['mario']['training_data_collection']
        self.test_data_collection = self.mongo_client['mario']['test_data_collection']

    def generate_reward_model(self):
        action_image_input = keras.Input(shape=(self.continuous_image_number, self.img_rows, self.img_cols, 1), name="action_image_input")
        x = keras.layers.ConvLSTM2D(128, kernel_size=8, strides=32, padding='same', return_sequences=True)(action_image_input)
        x = keras.layers.ConvLSTM2D(128, kernel_size=2, strides=3, padding='same')(x)
        x = keras.layers.Flatten()(x)
        commom_layer = keras.layers.Dense(512, activation='relu')(x)

        action_input = keras.Input(shape=(1), name="action_input")
        x2 = keras.layers.Dense(self.number_of_actions, activation='relu')(action_input)
        x2 = keras.layers.Dense(64, activation='relu')(action_input)
        x2 = keras.layers.Dense(64, activation='tanh')(action_input)
        x2 = keras.layers.concatenate([commom_layer, x2])
        x2 = keras.layers.Dense(512, activation='relu')(x2)
        x2 = keras.layers.Dense(128, activation='tanh')(x2)
        x2 = keras.layers.Dense(128, activation='relu')(x2)
        reward_output = keras.layers.Dense(3, activation="softmax", name="reward_output")(x2)

        model = keras.Model([action_image_input, action_input], [reward_output], name="yingshaoxo_and_mario_reward_model")
        #print(model.summary())

        model.compile(
            optimizer=keras.optimizers.Adam(0.1),
            loss={
                "reward_output": "huber",
            },
            loss_weights={
                "reward_output": 1,
            },
            metrics=["accuracy"],
        )

        return model

    def generate_action_model(self):
        action_image_input = keras.Input(shape=(self.continuous_image_number, self.img_rows, self.img_cols, 1), name="action_image_input")
        x = keras.layers.Conv3D(self.continuous_image_number, 8, strides=8, padding='same', activation='relu')(action_image_input)
        x = keras.layers.Conv3D(32, 4, strides=2, padding='same', activation='relu')(x)
        x = keras.layers.Conv3D(64, 3, strides=1, padding='same', activation='relu')(x)
        x = keras.layers.Dense(32, activation='relu')(x)
        x = keras.layers.Flatten()(x)
        commom_layer = keras.layers.Dense(512, activation='relu')(x)

        x2 = keras.layers.Dense(512, activation='relu')(commom_layer)
        x2 = keras.layers.Dense(256, activation='relu')(x2)
        x2 = keras.layers.Dense(128, activation='relu')(x2)
        action_output = keras.layers.Dense(self.number_of_actions, activation="softmax", name="action_output")(x2)

        model = keras.Model([action_image_input], [action_output], name="yingshaoxo_and_mario_action_model")
        #print(model.summary())

        model.compile(
            optimizer=keras.optimizers.Adam(0.001),
            loss={
                "action_output": "sparse_categorical_crossentropy",
            },
            loss_weights={
                "action_output": 1,
            },
            metrics=["accuracy"],
        )

        return model

    def load_model(self):
        if os.path.exists(self.reward_model_file_path):
            self.reward_model = tf.keras.models.load_model(self.reward_model_file_path)
            print("reward model loaded.")
        if os.path.exists(self.action_model_file_path):
            self.action_model = tf.keras.models.load_model(self.action_model_file_path)
            print("action model loaded.")
    
    def save_model(self):
        self.reward_model.save(self.reward_model_file_path)
        self.action_model.save(self.action_model_file_path)
        print("model saved.")
    
    def image_process(self, frame: np.ndarray):
        if frame is not None:
            width, height = frame.shape[1], frame.shape[0]
            frame = frame[40:height, 0:width]
            #print(frame.shape)
            #(240,256,3)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #print(frame.shape)
            #(240,256)
        else:
            # frame = cv2.zeros((3, self.image_one_side_length, self.image_one_side_length)) #may it is not a good idea to put 0 here, may cause other weights get into 0 quickly
            frame = cv2.zeros((self.img_rows, self.img_cols))
            frame.fill(255)
        frame = cv2.resize(frame, (self.img_cols, self.img_rows))
        frame = frame / 255
        frame = np.expand_dims(frame, axis=2) # for 2d image, it should be (y,x,3) or (y,x,1)
        cv2.imshow('mario_view', frame)
        return frame
    
    def collect_random_data(self):
        max_steps_per_episode = 500

        state_history_for_continuous_input = []
        state_history = []
        action_history = []
        real_rewards_history = []

        last_reward = None
        episode_count = 0

        while 1:
            done = False
            state, _ = self.env.reset()
            state = self.image_process(state)
            state_history_for_continuous_input += [state] * self.continuous_image_number

            training_data_cache = []
            test_data_cache = []
            for step in range(max_steps_per_episode):
                if done:
                    state, _ = self.env.reset()
                    state = self.image_process(state)
                    state_history_for_continuous_input += [state] * self.continuous_image_number

                state_history.append([one.tolist() for one in state_history_for_continuous_input[-self.continuous_image_number:]])

                action = np.random.choice(self.number_of_actions)
                action_history.append(action)

                jump_counting = 0
                reward_adding_count = 0
                temp_reward = 0
                while jump_counting < self.time_jump_number:
                    result = self.env.step(action)
                    state, reward, terminated1, terminated2, info = result
                    state = self.image_process(state)
                    state_history_for_continuous_input += [state]
                    temp_reward += reward
                    reward_adding_count += 1
                    done = terminated1 or terminated2
                    self.env.render()
                    if done == True:
                        break
                    jump_counting += 1
                temp_reward /= reward_adding_count
                print(temp_reward)
                real_rewards_history.append(temp_reward)
                last_reward = temp_reward

                the_data = {
                    "state": state_history[-1],
                    "action": action_history[-1],
                    "reward": real_rewards_history[-1]
                }
                if step % 10 == 0:
                    # self.test_data_collection.insert_one(the_data.copy())
                    test_data_cache.append(the_data.copy())
                else:
                    # self.training_data_collection.insert_one(the_data.copy())
                    training_data_cache.append(the_data.copy())

            self.test_data_collection.insert_many(test_data_cache)
            self.training_data_collection.insert_many(training_data_cache)

            state_history.clear()
            action_history.clear()
            real_rewards_history.clear()
            state_history_for_continuous_input.clear()

            print(f"episode {episode_count}, last_reward: {last_reward}")
            episode_count += 1

        self.env.close()
    
    def _reward_number_list_to_category_reward_list(self, reward_list):
        result = []
        for reward in reward_list:
            if reward < 0:
                result.append([1, 0, 0])
            elif reward == 0:
                result.append([0, 1, 0])
            elif reward > 0:
                result.append([0, 0, 1])
        return result

    def use_collect_random_data_to_train_reward_model(self, train_single_time=False):
        """
        {
            "state": state_history[-1].tolist(),
            "action": predict_action_history[-1],
            "reward": real_rewards_history[-1]
        }
        """
        page_size = 300
        # page_size = 1
        page_number = 0
        while True:
            state_history = []
            action_history = []
            real_rewards_history = []
            # for one in self.training_data_collection.find().skip(page_number*page_size).limit(page_size):
            for one in self.training_data_collection.aggregate([
                { "$sample": { "size": page_size } }
            ]):
                state_history.append(one["state"])
                action_history.append(one["action"])
                real_rewards_history.append(one["reward"])

            data_x = {
                "action_image_input": np.array(state_history), 
                "action_input": np.array(action_history),
            }
            data_y = {
                "reward_output": np.array(self._reward_number_list_to_category_reward_list(real_rewards_history)),
            }
            for i in range(3):
                self.reward_model.fit(
                    x=data_x, 
                    y=data_y,
                )

            if page_number % 3 == 0:
                self.save_model()
                page_number = 0
                
            page_number += 1

            if train_single_time == True:
                break

    def use_reward_model_to_run(self):
        max_steps_per_episode = 500

        state_history_for_continuous_input = []
        state_history = []
        action_history = []
        real_rewards_history = []

        last_reward = None
        episode_count = 0

        while 1:
            done = False
            state, _ = self.env.reset()
            state = self.image_process(state)
            state_history_for_continuous_input += [state] * self.continuous_image_number

            training_data_cache = []
            test_data_cache = []
            for step in range(max_steps_per_episode):
                if done:
                    state, _ = self.env.reset()
                    state = self.image_process(state)
                    state_history_for_continuous_input += [state] * self.continuous_image_number

                state_history.append([one.tolist() for one in state_history_for_continuous_input[-self.continuous_image_number:]])

                # action = np.random.choice(self.number_of_actions)
                reward_result = self.reward_model.predict(
                    x= {
                        "action_image_input": np.array([state_history[-1]] * self.number_of_actions), 
                        "action_input": np.array([i for i in range(self.number_of_actions)]),
                    }, 
                )

                action_probability = np.copy(reward_result)
                temp_array = np.empty((0,))
                for one in action_probability:
                    category = np.argmax(one)
                    if category == 2:
                        # positive reward
                        temp_array = np.append(temp_array, 1)
                    else:
                        temp_array = np.append(temp_array, 0)
                temp_array /= temp_array.sum()
                action = np.random.choice(self.number_of_actions, p=np.squeeze(temp_array))

                action_history.append(action)

                jump_counting = 0
                reward_adding_count = 0
                temp_reward = 0
                while jump_counting < self.time_jump_number:
                    result = self.env.step(action)
                    state, reward, terminated1, terminated2, info = result
                    state = self.image_process(state)
                    state_history_for_continuous_input += [state]
                    temp_reward += reward
                    reward_adding_count += 1
                    done = terminated1 or terminated2
                    self.env.render()
                    if done == True:
                        break
                    jump_counting += 1
                temp_reward /= reward_adding_count
                real_rewards_history.append(temp_reward)
                last_reward = temp_reward

                the_data = {
                    "state": state_history[-1],
                    "action": action_history[-1],
                    "reward": real_rewards_history[-1]
                }
                if step % 10 == 0:
                    test_data_cache.append(the_data)
                else:
                    training_data_cache.append(the_data)

            self.test_data_collection.insert_many(test_data_cache)
            self.training_data_collection.insert_many(training_data_cache)

            state_history.clear()
            action_history.clear()
            real_rewards_history.clear()
            state_history_for_continuous_input.clear()

            print(f"episode {episode_count}, last_reward: {last_reward}")
            episode_count += 1

        self.env.close()

    def loop1_reward_model_train_and_predict(self):
        max_steps_per_episode = 300

        state_history_for_continuous_input = []
        state_history = []
        action_history = []
        real_rewards_history = []

        last_reward = None
        episode_count = 0

        while 1:
            done = False
            state, _ = self.env.reset()
            state = self.image_process(state)
            state_history_for_continuous_input += [state] * self.continuous_image_number

            training_data_cache = []
            test_data_cache = []
            for step in range(max_steps_per_episode):
                if done:
                    state, _ = self.env.reset()
                    state = self.image_process(state)
                    state_history_for_continuous_input += [state] * self.continuous_image_number

                state_history.append([one.tolist() for one in state_history_for_continuous_input[-self.continuous_image_number:]])

                reward_result = self.reward_model.predict(
                    x= {
                        "action_image_input": np.array([state_history[-1]] * self.number_of_actions), 
                        "action_input": np.array([i for i in range(self.number_of_actions)]),
                    }, 
                )
                temp_array = np.empty((0,))
                for one in reward_result:
                    category = np.argmax(one)
                    if category == 2:
                        # positive reward
                        temp_array = np.append(temp_array, 1)
                    else:
                        temp_array = np.append(temp_array, 0)
                temp_array /= temp_array.sum()
                action = np.random.choice(self.number_of_actions, p=np.squeeze(temp_array))
                action_history.append(action)

                jump_counting = 0
                reward_adding_count = 0
                temp_reward = 0
                while jump_counting < self.time_jump_number:
                    result = self.env.step(action)
                    state, reward, terminated1, terminated2, info = result
                    state = self.image_process(state)
                    state_history_for_continuous_input += [state]
                    temp_reward += reward
                    reward_adding_count += 1
                    done = terminated1 or terminated2
                    self.env.render()
                    if done == True:
                        break
                    jump_counting += 1
                temp_reward /= reward_adding_count
                real_rewards_history.append(temp_reward)
                last_reward = temp_reward

                the_data = {
                    "state": state_history[-1],
                    "action": action_history[-1],
                    "reward": real_rewards_history[-1]
                }
                if step % 10 == 0:
                    test_data_cache.append(the_data)
                else:
                    training_data_cache.append(the_data)

            self.test_data_collection.insert_many(test_data_cache)
            self.training_data_collection.insert_many(training_data_cache)

            for i in range(3):
                self.reward_model.fit(
                    x= {
                        "action_image_input": np.array(state_history), 
                        "action_input": np.array(action_history),
                    }, 
                    y={
                        "reward_output": np.array(self._reward_number_list_to_category_reward_list(real_rewards_history)),
                    },
                )
            self.save_model()
            self.use_collect_random_data_to_train_reward_model(train_single_time=True)

            state_history.clear()
            action_history.clear()
            real_rewards_history.clear()
            state_history_for_continuous_input.clear()

            print(f"episode {episode_count}, last_reward: {last_reward}")
            episode_count += 1

        self.env.close()

    def run(self):
        return


if __name__ == "__main__":
    trainer = Trainer()
    trainer.load_model()

    # trainer.collect_random_data()
    # trainer.use_collect_random_data_to_train_reward_model()

    # trainer.use_reward_model_to_run()

    trainer.loop1_reward_model_train_and_predict()