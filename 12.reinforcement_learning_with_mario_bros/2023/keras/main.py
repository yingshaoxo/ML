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
            ['NOOP'],
            ['A', 'right', 'B'],
            ['A', 'left', 'B'],
            ['A', 'right'],
            ['A', 'left'],
            ['right'],
            ['left'],
        ]
        self.number_of_actions = len(MY_MARIO_MOVEMENT)
        self.env = JoypadSpace(self.env, MY_MARIO_MOVEMENT)

        # self.img_rows , self.img_cols = 240, 256
        self.img_rows , self.img_cols = 80, 80
        self.continuous_image_number = 8
        self.time_jump_number = 25

        self.reward_model_file_path = './reward_nn_keras_model'
        self.action_model_file_path = './action_nn_keras_model'

        self.reward_model = self.generate_reward_model()
        self.action_model = self.generate_action_model()

        self.mongo_client = MongoClient('mongodb://localhost:27017/')
        self.training_data_collection = self.mongo_client['mario']['training_data_collection']
        self.test_data_collection = self.mongo_client['mario']['test_data_collection']

    def generate_reward_model(self):
        action_image_input = keras.Input(shape=(self.continuous_image_number, self.img_rows, self.img_cols, 1), name="action_image_input")
        x = keras.layers.Conv3D(self.continuous_image_number, 8, strides=4, padding='same', activation='relu')(action_image_input)
        x = keras.layers.Conv3D(32, 4, strides=2, padding='same', activation='relu')(x)
        x = keras.layers.Conv3D(64, 3, strides=1, padding='same', activation='relu')(x)
        x = keras.layers.Dense(32, activation='relu')(x)
        x = keras.layers.Flatten()(x)
        commom_layer = keras.layers.Dense(512, activation='relu')(x)

        action_input = keras.Input(shape=(1), name="action_input")
        x2 = keras.layers.Dense(self.number_of_actions, activation='relu')(action_input)
        x2 = keras.layers.Dense(512, activation='relu')(action_input)
        x2 = keras.layers.concatenate([commom_layer, x2])
        x2 = keras.layers.Dense(512, activation='relu')(x2)
        x2 = keras.layers.Dense(256, activation='relu')(x2)
        x2 = keras.layers.Dense(128, activation='relu')(x2)
        reward_output = keras.layers.Dense(1, activation="linear", name="reward_output")(x2)

        model = keras.Model([action_image_input, action_input], [reward_output], name="yingshaoxo_and_mario_reward_model")
        #print(model.summary())

        model.compile(
            optimizer=keras.optimizers.Adam(0.01),
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
        x = keras.layers.Conv3D(self.continuous_image_number, 8, strides=4, padding='same', activation='relu')(action_image_input)
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
        cv2.imshow('mario_seen', frame)
        return frame
    
    def collect_random_data(self):
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

                action = np.random.choice(self.number_of_actions)
                action_history.append(action)

                jump_counting = 0
                reward_counting = 0
                while jump_counting < self.time_jump_number:
                    result = self.env.step(action)
                    state, reward, terminated1, terminated2, info = result
                    state = self.image_process(state)
                    state_history_for_continuous_input += [state]
                    reward_counting += reward
                    done = terminated1 or terminated2
                    self.env.render()
                    if done == True:
                        break
                    jump_counting += 1
                real_rewards_history.append(reward_counting)
                last_reward = reward_counting

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

    def use_collect_random_data_to_train_reward_model(self):
        """
        {
            "state": state_history[-1].tolist(),
            "action": predict_action_history[-1],
            "reward": real_rewards_history[-1]
        }
        """
        page_size = 200
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
                "reward_output": np.array(real_rewards_history),
            }
            for i in range(10):
                self.reward_model.fit(
                    x=data_x, 
                    y=data_y,
                )

            if page_number % 3 == 0:
                self.save_model()
                page_number = 0
                
            page_number += 1

    def use_reward_model_to_run(self):
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

                # action = np.random.choice(self.number_of_actions)
                reward_result = self.reward_model.predict(
                    x= {
                        "action_image_input": np.array([state_history[-1]] * self.number_of_actions), 
                        "action_input": np.array([i for i in range(self.number_of_actions)]),
                    }, 
                )

                # for x in np.copy(reward_result):
                #     if x < 0:
                #         reward_result += np.absolute(x)
                reward_result[reward_result < 0] = 0
                action_probability = np.copy(reward_result)
                action_probability /= action_probability.sum()
                action = np.random.choice(self.number_of_actions, p=np.squeeze(action_probability))

                # action = np.argmax(np.squeeze(reward_result))

                action_history.append(action)

                jump_counting = 0
                reward_counting = 0
                while jump_counting < self.time_jump_number:
                    result = self.env.step(action)
                    state, reward, terminated1, terminated2, info = result
                    state = self.image_process(state)
                    state_history_for_continuous_input += [state]
                    reward_counting += reward
                    done = terminated1 or terminated2
                    self.env.render()
                    if done == True:
                        break
                    jump_counting += 1
                real_rewards_history.append(reward_counting)
                last_reward = reward_counting

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
                for x in np.copy(reward_result):
                    if x < 0:
                        reward_result += np.absolute(x)
                # reward_result[reward_result < 0] = 0
                action_probability = np.copy(reward_result)
                action_probability /= action_probability.sum()
                action = np.random.choice(self.number_of_actions, p=np.squeeze(action_probability))
                action_history.append(action)

                jump_counting = 0
                reward_counting = 0
                while jump_counting < self.time_jump_number:
                    result = self.env.step(action)
                    state, reward, terminated1, terminated2, info = result
                    state = self.image_process(state)
                    state_history_for_continuous_input += [state]
                    reward_counting += reward
                    done = terminated1 or terminated2
                    self.env.render()
                    if done == True:
                        break
                    jump_counting += 1
                real_rewards_history.append(reward_counting)
                last_reward = reward_counting

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

            for i in range(10):
                self.reward_model.fit(
                    x= {
                        "action_image_input": np.array(state_history), 
                        "action_input": np.array(action_history),
                    }, 
                    y={
                        "reward_output": np.array(real_rewards_history),
                    },
                )
            self.save_model()

            state_history.clear()
            action_history.clear()
            real_rewards_history.clear()
            state_history_for_continuous_input.clear()

            print(f"episode {episode_count}, last_reward: {last_reward}")
            episode_count += 1

        self.env.close()

    def run(self):
        return
        # env.action_space.sample() = numbers, for example, 0,1,2,3...
        # state = RGB of raw picture; is a numpy array with shape (240, 256, 3)
        # reward = int; for example, 0, 1 ,2, ...
        # done = False or True
        # info = {'coins': 0, 'flag_get': False, 'life': 3, 'score': 0, 'stage': 1, 'status': 'small', 'time': 400, 'world': 1, 'x_pos': 40}

        gamma = 0.99  # Discount factor for past rewards
        max_steps_per_episode = 3000
        eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0

        done = True

        state_history = []
        predict_action_history = []
        predict_action_probability_history = []
        last_predict_action_probability_distribution = None
        predict_rewards_history = []
        real_rewards_history = []
        running_reward = 0
        episode_count = 0

        while 1:
            state, _ = self.env.reset()
            state = self.image_process(state)
            episode_reward = 0

            for step in range(max_steps_per_episode):
                if done:
                    state, _ = self.env.reset()
                    state = self.image_process(state)

                state_history.append(state)

                action = np.random.choice(self.number_of_actions)
                predict_action_history.append(action)
                # predict_action_probabilitys, predict_reward = self.model.predict(
                #     {
                #         "action_image_input": np.expand_dims(state, axis=0), 
                #     }
                # )
                # predict_reward = predict_reward[0][0]
                # predict_rewards_history.append(predict_reward)

                # if predict_reward > -20:
                #     print('predict action because the predict reward is: ', predict_reward)
                #     action = np.random.choice(self.number_of_actions, p=np.squeeze(predict_action_probabilitys[0]))
                # else:
                #     print('random action because the predict reward is: ', predict_reward)
                #     action = np.random.choice(self.number_of_actions)
                # predict_action_history.append(action)
                # predict_action_probability_history.append(tf.math.log(predict_action_probabilitys[0][action]))
                # last_predict_action_probability_distribution = predict_action_probabilitys[0]

                jump_counting = 0
                reward_counting = 0
                while jump_counting < 30:
                    result = self.env.step(action)
                    state, reward, terminated1, terminated2, info = result
                    state = self.image_process(state)
                    reward_counting += reward
                    done = terminated1 or terminated2
                    self.env.render()
                    if done == True:
                        state, _ = self.env.reset()
                        break
                    jump_counting += 1

                real_rewards_history.append(reward_counting)
                episode_reward += reward_counting

                if self.collect_data_to_mongodb == True:
                    self.add_training_data({
                        "state": state_history[-1].tolist(),
                        "action": predict_action_history[-1],
                        "reward": real_rewards_history[-1]
                    })
                    state_history.clear()
                    predict_action_history.clear()
                    real_rewards_history.clear()

                last_predict_action_probability_distribution = reward_counting

            # running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
            # continues_real_rewards = []
            # discounted_sum = 0
            # for reward in real_rewards_history[::-1]:
            #     discounted_sum = reward + gamma * discounted_sum
            #     continues_real_rewards.insert(0, discounted_sum)

            # continues_real_rewards = np.array(continues_real_rewards)
            # continues_real_rewards = (continues_real_rewards - np.mean(continues_real_rewards)) / (np.std(continues_real_rewards) + eps)
            # continues_real_rewards = continues_real_rewards.tolist()

            # self.model.fit(
            #     x= {
            #         "action_image_input": np.array(state_history), 
            #         "action_input": np.array(predict_action_history), 
            #     }, 
            #     y={
            #         "reward_output": np.array(continues_real_rewards),
            #     },
            # )

            episode_count += 1
            state_history.clear()
            predict_action_history.clear()
            predict_action_probability_history.clear()
            predict_rewards_history.clear()
            real_rewards_history.clear()
            # continues_real_rewards.clear()

            template = "running reward: {:.2f} at episode {}, last_action_probability_distribution: {}"
            print(template.format(running_reward, episode_count, last_predict_action_probability_distribution))

            # self.save_model()

        self.env.close()


if __name__ == "__main__":
    trainer = Trainer()
    trainer.load_model()

    # trainer.collect_random_data()
    # trainer.use_collect_random_data_to_train_reward_model()

    #trainer.use_reward_model_to_run()

    trainer.loop1_reward_model_train_and_predict()