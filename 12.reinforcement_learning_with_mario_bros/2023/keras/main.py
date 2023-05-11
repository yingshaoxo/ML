import numpy as np
import tensorflow as tf
tf.keras.utils.disable_interactive_logging()
from tensorflow import keras

from random import randint
import os
import cv2


class Trainer:
    def __init__(self):
        from nes_py.wrappers import JoypadSpace
        import gym_super_mario_bros

        self.env = gym_super_mario_bros.make('SuperMarioBros-v2', apply_api_compatibility=True, render_mode="human")
        MY_MARIO_MOVEMENT = [
            ['NOOP'],
            ['A', 'right', 'B'],
            ['A', 'left', 'B'],
        ]
        self.number_of_actions = len(MY_MARIO_MOVEMENT)
        self.env = JoypadSpace(self.env, MY_MARIO_MOVEMENT)

        self.img_rows , self.img_cols = 240, 256

        self.model_file_path = './nn_keras_model'
        self.model = self.generate_model()

    def generate_model(self):
        action_image_input = keras.Input(shape=(self.img_rows, self.img_cols, 1), name="action_image_input")
        x = keras.layers.Conv2D(32, 3, strides=(2,2), activation='relu')(action_image_input)
        x = keras.layers.Conv2D(32, 3, strides=(2,2), activation='relu')(x)
        x = keras.layers.Conv2D(32, 3, strides=(2,2), activation='relu')(x)
        x = keras.layers.Dense(32, activation='relu')(x)
        x = keras.layers.Flatten()(x)
        commom_layer = keras.layers.Dense(32, activation='relu')(x)

        action_output = keras.layers.Dense(self.number_of_actions, activation="softmax", name="action_output")(commom_layer)
        reward_output = keras.layers.Dense(1, name="reward_output")(commom_layer)

        model = keras.Model([action_image_input], [action_output, reward_output], name="yingshaoxo_and_mario")
        #print(model.summary())

        model.compile(
            optimizer=keras.optimizers.Adam(0.001),
            loss={
                "action_output": "sparse_categorical_crossentropy",
                "reward_output": "mean_squared_error",
            },
            loss_weights={
                "action_output": 1,
                "reward_output": 1,
            },
            metrics=["accuracy"],
        )

        return model

    def load_model(self):
        if os.path.exists(self.model_file_path):
            self.model = tf.keras.models.load_model(self.model_file_path)
    
    def save_model(self):
        self.model.save(self.model_file_path)
    
    def image_process(self, frame: np.ndarray):
        if frame is not None:
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # width, height = frame.shape[1], frame.shape[0]
            # frame = frame[40:height, 0:width]
            # frame = cv2.resize(frame, (self.image_one_side_length, self.image_one_side_length)).astype(cv2.uint8)

            #print(frame.shape)
            #(240,256,3)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #print(frame.shape)
            #(240,256)
        else:
            # frame = cv2.zeros((3, self.image_one_side_length, self.image_one_side_length)) #may it is not a good idea to put 0 here, may cause other weights get into 0 quickly
            frame = cv2.zeros((self.img_rows, self.img_cols))
            frame.fill(255)
        frame = frame.reshape(self.img_rows, self.img_cols, 1)
        frame = frame / 255
        return frame
    
    def run(self):
        # env.action_space.sample() = numbers, for example, 0,1,2,3...
        # state = RGB of raw picture; is a numpy array with shape (240, 256, 3)
        # reward = int; for example, 0, 1 ,2, ...
        # done = False or True
        # info = {'coins': 0, 'flag_get': False, 'life': 3, 'score': 0, 'stage': 1, 'status': 'small', 'time': 400, 'world': 1, 'x_pos': 40}

        gamma = 0.99  # Discount factor for past rewards
        max_steps_per_episode = 3000
        eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0

        done = True
        last_state = None
        identity = np.identity(self.number_of_actions) # for quickly get a hot vector, like 0001000000000000

        state_history = []
        action_history = []
        predict_rewards_history = []
        real_rewards_history = []
        running_reward = 0
        episode_count = 0

        no_random_level = 3 # switch to 1000 to see what mario has learned
        while 1:
            state, _ = self.env.reset()
            state = self.image_process(state)
            episode_reward = 0

            for step in range(max_steps_per_episode):
                if done:
                    state, _ = self.env.reset()
                    state = self.image_process(state)

                if not isinstance(state, (np.ndarray, np.generic)):
                    action = self.env.action_space.sample()
                else:
                    predict_action_probability, predict_reward = self.model.predict(
                        {
                            "action_image_input": np.expand_dims(state, axis=0), 
                        }
                    )
                    predict_rewards_history.append(predict_reward[0, 0])

                    if randint(0, no_random_level) == 0:
                        action = self.env.action_space.sample()
                    else:
                        # predict_action_probability = predict_action_probability[0]
                        # predict_action_probability /= predict_action_probability.sum()
                        # action = np.random.choice(self.number_of_actions, p=np.squeeze(predict_action_probability))
                        action = np.random.choice(self.number_of_actions, p=np.squeeze(predict_action_probability[0]))

                    action_history.append(action)

                result = self.env.step(action)
                if len(result) == 5:
                    state, reward, terminated1, terminated2, info = result
                    done = terminated1 or terminated2
                    state = self.image_process(state)

                    state_history.append(state)
                    real_rewards_history.append(reward)
                    episode_reward += reward

                    self.env.render()
                else:
                    continue

            running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
            continues_real_rewards = []
            discounted_sum = 0
            for reward in real_rewards_history[::-1]:
                discounted_sum = reward + gamma * discounted_sum
                continues_real_rewards.insert(0, discounted_sum)

            continues_real_rewards = np.array(continues_real_rewards)
            continues_real_rewards = (continues_real_rewards - np.mean(continues_real_rewards)) / (np.std(continues_real_rewards) + eps)
            continues_real_rewards = continues_real_rewards.tolist()

            temp_x = []
            temp_y = []
            for i in range(len(real_rewards_history)):
                the_x = {
                        "action_image_input": tf.expand_dims(state_history[i], axis=0), 
                    } 
                the_y = {
                        "action_output": tf.expand_dims(action_history[i], axis=0),
                        "reward_output": tf.expand_dims(continues_real_rewards[i], axis=0),
                }
                temp_x.append(the_x)
                temp_y.append(the_y)
            self.model.fit(
                x=temp_x, 
                y=temp_y,
            )

            state_history.clear()
            action_history.clear()
            predict_rewards_history.clear()
            real_rewards_history.clear()
            continues_real_rewards.clear()

            episode_count += 1
            template = "running reward: {:.2f} at episode {}"
            print(template.format(running_reward, episode_count))

            self.save_model()

        self.env.close()


if __name__ == "__main__":
    trainer = Trainer()
    trainer.load_model()
    trainer.run()