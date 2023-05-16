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
            ['A', 'left'],
            ['A', 'right'],
            ['left'],
            ['right'],
        ]
        self.number_of_actions = len(MY_MARIO_MOVEMENT)
        self.env = JoypadSpace(self.env, MY_MARIO_MOVEMENT)

        self.img_rows , self.img_cols = 240, 256

        self.model_file_path = './nn_keras_model'
        self.model = self.generate_model()

    def generate_model(self):
        action_image_input = keras.Input(shape=(self.img_rows, self.img_cols, 1), name="action_image_input")
        x = keras.layers.Conv2D(32, 8, strides=(4,4), activation='relu')(action_image_input)
        x = keras.layers.Conv2D(32, 4, strides=(2,2), activation='relu')(x)
        x = keras.layers.Conv2D(64, 3, strides=(1,1), activation='relu')(x)
        x = keras.layers.Dense(32, activation='relu')(x)
        x = keras.layers.Flatten()(x)
        commom_layer = keras.layers.Dense(512, activation='relu')(x)

        x2 = keras.layers.Dense(128, activation='relu')(commom_layer)
        x2 = keras.layers.Dense(128, activation='relu')(x2)
        action_output = keras.layers.Dense(self.number_of_actions, activation="softmax", name="action_output")(x2)

        reward_output = keras.layers.Dense(1, activation="linear", name="reward_output")(commom_layer)

        model = keras.Model([action_image_input], [action_output, reward_output], name="yingshaoxo_and_mario")
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

    def load_model(self):
        if os.path.exists(self.model_file_path):
            self.model = tf.keras.models.load_model(self.model_file_path)
    
    def save_model(self):
        self.model.save(self.model_file_path)
    
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
        # frame = np.expand_dims(frame, axis=2) # for 2d image, it should be (y,x,3) or (y,x,1)
        cv2.imshow('mario_seen', frame)
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

                predict_action_probabilitys, predict_reward = self.model.predict(
                    {
                        "action_image_input": np.expand_dims(state, axis=0), 
                    }
                )
                predict_rewards_history.append(predict_reward[0][0])
                print(predict_reward[0][0])

                action = np.random.choice(self.number_of_actions, p=np.squeeze(predict_action_probabilitys[0]))
                predict_action_history.append(action)
                predict_action_probability_history.append(tf.math.log(predict_action_probabilitys[0][action]))
                last_predict_action_probability_distribution = predict_action_probabilitys[0]

                result = self.env.step(action)

                state, reward, terminated1, terminated2, info = result
                done = terminated1 or terminated2
                state = self.image_process(state)

                real_rewards_history.append(reward)
                episode_reward += reward

                self.env.render()

            running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
            continues_real_rewards = []
            discounted_sum = 0
            for reward in real_rewards_history[::-1]:
                discounted_sum = reward + gamma * discounted_sum
                continues_real_rewards.insert(0, discounted_sum)

            continues_real_rewards = np.array(continues_real_rewards)
            continues_real_rewards = (continues_real_rewards - np.mean(continues_real_rewards)) / (np.std(continues_real_rewards) + eps)
            continues_real_rewards = continues_real_rewards.tolist()

            self.model.fit(
                x= {
                    "action_image_input": np.array(state_history), 
                }, 
                y={
                    "action_output": np.array(predict_action_history),
                    "reward_output": np.array(continues_real_rewards),
                },
            )

            episode_count += 1
            state_history.clear()
            predict_action_history.clear()
            predict_action_probability_history.clear()
            predict_rewards_history.clear()
            real_rewards_history.clear()
            continues_real_rewards.clear()

            template = "running reward: {:.2f} at episode {}, last_action_probability_distribution: {}"
            print(template.format(running_reward, episode_count, last_predict_action_probability_distribution))

            self.save_model()

        self.env.close()


if __name__ == "__main__":
    trainer = Trainer()
    trainer.load_model()
    trainer.run()