import numpy as np
import tensorflow as tf
tf.keras.utils.disable_interactive_logging()

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
        model = tf.keras.models.Sequential([
            tf.keras.layers.Convolution2D(32, 2, 2, input_shape=(self.img_rows, self.img_cols, 1)),
            tf.keras.layers.Activation('relu'),

            tf.keras.layers.Convolution2D(32, 3, 3),
            tf.keras.layers.Activation('relu'),

            tf.keras.layers.Convolution2D(32, 2, 2),
            tf.keras.layers.Activation('relu'),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(32),
            tf.keras.layers.Activation('relu'),

            tf.keras.layers.Dense(self.number_of_actions, activation=tf.nn.softmax),
        ])

        model.compile(optimizer='adam',
                    loss='mse',
                    metrics=['accuracy'])

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

        done = True
        last_state = None
        identity = np.identity(self.number_of_actions) # for quickly get a hot vector, like 0001000000000000

        experimence_cache_list_for_x = []
        experimence_cache_list_for_y = []

        no_random_level = 3 # switch to 1000 to see what mario has learned
        while 1:
            for step in range(3000):
                if done:
                    state, _ = self.env.reset()

                if randint(0, no_random_level) == 0 or not isinstance(last_state, (np.ndarray, np.generic)):
                    action = self.env.action_space.sample()
                else:
                    result = self.model.predict(np.expand_dims(last_state, axis=0))

                    action_probability = result[0]
                    action_probability /= action_probability.sum()
                    action = np.random.choice(self.number_of_actions, p=np.squeeze(action_probability))
                    
                    # action = np.argmax(result)

                    # print(f"                                             {action}")

                result = self.env.step(action)
                if len(result) == 5:
                    state, reward, terminated1, terminated2, info = result
                    state = self.image_process(state)
                    done = terminated1 or terminated2
                else:
                    continue
                last_state = state

                experimence_cache_list_for_x.append(last_state)
                experimence_cache_list_for_y.append(identity[action: action+1])

                if len(experimence_cache_list_for_x) == 150:
                    print(f"traning..., in step {step}", flush=True)
                    self.model.train_on_batch(x=np.array(experimence_cache_list_for_x), y=np.array(experimence_cache_list_for_y))
                    experimence_cache_list_for_x = []
                    experimence_cache_list_for_y = []

                self.env.render()

            self.save_model()

        self.env.close()


if __name__ == "__main__":
    trainer = Trainer()
    trainer.load_model()
    trainer.run()