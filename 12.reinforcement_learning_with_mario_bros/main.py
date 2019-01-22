from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
env = gym_super_mario_bros.make('SuperMarioBros-v1')
env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)

from random import randint
import numpy as np
import os
import tensorflow as tf
from model import generate_model

model_file_path = './nn_model.HDF5'
if os.path.exists(model_file_path):
    model = tf.keras.models.load_model(model_file_path)
else:
    model = generate_model()

# env.action_space.sample() = numbers, for example, 0,1,2,3...
# state = RGB of raw picture; is a numpy array with shape (240, 256, 3)
# reward = int; for example, 0, 1 ,2, ...
# done = False or True
# info = {'coins': 0, 'flag_get': False, 'life': 3, 'score': 0, 'stage': 1, 'status': 'small', 'time': 400, 'world': 1, 'x_pos': 40}

done = True
last_state = None
identity = np.identity(env.action_space.n) # for quickly get a hot vector, like 0001000000000000

while 1:
    for step in range(5000):
        if done:
            state = env.reset()

        if randint(0, 10) == 0 or not isinstance(last_state, (np.ndarray, np.generic)):
            action = env.action_space.sample()
        else:
            action = np.argmax(model.predict(np.expand_dims(last_state, axis=0)))
            #print(action)

        state, reward, done, info = env.step(action)
        last_state = state
        if reward > 0:
            model.train_on_batch(x=np.expand_dims(last_state, axis=0), y=identity[action: action+1])

        env.render()
    model.save(model_file_path)

env.close()
