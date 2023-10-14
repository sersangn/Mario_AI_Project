from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
from matplotlib import pyplot as plt
from gym.wrappers import GrayScaleObservation
from PIL import Image
import numpy as np
from numpy import asarray
from matplotlib import image as img
import game_simple

env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = GrayScaleObservation(env)

done = True
state = env.reset()
plt.imshow(state[0])
ugh = state[0][-16:][:, :16]

new_state = game_simple.get_simple_game_state(state)


# ugh2 = state[0][-16:][0:,:16]

# print(ugh2)

# block = game_simple.get_block()

# if(block == state):
#     print("yes")
# else:
#     print("shit")

# print(ugh)

# print("######")

# # print(block)

# block = np.asarray(block)

# if (ugh == block).all():
#     print("YES")
# else:
#     print("FUCK")

# image = Image.open('block2.png')


# data = asarray(image)

data = np.dot(data[...,:3],[.2989, .5870, .1140])
data = np.round(data).astype(np.uint8)

# print(data)



# for step in range(100000):
#     action = env.action_space.sample()
#     obs, reward, terminated, truncated, info = env.step(action)
#     done = terminated or truncated
#     print(obs)

#     if done:
#         state = env.reset()

# env.close()