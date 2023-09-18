from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import gym
import numpy as np
import gym_super_mario_bros

#Code for a simple rule based agent

#Code for rule based agent, takes observation and info
def rule_based_agent(observation, info):
    #Extract relevant information from the observation
    #Observation is actual RGB pixel image of what you'll see on the screen
    #Info is a dictionary that contains things like x y position, whether you have a flag, misc
    player_x = info['x_pos']
    player_y = info['y_pos']

    #Define some basic rules
    if player_x >10: #Jump if Mario is above the ground
        return 2 #2 corresponds to jump action
    else:       #Default action
        return 1 #corresponds to go right action

def main():
    #Creating the environment
    env = gym.make("SuperMarioBros-v0", apply_api_compatibility=True, render_mode="human")
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    observation = env.reset()

    #Rewards
    total_reward = 0
    done = False

    #this time I'm creating a rule based agent:
    action = env.action_space.sample()
    while not done:
        obs, reward, terminated, truncated, info = env.step(action)
        action = rule_based_agent(obs,info)
        total_reward += reward
    
    print("Total reward:", total_reward)


if __name__ == "__main__":
    main()


