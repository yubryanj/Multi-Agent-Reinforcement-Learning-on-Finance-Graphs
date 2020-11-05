import gym
from gym import error, spaces, utils

class Financial_Network_Env(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    print("Running Financial Environment")
    pass
  def step(self, action):
    pass
  def reset(self):
    pass
  def render(self, mode='human'):
    pass
  def close(self):
    pass
