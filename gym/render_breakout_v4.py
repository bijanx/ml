import gym
env = gym.make('BreakoutDeterministic-v4')
frame = env.reset()
env.render()

while True:
  frame, reward, is_done, _ = env.step(env.action_space.sample())
  env.render()
