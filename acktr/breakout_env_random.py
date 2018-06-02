import gym

env = gym.make('Breakout-v0')
observation = env.reset()
done = False
t = 0
while not done:
    env.render()
    action = env.action_space.sample()  # choose random action
    observation, reward, done, info = env.step(action)  # feedback from environment
    t += 1
    if not t % 100:
        print(t, info)

