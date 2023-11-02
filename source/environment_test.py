import numpy as np
from environment import SimpleEnv

# instantiate the env
length, width, capacity = 2, 2, 2
edge_capacity = np.full((length, width, 4), capacity)
macros = []
env = SimpleEnv(length, width, macros, edge_capacity)
num_episodes = 1

# render the env with random moves
for ep in range(num_episodes):
    total_reward = 0
    obs = env.reset()
    print(obs)
    done = False
    while True:
        action = env.action_space.sample()
        print(action)
        new_obs, reward, done, info = env.step(action)
        print(new_obs)
        total_reward += reward

        if done:
            break

    print(f"episode: {ep}")
    print(f"cumulative reward: {total_reward}")
    #env.render()
