import gym
import time
import os
import matplotlib.pyplot as plt
import numpy as np
from gym.envs.registration import register
from agent import QAgent

register(
    id='FrozenLake8x8NoSLip-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    # kwargs={"map_name": "8x8", 'is_slippery': False},
    kwargs={'is_slippery': True},
    max_episode_steps=200,
    reward_threshold=0.99,
    # optimum = 1
)

env = gym.make("FrozenLake8x8NoSLip-v0")
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

steps = 0
total_reward = 0
ep_rewards = []
aggr_ep_rewards = {
    'ep': [],
    'avg': [],
    'min': [],
    'max': [],
}

agent = QAgent(env)

for ep in range(200):
    episode_reward = 0
    state = env.reset()
    done = False
    while not done:
        steps += 1
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)

        if not reward == 1:
            if done:
                reward = -0.1
            else:
                reward = -0.001

        episode_reward += reward
        agent.train((state, action, next_state, reward, done))
        state = next_state
        total_reward += reward
        print("s:", state, "a:", action)
        print("Episode: {}, Total reward: {}, eps: {}".format(ep, total_reward, agent.eps))
        env.render()
        # print(agent.q_table)
        time.sleep(0.05)
        if not ep == 199:
            os.system('cls' if os.name == 'nt' else 'clear')
    ep_rewards.append(episode_reward)
    average_reward = sum(ep_rewards)/len(ep_rewards)
    aggr_ep_rewards['ep'].append(ep)
    aggr_ep_rewards['avg'].append(average_reward)
    aggr_ep_rewards['min'].append(min(ep_rewards))
    aggr_ep_rewards['max'].append(max(ep_rewards))

env.close()

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label='avg reward')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label='min reward')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label='max reward')
plt.xlabel('iterations')
plt.legend(loc=4)
plt.show()

plt.figure()
plt.plot(np.arange(0, steps, 1), agent.max_q, label='Max Q')
plt.xlabel('iterations')
plt.ylabel('max Q')
plt.legend(loc=4)
plt.show()
