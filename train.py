# 🏋️ training script:

import gym
import torch
import torch.optim as optim
from smart_microgrid_env import SmartMicrogridEnv
from ppo_agent import PolicyNetwork, ValueNetwork
import numpy as np

env = SmartMicrogridEnv()
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

policy_net = PolicyNetwork(obs_dim, action_dim)
value_net = ValueNetwork(obs_dim)
policy_optimizer = optim.Adam(policy_net.parameters(), lr=3e-4)
value_optimizer = optim.Adam(value_net.parameters(), lr=1e-3)

def compute_returns(rewards, gamma=0.99):
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns

def train():
    for episode in range(500):
        obs = env.reset()
        log_probs, rewards, values, states = [], [], [], []

        for t in range(200):
            obs_tensor = torch.FloatTensor(obs)
            dist = policy_net(obs_tensor)
            value = value_net(obs_tensor)

            action = dist.sample()
            log_prob = dist.log_prob(action).sum()

            next_obs, reward, done, _ = env.step(action.detach().numpy())
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)
            states.append(obs_tensor)

            obs = next_obs

        returns = compute_returns(rewards)
        returns = torch.tensor(returns)
        values = torch.cat(values)
        log_probs = torch.stack(log_probs)

        advantage = returns - values.detach().squeeze()
        policy_loss = -(log_probs * advantage).mean()
        value_loss = (returns - values.squeeze()).pow(2).mean()

        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

        if episode % 10 == 0:
            print(f"Episode {episode}, Reward Sum: {sum(rewards):.2f}")

if __name__ == "__main__":
    train()
