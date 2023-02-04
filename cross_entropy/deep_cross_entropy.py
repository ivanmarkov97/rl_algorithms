import time
import typing as t

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as opt


class Episode:
    """Episode history tracker from start to terminate state."""

    def __init__(self):
        self.total_reward = 0
        self.states_history = []
        self.actions_history = []

    def append(self, state: np.ndarray, action: int, reward: float) -> None:
        self.states_history.append(state)
        self.actions_history.append(action)
        self.total_reward += reward


class ReplayBuffer:
    """Replay buffer to store episodes and find 'elites'."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.last_episodes = []

    def append(self, episode: Episode) -> None:
        self.last_episodes.append(episode)
        self.last_episodes = self.last_episodes[:self.capacity]

    def get_elite_episodes(self, quantile: float) -> t.List[Episode]:
        min_elite_reward = np.quantile([episode.total_reward for episode in self.last_episodes], q=quantile)
        return [episode for episode in self.last_episodes if episode.total_reward >= min_elite_reward]


class Agent(nn.Module):
    """RL algorithm to take actions in particular state."""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim

        self.network = nn.Sequential(
            nn.Linear(self.state_dim, 150),
            nn.ReLU(),
            nn.Linear(150, 75),
            nn.ReLU(),
            nn.Linear(75, self.action_dim)
        )
        self.softmax = nn.Softmax(dim=-1)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = opt.Adam(self.network.parameters(), lr=0.05)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward flow to train agent."""
        x = self.network(state)
        return x

    def action(self, state: np.ndarray) -> int:
        """Take action in state with no learning."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            logits = self.network(state_tensor)
            action_probs = self.softmax(logits).detach().numpy()[0]
            action = np.random.choice(range(self.action_dim), p=action_probs)
            return action

    def update_policy(self, elite_episodes: t.List[Episode]) -> None:
        """Update RL policy from 'elite' episodes in recent history."""
        target = []
        train_data = []

        for episode in elite_episodes:
            for state, action in zip(episode.states_history, episode.actions_history):
                train_data.append(state)
                target.append(action)

        train_data = np.array(train_data)
        target = np.array(target)

        train_data_t = torch.FloatTensor(train_data)
        target_t = torch.LongTensor(target)

        self.optimizer.zero_grad()
        print(train_data_t.size())
        print(target_t.size())
        action_logits = self.network(train_data_t)
        loss = self.loss_fn(action_logits, target_t)
        loss.backward()
        self.optimizer.step()


def train_agent(
        env: gym.Env,
        agent: Agent,
        buffer: ReplayBuffer,
        quantile: float,
        n_epochs: int,
        n_episodes: int
    ) -> None:
    """
    Train agent via cross entropy method (from elite episodes).

    Args:
        env: gym.Env - Environment for agent to act.
        agent: Agent - RL algorithm.
        buffer: ReplayBuffer - Buffer to store last episodes.
        quantile: float - Threshold for reward to detect elite episodes.
        n_epochs: int - How many times update RL policy after playing n_episodes.
        n_episodes: int - How many episodes agent needs to play to update policy.
    """
    for _ in range(n_epochs):
        total_rewards = []
        for n in range(n_episodes):
            state = env.reset()
            episode = Episode()
            total_reward = 0

            while True:
                action = agent.action(state)
                new_state, reward, done, _ = env.step(action)
                episode.append(state, action, reward)
                state = new_state
                total_reward += reward

                if done:
                    break

            buffer.append(episode)
            total_rewards.append(total_reward)

        mean_episode_reward = np.mean(total_rewards)
        print('#################', mean_episode_reward)
        if mean_episode_reward >= 480:
            break
        elite_episodes = buffer.get_elite_episodes(quantile=quantile)
        agent.update_policy(elite_episodes)


def eval_agent(env: gym.Env, agent: Agent) -> float:
    """
    Evaluating agent on existing env.

    Args:
        env: gym.Env - Environment for agent to act.
        agent: Agent - RL algorithm.

    Returns:
        float - cumulative reward from start to terminate state.
    """
    state = env.reset()
    total_reward = 0

    while True:
        action = agent.action(state)
        new_state, reward, done, _ = env.step(action)
        state = new_state
        total_reward += reward

        env.render()
        time.sleep(0.05)

        if done:
            break

    return total_reward


if __name__ == '__main__':
    env = gym.make('CartPole-v1')

    action_dim = env.action_space.n
    state_dim = env.observation_space.shape[0]

    agent = Agent(state_dim, action_dim)
    buffer = ReplayBuffer(capacity=5_000)

    # train_agent(env, agent, buffer, quantile=0.99, n_epochs=10_000, n_episodes=50)
    # torch.save(agent.network.state_dict(), 'agents_store/cross_entropy/agent.pt')

    agent.network.load_state_dict(torch.load('agents_store/cross_entropy/agent.pt'))
    agent.network.eval()

    reward = eval_agent(env, agent)
    print(reward)
