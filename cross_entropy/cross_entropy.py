import typing as t

import gym
import numpy as np


class Way:

    def __init__(self):
        self.total_reward = 0
        self.state_action_history = []

    def append(self, state: int, action: int, reward: float) -> None:
        self.state_action_history.append((state, action))
        self.total_reward += reward


class ReplayBuffer:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.ways = []

    def append(self, way: Way) -> None:
        self.ways.append(way)
        self.ways = self.ways[:self.capacity]

    def get_ways_by_quantile(self, q: float) -> t.List[Way]:
        total_rewards = list(map(lambda w: w.total_reward, self.ways))
        quantile_reward = np.quantile(total_rewards, q=q)
        elite_ways = filter(lambda w: w.total_reward >= quantile_reward, self.ways)
        return list(elite_ways)


class Agent:

    def __init__(self, n_action: int, n_state: int):
        if n_action < 2 or n_state < 2:
            raise ValueError('Invalid dim spaces')

        self.n_action = n_action
        self.state_dim = n_state
        self.policy_matrix = np.full((n_state, n_action), fill_value=1 / self.n_action)

    def select_action(self, state: int, greedy: bool = False):
        if state < 0 or state > self.state_dim:
            raise ValueError('Invalid input state')

        action_distribution = self.policy_matrix[state]

        if greedy:
            return np.argmax(action_distribution)

        action_probs = action_distribution / np.sum(action_distribution)
        action = np.random.choice(range(self.n_action), p=action_probs)
        return action

    def update_policy(self, buffer: ReplayBuffer, q: float, alpha: float = 0.5) -> None:
        elite_ways = buffer.get_ways_by_quantile(q)
        elite_policy_matrix = np.zeros((self.state_dim, self.n_action))

        for way in elite_ways:
            for state, action in way.state_action_history:
                elite_policy_matrix[state, action] += 1

        denom = elite_policy_matrix.sum(axis=1)[:, np.newaxis]
        denom = np.clip(denom, 1, np.inf)
        elite_policy_matrix /= denom

        self.policy_matrix = (1 - alpha) * self.policy_matrix + alpha * elite_policy_matrix
        self.policy_matrix /= self.policy_matrix.sum(axis=1)[:, np.newaxis]


if __name__ == '__main__':
    env = gym.make('FrozenLake-v0', is_slippery=False)  # deterministic
    agent = Agent(env.action_space.n, env.observation_space.n)
    replay_buffer = ReplayBuffer(capacity=2000)

    for n in range(10_000):

        state = env.reset()
        final_reward = 0
        n_steps = 0

        way = Way()

        while True:
            action = agent.select_action(state)
            new_state, reward, done, _ = env.step(action)
            final_reward += reward
            n_steps += 1

            if n_steps > 30:
                done = True

            if reward > 0:
                env.render()

            if done:
                if reward > 0 and n_steps < 10:
                    reward += 100
                way.append(state, action, reward)
                replay_buffer.append(way)
                break

            way.append(state, action, reward)
            state = new_state

        if n > 0 and n % 1500 == 0:
            agent.update_policy(buffer=replay_buffer, q=0.99, alpha=0.5)

    state = env.reset()
    final_reward = 0
    n_steps = 0

    while True:
        action = agent.select_action(state, greedy=True)
        new_state, reward, done, _ = env.step(action)

        state = new_state
        final_reward += reward
        n_steps += 1

        if done:
            break

        env.render()
