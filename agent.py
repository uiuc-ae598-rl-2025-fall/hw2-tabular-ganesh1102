import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

class Agent:
    def __init__(
            self,
            env: gym.Env,
            gamma: float,
            lr: float,
            epsilon_start: float,
    ):
        self.env = env
        self.gamma = gamma
        self.lr = lr
        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start

        self.num_state = env.observation_space.n
        self.num_action = env.action_space.n

        self.Q = np.zeros((self.num_state, self.num_action))

        self.eval_return = []
        self.eval_return.append(np.max(self.Q[0, :]))

    def epsilon_greedy(self, s, eps):
        if np.random.rand() < eps:
            return np.random.choice(self.num_action)
        else:
            return np.argmax(self.Q[s, :])

    def greedy_episode(self):
        s, info = self.env.reset()
        s_hist = [s]
        a_hist = []
        r_hist = []
        while True:
            a = self.epsilon_greedy(s, self.epsilon)
            a_hist.append(a)

            s, r, done, truncated, info = self.env.step(a)
            s_hist.append(s)
            r_hist.append(r)

            if s == 15:
                break
            if done or truncated:
                break
        return s_hist, a_hist, r_hist

    def mc_control(self, num_episodes):
        for i in range(num_episodes):
            if i >= 1:
                self.epsilon = max(self.epsilon_start * 0.9999 ** (i - 1), 0)
            else:
                self.epsilon = self.epsilon_start

            if i >= 5000:
                lr = max(self.lr * 0.9999 ** (i - 5000), 0.0001)
            else:
                lr = self.lr

            s_hist, a_hist, r_hist = self.greedy_episode()

            G = 0
            for t in reversed(range(len(a_hist))):
                G = self.gamma * G + r_hist[t]
                if is_first_visit(s_hist, a_hist, t):
                    self.Q[s_hist[t], a_hist[t]] += lr * (G - self.Q[s_hist[t], a_hist[t]])
                    self.eval_return.append(np.max(self.Q[0, :]))

        Q_opt = self.Q.copy()
        opt_pi = np.argmax(Q_opt, axis=1)
        return Q_opt, opt_pi

    def sarsa(self, num_episodes):
        for i in range(num_episodes):
            if i >= 1:
                self.epsilon = max(self.epsilon_start * 0.9999 ** (i - 1), 0)
            else:
                self.epsilon = self.epsilon_start

            if i >= 5000:
                lr = max(self.lr * 0.9999 ** (i - 5000), 0.001)
            else:
                lr = self.lr

            s, info = self.env.reset()
            a = self.epsilon_greedy(s, self.epsilon)

            while True:
                s_plus, r, done, truncated, info = self.env.step(a)
                a_plus = self.epsilon_greedy(s_plus, self.epsilon)
                self.Q[s, a] += lr * (r + self.gamma * self.Q[s_plus, a_plus] - self.Q[s, a])
                s = s_plus
                a = a_plus
                self.eval_return.append(np.max(self.Q[0, :]))
                if s == 15 or done or truncated:
                    break

    def q_learning(self, num_episodes):
        for i in range(num_episodes):
            if i >= 1:
                self.epsilon = max(self.epsilon_start * 0.9999 ** (i - 1), 0)
            else:
                self.epsilon = self.epsilon_start

            if i >= 5000:
                lr = max(self.lr * 0.9999 ** (i - 5000), 0.001)
            else:
                lr = self.lr

            s, info = self.env.reset()
            while True:
                a = self.epsilon_greedy(s, self.epsilon)
                s_plus, r, done, truncated, info = self.env.step(a)
                self.Q[s, a] += lr * (r + self.gamma * np.max(self.Q[s_plus, :]) - self.Q[s, a])
                s = s_plus
                self.eval_return.append(np.max(self.Q[0, :]))
                if s == 15 or done or truncated:
                    break
    def plot_value_and_policy(self, **kwargs):
        grid_size = self.env.unwrapped.desc.shape
        desc = self.env.unwrapped.desc.astype(str)

        fig1, ax1 = plt.subplots(figsize=(5, 5))
        fig2, ax2 = plt.subplots(figsize=(5, 5))

        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                if desc[i, j] == 'S':
                    rect = plt.Rectangle((j, i), 1, 1, facecolor='blue', edgecolor='black', alpha=0.35)
                elif desc[i, j] == 'H':
                    rect = plt.Rectangle((j, i), 1, 1, facecolor='red', edgecolor='black', alpha=0.35)
                elif desc[i, j] == 'G':
                    rect = plt.Rectangle((j, i), 1, 1, facecolor='green', edgecolor='black', alpha=0.35)
                else:
                    rect = plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black')
                ax1.add_patch(rect)
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                if desc[i, j] == 'S':
                    rect = plt.Rectangle((j, i), 1, 1, facecolor='blue', edgecolor='black', alpha=0.35)
                elif desc[i, j] == 'H':
                    rect = plt.Rectangle((j, i), 1, 1, facecolor='red', edgecolor='black', alpha=0.35)
                elif desc[i, j] == 'G':
                    rect = plt.Rectangle((j, i), 1, 1, facecolor='green', edgecolor='black', alpha=0.35)
                else:
                    rect = plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black')
                ax2.add_patch(rect)

        arrow_dict = {0: (-0.3, 0), 1: (0, 0.3), 2: (0.3, 0), 3: (0, -0.3)}

        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                s = i * grid_size[1] + j
                v = round(np.max(self.Q[s, :]), 2)
                ax1.text(j + 0.5, i + 0.5, str(v), color='blue',
                         fontsize=12, ha='center', va='center')

                a = np.argmax(self.Q[s, :])
                dx, dy = arrow_dict[a]
                if desc[i, j] in ['H', 'G']:
                    continue
                ax2.arrow(j + 0.5, i + 0.5, dx, dy, head_width=0.1, head_length=0.1, fc='red', ec='red')

        ax1.set_xlim(0, grid_size[1])
        ax1.set_ylim(0, grid_size[0])
        ax1.set_aspect('equal')
        ax1.invert_yaxis()
        ax1.set_xticks(np.arange(grid_size[1] + 1))
        ax1.set_yticks(np.arange(grid_size[0] + 1))
        ax1.grid(False)
        ax1.set_title("V(s): {}, is_slippery={}".format(kwargs.get("algorithm", ""), kwargs.get("is_slippery", False)))

        ax2.set_xlim(0, grid_size[1])
        ax2.set_ylim(0, grid_size[0])
        ax2.set_aspect('equal')
        ax2.invert_yaxis()
        ax2.set_xticks(np.arange(grid_size[1] + 1))
        ax2.set_yticks(np.arange(grid_size[0] + 1))
        ax2.grid(False)
        ax2.set_title(
            "Trained Policy: {}, is_slippery={}".format(kwargs.get("algorithm", ""), kwargs.get("is_slippery", False)))
        plt.show()

def is_first_visit(s_hist, a_hist, t):
    for i in range(t):
        if s_hist[i] == s_hist[t] and a_hist[i] == a_hist[t]:
            return False
    return True