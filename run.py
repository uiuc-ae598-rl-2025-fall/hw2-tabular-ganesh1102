import pickle
import gymnasium as gym
from agent import Agent
from matplotlib import pyplot as plt

if __name__ == "__main__":
    is_slippery = True
    env = gym.make('FrozenLake-v1', desc=["SFFF", "FHFH", "FFFH", "HFFG"], map_name="4x4", is_slippery=is_slippery)
    gamma = 0.95
    if is_slippery:
        n_episodes = 150000
    else:
        n_episodes = 100000
    epsilon = 1.0
    learning_rate = 0.1
    mc_slip = Agent(env, gamma, learning_rate, epsilon)
    mc_slip.mc_control(n_episodes)
    # mc_slip.plot_value_and_policy(is_slippery=is_slippery, algorithm="MC Control")
    slip_mc_return = mc_slip.eval_return

    is_slippery = False
    env = gym.make('FrozenLake-v1', desc=["SFFF", "FHFH", "FFFH", "HFFG"], map_name="4x4", is_slippery=is_slippery)
    gamma = 0.95
    if is_slippery:
        n_episodes = 150000
    else:
        n_episodes = 100000
    epsilon = 1.0
    learning_rate = 0.1
    mc_det = Agent(env, gamma, learning_rate, epsilon)
    mc_det.mc_control(n_episodes)
    # mc_det.plot_value_and_policy(is_slippery=is_slippery, algorithm="MC Control")
    det_mc_return = mc_det.eval_return


    is_slippery = True
    env = gym.make('FrozenLake-v1', desc=["SFFF", "FHFH", "FFFH", "HFFG"], map_name="4x4", is_slippery=is_slippery)
    gamma = 0.95
    n_episodes = 50000
    epsilon = 1.0
    learning_rate = 0.1
    q_slip = Agent(env, gamma, learning_rate, epsilon)
    slip_q_return = q_slip.q_learning(n_episodes)
    # q_slip.plot_value_and_policy(is_slippery=is_slippery, algorithm="Q-Learning")
    slip_q_eval_return = q_slip.eval_return

    is_slippery = False
    env = gym.make('FrozenLake-v1', desc=["SFFF", "FHFH", "FFFH", "HFFG"], map_name="4x4", is_slippery=is_slippery)
    gamma = 0.95
    n_episodes = 50000
    epsilon = 1.0
    learning_rate = 0.1
    q_det = Agent(env, gamma, learning_rate, epsilon)
    det_q_return = q_det.q_learning(n_episodes)
    det_q_eval_return = q_det.eval_return
    # q_det.plot_value_and_policy(is_slippery=is_slippery, algorithm="Q-Learning")
    
    is_slippery = False
    env = gym.make('FrozenLake-v1', desc=["SFFF", "FHFH", "FFFH", "HFFG"], map_name="4x4", is_slippery=is_slippery)
    gamma = 0.95
    n_episodes = 50000
    epsilon = 1.0
    learning_rate = 0.1
    sar_det = Agent(env, gamma, learning_rate, epsilon)
    sar_det.sarsa(n_episodes)
    sar_det.plot_value_and_policy(is_slippery=is_slippery, algorithm="SARSA")
    det_sar_eval_return = sar_det.eval_return

    is_slippery = True
    env = gym.make('FrozenLake-v1', desc=["SFFF", "FHFH", "FFFH", "HFFG"], map_name="4x4", is_slippery=is_slippery)
    gamma = 0.95
    n_episodes = 50000
    epsilon = 1.0
    learning_rate = 0.1
    sar_slip = Agent(env, gamma, learning_rate, epsilon)
    sar_slip.sarsa(n_episodes)
    sar_slip.plot_value_and_policy(is_slippery=is_slippery, algorithm="SARSA")
    slip_sar_eval_return = sar_slip.eval_return

    # compare the eval returns

    plt.figure()
    plt.plot(det_mc_return, label="MC Control")
    plt.plot(det_sar_eval_return, label="SARSA")
    plt.plot(det_q_eval_return, label="Q-Learning")
    plt.hlines(0.776, 0,
               max(len(det_mc_return), len(det_sar_eval_return), len(det_q_eval_return)),
               colors='k', linestyles='dashed', label="Optimal Return by Value Iteration")
    plt.grid()
    plt.xlabel("Cumulative Time Steps")
    plt.ylabel("Evaluation Return")
    plt.legend()
    plt.title("is_slippery = False")
    plt.show()


    plt.figure()
    plt.plot(slip_mc_return, label="MC Control")
    plt.plot(slip_sar_eval_return, label="SARSA")
    plt.plot(slip_q_eval_return, label="Q-Learning")
    plt.hlines(0.183, 0,
               max(len(slip_mc_return), len(slip_sar_eval_return), len(slip_q_eval_return)),
               colors='k', linestyles='dashed', label="Optimal Return by Value Iteration")
    plt.grid()
    plt.xlabel("Cumulative Time Steps")
    plt.ylabel("Evaluation Return")
    plt.legend()
    plt.title("is_slippery = True")
    plt.show()