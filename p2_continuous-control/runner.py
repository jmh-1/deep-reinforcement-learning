from collections import deque
import numpy as np
from IPython.core.debugger import set_trace
import timeit
import matplotlib.pyplot as plt
import torch

def run(env, agent, n_episodes=2000, max_t=5000, brain_name="", breakWhenSolved=True, state_file_name="checkpoint.pth"):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    solved = False
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        state = env_info.vector_observations[0]            # get the current state        
        score = 0
        
        steps = 0
        start = timeit.default_timer()
        total_env_time = 0
        negCount = 0
        for t in range(max_t):
#             print(f'step {total_steps}')
            steps += 1
            action = agent.act(state)
            if np.any(action < 0):
                negCount += 1;
            env_start = timeit.default_timer()
            env_info = env.step(action)[brain_name]        # send the action to the environment
            total_env_time += timeit.default_timer() - env_start
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            print('\rstep {}\treward: {:.2f}\tnegCount {}'.format(steps, reward, negCount), end="")
#             print(action)
#             if reward>0:
#                 break
            done = env_info.local_done[0]                  # see if episode has finished
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 


        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}\tSteps: {:.2f}'.format(i_episode, np.mean(scores_window), score, steps), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
#             print('Time: ', timeit.default_timer() - start)
#             print('env Time: ', total_env_time)
        if np.mean(scores_window)>=30.0 and not solved:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), state_file_name)
            if breakWhenSolved:
                break
            solved = True
            
    return scores

def plot_scores(scores):
    n=100
    running_average = np.correlate(scores, np.ones(n)/n, mode='valid')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores, label='Score')
    plt.plot(np.arange(len(running_average)) + n - 1 , running_average, label=f'Average Score over prior {n} episodes')
    plt.xlabel('Episode #')
    ax.legend()
    plt.show()