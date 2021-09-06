from collections import deque
import numpy as np
from IPython.core.debugger import set_trace
import timeit
import matplotlib.pyplot as plt
import torch

def run(env, agent, n_episodes=3000, brain_name=""):
    scores_deque = deque(maxlen=100)  
    scores = []                        
    for i_episode in range(1, n_episodes+1):
        
        env_info = env.reset(train_mode=True)[brain_name] 
        state = env_info.vector_observations[0]           
        agent.reset()
        score = 0
        while True:
            action = agent.act(state)
            env_info = env.step(action)[brain_name]        
            next_state = env_info.vector_observations[0]   

            reward = env_info.rewards
            done = env_info.local_done

            agent.step(state, action, reward, next_state, done)

            score += reward[0]
            state = next_state
            if np.any(done):
                break 

        scores_deque.append(score)    
        scores.append(score)          
        avg_score = np.mean(scores_deque)

        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, avg_score, score), end="")
        if i_episode % 100 == 0:
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
        if avg_score>=30:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_deque)))
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            break
    return scores
