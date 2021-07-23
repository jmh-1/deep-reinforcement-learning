import numpy as np
import random
from collections import namedtuple, deque
import model
# from model import QNetwork

from importlib import reload
reload( model)
QNetwork = model.QNetwork
import torch
import torch.nn.functional as F
import torch.optim as optim
from IPython.core.debugger import set_trace
import timeit
import dqn_agent


BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent(dqn_agent.Agent):

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        ## TODO: compute and minimize the loss
        
        start = timeit.default_timer()
        self.optimizer.zero_grad()
        preds = self.qnetwork_local(states).gather(1,actions)
        next_action_indices = self.qnetwork_local(next_states).detach().max(1)[1]
        targets = rewards + gamma*self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)*(1-dones)
        loss = F.mse_loss(targets, preds)
        loss.backward()
        self.optimizer.step()
        #print('learn time: ', timeit.default_timer() - start)
        "*** YOUR CODE HERE ***"
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)                     

