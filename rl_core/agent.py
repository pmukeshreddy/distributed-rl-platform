import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np 

class ActorCritic(nn.Module):
    def __init__(self,state_dim,action_dim,hidden_dim=64):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(state_dim,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,hidden_dim),nn.ReLU())

        self.actor = nn.Linear(hidden_dim,action_dim)
        self.critic = nn.Linear(hidden_dim,1)

    def forward(self,x):
        x = self.shared(x)
        return self.actor(x) , self.critic(x)
    
    def act(self,state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            logits , value = self.forward(state)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.item() , log_prob.item() , value.item()
    
    def evaluate(self,states,actions):
        logits , values = self.forward(states)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob , values.squeeze() , entropy
    

