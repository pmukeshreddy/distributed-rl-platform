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
    

class PPOAgent:
    def __init__(self,state_dim,action_dim,lr=3e-4,gamma=0.99,clip_ratio=0.2,epochs=10,batch_size=64):
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.epochs = epochs
        self.batch_size = batch_size

        self.model = ActorCritic(state_dim,action_dim)
        self.optimizer = optim.Adam(self.model.parameters(),lr=lr)

        self.config = {"lr":lr,"gamma":gamma,"clip_ratio":clip_ratio,"epochs":epochs,"batch_size":batch_size}

    def select_action(self,state):
        return self.model.act(state)
    
    def update(self,experiences):
        states = torch.FloatTensor(np.array([e['state'] for e in experiences]))
        actions = torch.LongTensor([e['action'] for e in experiences])
        old_log_probs = torch.FloatTensor([e['log_prob'] for e in experiences])
        rewards = [e['reward'] for e in experiences]
        dones = [e['done'] for e in experiences]

        #compute returns
        returns = []
        R = 0
        for r,d in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1-d)
            returns.insert(0, R)
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        metrics = {"policy_loss":0,"value_loss":0,"entropy":0}
        for _ in range(self.epochs):
            log_probs, values, entropy = self.model.evaluate(states, actions)
            
            advantages = returns - values.detach()
            ratio = torch.exp(log_probs - old_log_probs)
            
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * (returns - values).pow(2).mean()
            entropy_loss = -0.01 * entropy.mean()
            
            loss = policy_loss + value_loss + entropy_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            metrics['policy_loss'] += policy_loss.item()
            metrics['value_loss'] += value_loss.item()
            metrics['entropy'] += entropy.mean().item()
        
        for k in metrics:
            metrics[k] /= self.epochs
        
        return metrics
    
    def save(self, path):
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
    
    def set_hyperparam(self, key, value):
        if key == 'lr':
            for pg in self.optimizer.param_groups:
                pg['lr'] = value
        elif key in self.config:
            setattr(self, key, value)
        self.config[key] = value
