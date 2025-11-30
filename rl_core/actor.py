import gymnasium as gym
import json
import time
import os
from kafka import KafkaProducer
from agent import PPOAgent
import numpy as np




class Actor:
    def __init__(self,actor_id,env_name,kafka_bootstrap):
        self.actor_id = actor_id
        self.env_name = env_name
        self.env = gym.make(env_name)
        
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        self.agent = PPOAgent(state_dim, action_dim)
        
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_bootstrap,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
        self.episode_count = 0
        self.total_steps = 0

    def sync_weights(self, weights_path):
        """Load latest weights from shared storage"""
        if os.path.exists(weights_path):
            self.agent.load(weights_path)

    def collect_episode(self):
        state, _ = self.env.reset()
        episode_reward = 0
        experiences = []
        
        while True:
            action, log_prob, value = self.agent.select_action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            experience = {
                'actor_id': self.actor_id,
                'state': state.tolist(),
                'action': action,
                'reward': reward,
                'next_state': next_state.tolist(),
                'done': done,
                'log_prob': log_prob,
                'value': value,
                'timestamp': time.time()
            }
            experiences.append(experience)
            
            # Send to Kafka
            self.producer.send('experiences.raw', experience)
            
            episode_reward += reward
            self.total_steps += 1
            state = next_state
            
            if done:
                break
        
        self.episode_count += 1    
        # Send episode metrics
        self.producer.send('metrics.actors', {
            'actor_id': self.actor_id,
            'episode': self.episode_count,
            'reward': episode_reward,
            'steps': len(experiences),
            'timestamp': time.time()
        })
        self.producer.flush()   
        return episode_reward, len(experiences)

    def run(self, weights_path, sync_interval=10):
        print(f"Actor {self.actor_id} starting on {self.env_name}")
        
        while True:
            # Sync weights periodically
            if self.episode_count % sync_interval == 0:
                self.sync_weights(weights_path)
            
            reward, steps = self.collect_episode()
            print(f"Actor {self.actor_id} | Episode {self.episode_count} | Reward: {reward:.2f} | Steps: {steps}")


if __name__ == "__main__":
    actor_id = int(os.environ.get('ACTOR_ID', 0))
    env_name = os.environ.get('ENV_NAME', 'CartPole-v1')
    kafka_bootstrap = os.environ.get('KAFKA_BOOTSTRAP', 'localhost:9092')
    weights_path = os.environ.get('WEIGHTS_PATH', '/shared/model.pt')
    
    actor = Actor(actor_id, env_name, kafka_bootstrap)
    actor.run(weights_path)
