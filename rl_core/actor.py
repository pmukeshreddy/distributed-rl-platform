import gymnasium as gym
import json
import time
import os
import threading
from kafka import KafkaProducer, KafkaConsumer
from agent import PPOAgent
import numpy as np


class Actor:
    def __init__(self, actor_id, env_name, kafka_bootstrap):
        self.actor_id = actor_id
        self.env_name = env_name
        self.kafka_bootstrap = kafka_bootstrap
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
        self.is_running = True

    def consume_commands(self):
        """Listen for control commands"""
        cmd_consumer = KafkaConsumer(
            'commands.control',
            bootstrap_servers=self.kafka_bootstrap,
            value_deserializer=lambda v: json.loads(v.decode('utf-8')),
            auto_offset_reset='latest',
            group_id=f'actor-{self.actor_id}-commands'
        )
        # Wait for partition assignment, then skip to end
        cmd_consumer.poll(timeout_ms=2000)
        for tp in cmd_consumer.assignment():
            cmd_consumer.seek_to_end(tp)
        
        for message in cmd_consumer:
            cmd = message.value.get('command')
            if cmd == 'stop':
                print(f"Actor {self.actor_id} received stop command")
                self.is_running = False
                break

    def sync_weights(self, weights_path):
        """Load latest weights from shared storage"""
        if os.path.exists(weights_path):
            try:
                self.agent.load(weights_path)
            except Exception as e:
                print(f"Actor {self.actor_id} failed to load weights: {e}")

    def collect_episode(self):
        state, _ = self.env.reset()
        episode_reward = 0
        experiences = []
        
        while self.is_running:
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
        
        # Start command listener thread
        cmd_thread = threading.Thread(target=self.consume_commands, daemon=True)
        cmd_thread.start()
        
        while self.is_running:
            # Sync weights periodically
            if self.episode_count % sync_interval == 0:
                self.sync_weights(weights_path)
            
            reward, steps = self.collect_episode()
            print(f"Actor {self.actor_id} | Episode {self.episode_count} | Reward: {reward:.2f} | Steps: {steps}")
        
        print(f"Actor {self.actor_id} stopped.")


if __name__ == "__main__":
    actor_id = os.environ.get('ACTOR_ID', '0')
    env_name = os.environ.get('ENV_NAME', 'CartPole-v1')
    kafka_bootstrap = os.environ.get('KAFKA_BOOTSTRAP', 'localhost:9093')
    weights_path = os.environ.get('WEIGHTS_PATH', '/shared/model.pt')
    
    actor = Actor(actor_id, env_name, kafka_bootstrap)
    actor.run(weights_path)
