import json
import time
import os
import threading
from collections import deque
from kafka import KafkaConsumer, KafkaProducer
from agent import PPOAgent
import gymnasium as gym

class Learner:
    def __init__(self, env_name, kafka_bootstrap, batch_size=2048):
        self.env_name = env_name
        self.kafka_bootstrap = kafka_bootstrap
        self.batch_size = batch_size
        
        # Initialize agent
        env = gym.make(env_name)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        self.agent = PPOAgent(state_dim, action_dim)
        env.close()
        
        # Kafka consumer for experiences
        self.consumer = KafkaConsumer(
            'experiences.raw',
            bootstrap_servers=kafka_bootstrap,
            value_deserializer=lambda v: json.loads(v.decode('utf-8')),
            auto_offset_reset='latest',
            group_id='learner-group'
        )
        
        # Kafka producer for metrics
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_bootstrap,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
        self.experience_buffer = deque(maxlen=batch_size * 2)
        self.update_count = 0
        self.metrics_history = []
        
        # Training state
        self.is_training = True
        self.total_experiences = 0
        
        # Auto-stop settings (configurable via env vars)
        self.max_experiences = int(os.environ.get('MAX_EXPERIENCES', 100000))
        self.solve_reward = float(os.environ.get('SOLVE_REWARD', 195.0))
    
    def consume_experiences(self):
        """Background thread to consume experiences"""
        for message in self.consumer:
            if not self.is_training:
                break
            self.experience_buffer.append(message.value)
            self.total_experiences += 1
    
    def consume_commands(self):
        """Listen for control commands"""
        cmd_consumer = KafkaConsumer(
            'commands.control',
            bootstrap_servers=self.kafka_bootstrap,
            value_deserializer=lambda v: json.loads(v.decode('utf-8')),
            auto_offset_reset='latest',
            group_id=f'learner-commands-{int(time.time())}'
        )
        for message in cmd_consumer:
            cmd = message.value.get('command')
            if cmd == 'stop':
                print("Received stop command")
                self.is_training = False
                break
            elif cmd == 'set_hyperparam':
                key = message.value.get('key')
                value = message.value.get('value')
                self.set_hyperparam(key, value)
                print(f"Updated {key} = {value}")
    
    def train_step(self):
        if len(self.experience_buffer) < self.batch_size:
            return None
        
        # Sample batch
        batch = list(self.experience_buffer)[:self.batch_size]
        self.experience_buffer.clear()
        
        # Update agent
        metrics = self.agent.update(batch)
        self.update_count += 1
        
        # Compute average episode reward from batch
        rewards = [e['reward'] for e in batch]
        avg_reward = sum(rewards) / len(rewards)
        
        metrics.update({
            'update': self.update_count,
            'batch_size': len(batch),
            'avg_reward': avg_reward,
            'total_experiences': self.total_experiences,
            'timestamp': time.time()
        })
        
        # Send metrics to Kafka
        self.producer.send('metrics.training', metrics)
        self.producer.flush()
        
        self.metrics_history.append(metrics)
        return metrics
    
    def check_auto_stop(self):
        """Check if training should auto-stop"""
        # Stop at experience limit
        if self.total_experiences >= self.max_experiences:
            print(f"Reached {self.max_experiences} experiences. Auto-stopping.")
            return True
        
        # Stop when solved (check last 10 updates)
        if len(self.metrics_history) >= 10:
            recent_rewards = [m['avg_reward'] for m in self.metrics_history[-10:]]
            avg = sum(recent_rewards) / len(recent_rewards)
            if avg >= self.solve_reward:
                print(f"Solved! Avg reward {avg:.1f} >= {self.solve_reward}. Auto-stopping.")
                return True
        
        return False
    
    def save_weights(self, path):
        self.agent.save(path)
    
    def get_metrics(self):
        return {
            'update_count': self.update_count,
            'total_experiences': self.total_experiences,
            'buffer_size': len(self.experience_buffer),
            'recent_metrics': self.metrics_history[-10:] if self.metrics_history else [],
            'config': self.agent.config
        }
    
    def set_hyperparam(self, key, value):
        self.agent.set_hyperparam(key, value)
        return {'status': 'ok', 'key': key, 'value': value}
    
    def run(self, weights_path, save_interval=10):
        print(f"Learner starting for {self.env_name}")
        print(f"Auto-stop: {self.max_experiences} experiences or reward >= {self.solve_reward}")
        
        # Start consumer thread
        consumer_thread = threading.Thread(target=self.consume_experiences, daemon=True)
        consumer_thread.start()
        
        # Start command listener thread
        cmd_thread = threading.Thread(target=self.consume_commands, daemon=True)
        cmd_thread.start()
        
        while self.is_training:
            metrics = self.train_step()
            
            if metrics:
                print(f"Update {self.update_count} | Loss: {metrics['policy_loss']:.4f} | Avg Reward: {metrics['avg_reward']:.2f}")
                
                # Save periodically
                if self.update_count % save_interval == 0:
                    self.save_weights(weights_path)
                    print(f"Saved weights to {weights_path}")
            
            # Check auto-stop always, not just after training step
            if self.check_auto_stop():
                self.is_training = False
            
            time.sleep(0.1)
        
        # Save final weights on stop
        self.save_weights(weights_path)
        print("Training stopped. Final weights saved.")
        
        # Send stop signal to actors
        self.producer.send('commands.control', {'command': 'stop'})
        self.producer.flush()


if __name__ == "__main__":
    env_name = os.environ.get('ENV_NAME', 'CartPole-v1')
    kafka_bootstrap = os.environ.get('KAFKA_BOOTSTRAP', 'localhost:9093')
    weights_path = os.environ.get('WEIGHTS_PATH', '/shared/model.pt')
    batch_size = int(os.environ.get('BATCH_SIZE', 2048))
    
    learner = Learner(env_name, kafka_bootstrap, batch_size)
    learner.run(weights_path)
