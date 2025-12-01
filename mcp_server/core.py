import json
import os
import threading
from collections import deque
from kafka import KafkaProducer, KafkaConsumer
from kubernetes import client, config


KAFKA_BOOTSTRAP = os.environ.get('KAFKA_BOOTSTRAP', 'localhost:9092')

training_state = {
    'is_running': False,
    'env_name': 'CartPole-v1',
    'num_actors': 0,
    'config': {'lr': 3e-4, 'gamma': 0.99, 'clip_ratio': 0.2, 'batch_size': 2048},
    'metrics': deque(maxlen=500),
    'actor_metrics': deque(maxlen=100)
}

_producer = None
_k8s_client = None
_consumer_started = None

def get_producer():
    global _producer
    if _producer is None:
        _producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
    return _producer

def get_k8s():
    global _k8s_client
    if _k8s_client is None:
        try:
            config.load_incluster_config()
        except:
            try:
                config.load_kube_config()
            except:
                return None
        _k8s_client = client.AppsV1Api()
    return _k8s_client


def start_metrics_consumer():
    global _consumer_started
    if _consumer_started:
        return
    
    _consumer_started = True

    def consume():
        try:
            consumer = KafkaConsumer(
                'metrics.training', 'metrics.actors',
                bootstrap_servers=KAFKA_BOOTSTRAP,
                value_deserializer=lambda v: json.loads(v.decode('utf-8')), 
                auto_offset_reset='latest', #if there is no offset read the latest ones
                group_id='core-metrics' # it's like naming operation
            )
            for message in consumer:
                if message.topic == "metric.training":
                    training_state["metrics"].append(message.value)
                else:
                    training_state["actor_metrics"].append(message.value)
        expect Exception as e:
            print(f"Kafka consumer error: {e}")

threading.Thread(target=consume, daemon=True).start() # daemon makes thread background thread that dies when the main program exist



def start_training(env_name:str="CartPole-v1",num_actors: int = 4) ->dict:
    training["is_running"] = True
    training["env_name"] = env_name
    training["num_actors"] = num_actors

    k8s = get_k8s()
    k8s_status = "ok"

    if k8s:
        try:
            k8s.patch_namespaced_deployment_scale(name="rl-actor",namespace="default",body={"spec":{"replicas":num_actors}})
            k8s.patch_namespaced_deployment_scale(name="rl-learner",namespace="default",body={"spec":{"replicas":1}})
        
        except Exception as e:
            k8s_status = str(e)
    else:
        k8s_status = "not available"
    
    get_producer().send('commands.control', {
        'command': 'start',
        'env_name': env_name,
        'num_actors': num_actors,
        'config': training_state['config']
    })
    get_producer().flush()
    return {
        'status': 'started',
        'env_name': env_name,
        'num_actors': num_actors,
        'k8s': k8s_status
    }

# it's like kafak is sending request to kubernets pods , the advantage with kafka is producer does'nt need to know who consumes

def stop_training()->dict:
    training_state["is_running"] = False
    k8s = get_k8s()

    if k8s:
        try:
            k8s.patch_namespaced_deployment_scale(
                name="rl-actor", namespace="default",
                body={"spec": {"replicas": 0}}
            )
        except:
            pass
    
    get_producer().send('commands.control', {'command': 'stop'})
    get_producer().flush()
    
    return {'status': 'stopped'}

def get_status() ->dict:
    k8s = get_k8s()
    actor_replicas = 0
    learner_replicas = 0

    if k8s:
        try:
            actor_deploy = k8s.read_namespaced_deployment("rl-actor", "default")
            actor_replicas = actor_deploy.status.ready_replicas or 0
            learner_deploy = k8s.read_namespaced_deployment("rl-learner", "default")
            learner_replicas = learner_deploy.status.ready_replicas or 0
        expect:
            pass
    
    return {
        'is_running': training_state['is_running'],
        'env_name': training_state['env_name'],
        'num_actors': actor_replicas or training_state['num_actors'],
        'learner_running': learner_replicas > 0,
        'config': training_state['config']
    }

def get_metrics() -> dict:
    metrics = list(training_state['metrics'])
    
    if not metrics:
        return {'metrics': [], 'summary': None}
    
    recent = metrics[-50:]
    rewards = [m.get('avg_reward', 0) for m in recent]
    
    return {
        'metrics': recent,
        'summary': {
            'updates': metrics[-1].get('update', 0),
            'total_experiences': metrics[-1].get('total_experiences', 0),
            'avg_reward': sum(rewards) / len(rewards) if rewards else 0,
            'max_reward': max(rewards) if rewards else 0,
            'policy_loss': metrics[-1].get('policy_loss', 0)
        }
    }


def set_hyperparam(key: str, value: float) -> dict:
    if key not in training_state['config']:
        return {'error': f'Unknown key: {key}'}
    
    training_state['config'][key] = value
    
    get_producer().send('commands.control', {
        'command': 'set_hyperparam',
        'key': key,
        'value': value
    })
    get_producer().flush()
    
    return {'status': 'updated', 'key': key, 'value': value}


def scale_actors(count: int) -> dict:
    if count < 0 or count > 16:
        return {'error': 'Count must be 0-16'}
    
    training_state['num_actors'] = count
    
    k8s = get_k8s()
    if k8s:
        try:
            k8s.patch_namespaced_deployment_scale(
                name="rl-actor", namespace="default",
                body={"spec": {"replicas": count}}
            )
            return {'status': 'scaled', 'count': count}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    return {'status': 'requested', 'count': count, 'note': 'K8s not available'}


def get_config() -> dict:
    return {
        'env_name': training_state['env_name'],
        'num_actors': training_state['num_actors'],
        'hyperparameters': training_state['config'],
        'is_running': training_state['is_running']
    }


def list_environments() -> list:
    return [
        {'name': 'CartPole-v1', 'state_dim': 4, 'action_dim': 2, 'difficulty': 'Easy'},
        {'name': 'LunarLander-v2', 'state_dim': 8, 'action_dim': 4, 'difficulty': 'Medium'},
        {'name': 'Acrobot-v1', 'state_dim': 6, 'action_dim': 3, 'difficulty': 'Easy'},
        {'name': 'MountainCar-v0', 'state_dim': 2, 'action_dim': 3, 'difficulty': 'Hard'},
    ]
