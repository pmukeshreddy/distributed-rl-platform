from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from kafka import KafkaProducer, KafkaConsumer
from kubernetes import client, config
import json
import os
import asyncio
from collections import deque
from typing import List
import threading

app = FastAPI(title="RL Training Platform API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
KAFKA_BOOTSTRAP = os.environ.get('KAFKA_BOOTSTRAP', 'localhost:9092')

training_state = {
    'is_running': False,
    'env_name': 'CartPole-v1',
    'num_actors': 0,
    'config': {'lr': 3e-4, 'gamma': 0.99, 'clip_ratio': 0.2, 'batch_size': 2048}
}

metrics_buffer = deque(maxlen=500)
actor_metrics = deque(maxlen=100)
connected_websockets: List[WebSocket] = []


def get_kafka_producer():
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )


def get_k8s_client():
    try:
        config.load_incluster_config()
    except:
        try:
            config.load_kube_config()
        except:
            return None
    return client.AppsV1Api()


# Background metrics consumer
def consume_metrics():
    try:
        consumer = KafkaConsumer(
            'metrics.training', 'metrics.actors',
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_deserializer=lambda v: json.loads(v.decode('utf-8')),
            auto_offset_reset='latest',
            group_id='api-metrics'
        )
        
        for message in consumer:
            if message.topic == 'metrics.training':
                metrics_buffer.append(message.value)
            else:
                actor_metrics.append(message.value)
    except Exception as e:
        print(f"Kafka consumer error: {e}")


# Start consumer thread
threading.Thread(target=consume_metrics, daemon=True).start()


@app.get("/")
async def root():
    return {"status": "ok", "service": "RL Training Platform"}


@app.get("/api/status")
async def get_status():
    k8s = get_k8s_client()
    actor_replicas = 0
    learner_replicas = 0
    
    if k8s:
        try:
            actor_deploy = k8s.read_namespaced_deployment("rl-actor", "default")
            actor_replicas = actor_deploy.status.ready_replicas or 0
            
            learner_deploy = k8s.read_namespaced_deployment("rl-learner", "default")
            learner_replicas = learner_deploy.status.ready_replicas or 0
        except:
            pass
    
    return {
        "is_running": training_state['is_running'],
        "env_name": training_state['env_name'],
        "num_actors": actor_replicas,
        "learner_running": learner_replicas > 0,
        "config": training_state['config']
    }


@app.post("/api/training/start")
async def start_training(env_name: str = "CartPole-v1", num_actors: int = 4):
    training_state['is_running'] = True
    training_state['env_name'] = env_name
    training_state['num_actors'] = num_actors
    
    k8s = get_k8s_client()
    if k8s:
        try:
            # Update environment configmap
            core_v1 = client.CoreV1Api()
            core_v1.patch_namespaced_config_map(
                name="rl-config", namespace="default",
                body={"data": {"ENV_NAME": env_name}}
            )
            
            # Scale deployments
            k8s.patch_namespaced_deployment_scale(
                name="rl-actor", namespace="default",
                body={"spec": {"replicas": num_actors}}
            )
            k8s.patch_namespaced_deployment_scale(
                name="rl-learner", namespace="default",
                body={"spec": {"replicas": 1}}
            )
        except Exception as e:
            return {"status": "partial", "message": f"K8s error: {e}"}
    
    producer = get_kafka_producer()
    producer.send('commands.control', {
        'command': 'start',
        'env_name': env_name,
        'num_actors': num_actors
    })
    producer.flush()
    
    return {"status": "started", "env_name": env_name, "num_actors": num_actors}


@app.post("/api/training/stop")
async def stop_training():
    training_state['is_running'] = False
    
    k8s = get_k8s_client()
    if k8s:
        try:
            k8s.patch_namespaced_deployment_scale(
                name="rl-actor", namespace="default",
                body={"spec": {"replicas": 0}}
            )
        except:
            pass
    
    producer = get_kafka_producer()
    producer.send('commands.control', {'command': 'stop'})
    producer.flush()
    
    return {"status": "stopped"}


@app.get("/api/metrics")
async def get_metrics():
    metrics = list(metrics_buffer)
    
    if not metrics:
        return {"metrics": [], "summary": None}
    
    recent = metrics[-50:]
    rewards = [m.get('avg_reward', 0) for m in recent]
    
    return {
        "metrics": recent,
        "summary": {
            "updates": metrics[-1].get('update', 0) if metrics else 0,
            "total_experiences": metrics[-1].get('total_experiences', 0) if metrics else 0,
            "avg_reward": sum(rewards) / len(rewards) if rewards else 0,
            "max_reward": max(rewards) if rewards else 0,
            "policy_loss": metrics[-1].get('policy_loss', 0) if metrics else 0
        }
    }


@app.get("/api/metrics/history")
async def get_metrics_history(limit: int = 100):
    return {"history": list(metrics_buffer)[-limit:]}


@app.get("/api/actors")
async def get_actor_metrics():
    return {"actors": list(actor_metrics)}


@app.post("/api/config/hyperparam")
async def set_hyperparam(key: str, value: float):
    if key not in training_state['config']:
        return {"error": f"Unknown hyperparameter: {key}"}
    
    training_state['config'][key] = value
    
    producer = get_kafka_producer()
    producer.send('commands.control', {
        'command': 'set_hyperparam',
        'key': key,
        'value': value
    })
    producer.flush()
    
    return {"status": "updated", "key": key, "value": value}


@app.post("/api/scale")
async def scale_actors(count: int):
    if count < 0 or count > 16:
        return {"error": "Count must be 0-16"}
    
    training_state['num_actors'] = count
    
    k8s = get_k8s_client()
    if k8s:
        try:
            k8s.patch_namespaced_deployment_scale(
                name="rl-actor", namespace="default",
                body={"spec": {"replicas": count}}
            )
            return {"status": "scaled", "count": count}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    return {"status": "requested", "count": count, "note": "K8s not available"}


@app.get("/api/environments")
async def list_environments():
    return {
        "environments": [
            {"name": "CartPole-v1", "state_dim": 4, "action_dim": 2, "type": "discrete"},
            {"name": "LunarLander-v2", "state_dim": 8, "action_dim": 4, "type": "discrete"},
            {"name": "Acrobot-v1", "state_dim": 6, "action_dim": 3, "type": "discrete"},
            {"name": "MountainCar-v0", "state_dim": 2, "action_dim": 3, "type": "discrete"},
        ]
    }


@app.websocket("/ws/metrics")
async def websocket_metrics(websocket: WebSocket):
    await websocket.accept()
    connected_websockets.append(websocket)
    
    try:
        last_sent = 0
        while True:
            if len(metrics_buffer) > last_sent:
                new_metrics = list(metrics_buffer)[last_sent:]
                await websocket.send_json({"type": "metrics", "data": new_metrics})
                last_sent = len(metrics_buffer)
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        connected_websockets.remove(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
