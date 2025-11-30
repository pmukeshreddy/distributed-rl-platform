import json
import os
import asyncio
from typing import Any
from mcp.server import Server
from mcp.types import Tool, TextContent
from kafka import KafkaProducer, KafkaConsumer
from kubernetes import client, config
import redis
import threading
from collections import deque

# Initialize MCP server
app = Server("rl-training-server")

# Global state
training_state = {
    'is_running': False,
    'env_name': 'CartPole-v1',
    'num_actors': 1,
    'config': {
        'lr': 3e-4,
        'gamma': 0.99,
        'clip_ratio': 0.2,
        'batch_size': 2048
    },
    'metrics': deque(maxlen=100)
}

# Kafka setup
KAFKA_BOOTSTRAP = os.environ.get('KAFKA_BOOTSTRAP', 'localhost:9092')

def get_kafka_producer():
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

def get_k8s_client():
    try:
        config.load_incluster_config()
    except:
        config.load_kube_config()
    return client.AppsV1Api()


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="start_training",
            description="Start distributed RL training with specified environment and number of actors",
            inputSchema={
                "type": "object",
                "properties": {
                    "env_name": {"type": "string", "description": "Gym environment name (e.g., CartPole-v1)"},
                    "num_actors": {"type": "integer", "description": "Number of actor pods to spawn", "default": 4}
                },
                "required": ["env_name"]
            }
        ),
        Tool(
            name="stop_training",
            description="Stop all training processes",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="get_metrics",
            description="Get current training metrics including reward, loss, and throughput",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="set_hyperparam",
            description="Update a hyperparameter during training",
            inputSchema={
                "type": "object",
                "properties": {
                    "key": {"type": "string", "enum": ["lr", "gamma", "clip_ratio", "batch_size"]},
                    "value": {"type": "number"}
                },
                "required": ["key", "value"]
            }
        ),
        Tool(
            name="scale_actors",
            description="Scale the number of actor pods",
            inputSchema={
                "type": "object",
                "properties": {
                    "count": {"type": "integer", "minimum": 1, "maximum": 16}
                },
                "required": ["count"]
            }
        ),
        Tool(
            name="get_config",
            description="Get current training configuration",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="list_environments",
            description="List available Gym environments",
            inputSchema={"type": "object", "properties": {}}
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    
    if name == "start_training":
        env_name = arguments.get('env_name', 'CartPole-v1')
        num_actors = arguments.get('num_actors', 4)
        
        training_state['is_running'] = True
        training_state['env_name'] = env_name
        training_state['num_actors'] = num_actors
        
        # Scale K8s deployments
        try:
            k8s = get_k8s_client()
            
            # Scale actors
            k8s.patch_namespaced_deployment_scale(
                name="rl-actor",
                namespace="default",
                body={"spec": {"replicas": num_actors}}
            )
            
            # Ensure learner is running
            k8s.patch_namespaced_deployment_scale(
                name="rl-learner",
                namespace="default",
                body={"spec": {"replicas": 1}}
            )
            
            result = f"Started training on {env_name} with {num_actors} actors"
        except Exception as e:
            result = f"Started training (K8s scaling failed: {e}). Running locally."
        
        # Send start command via Kafka
        producer = get_kafka_producer()
        producer.send('commands.control', {
            'command': 'start',
            'env_name': env_name,
            'num_actors': num_actors,
            'config': training_state['config']
        })
        producer.flush()
        
        return [TextContent(type="text", text=result)]
    
    elif name == "stop_training":
        training_state['is_running'] = False
        
        try:
            k8s = get_k8s_client()
            k8s.patch_namespaced_deployment_scale(
                name="rl-actor", namespace="default",
                body={"spec": {"replicas": 0}}
            )
        except:
            pass
        
        producer = get_kafka_producer()
        producer.send('commands.control', {'command': 'stop'})
        producer.flush()
        
        return [TextContent(type="text", text="Training stopped")]
    
    elif name == "get_metrics":
        metrics = list(training_state['metrics'])
        
        if not metrics:
            return [TextContent(type="text", text="No metrics available yet. Training may not have started.")]
        
        latest = metrics[-1] if metrics else {}
        avg_reward = sum(m.get('avg_reward', 0) for m in metrics[-10:]) / min(len(metrics), 10)
        
        summary = {
            'is_running': training_state['is_running'],
            'env_name': training_state['env_name'],
            'num_actors': training_state['num_actors'],
            'updates': latest.get('update', 0),
            'total_experiences': latest.get('total_experiences', 0),
            'avg_reward_last_10': round(avg_reward, 2),
            'latest_policy_loss': round(latest.get('policy_loss', 0), 4),
            'latest_value_loss': round(latest.get('value_loss', 0), 4)
        }
        
        return [TextContent(type="text", text=json.dumps(summary, indent=2))]
    
    elif name == "set_hyperparam":
        key = arguments['key']
        value = arguments['value']
        
        training_state['config'][key] = value
        
        producer = get_kafka_producer()
        producer.send('commands.control', {
            'command': 'set_hyperparam',
            'key': key,
            'value': value
        })
        producer.flush()
        
        return [TextContent(type="text", text=f"Set {key} = {value}")]
    
    elif name == "scale_actors":
        count = arguments['count']
        training_state['num_actors'] = count
        
        try:
            k8s = get_k8s_client()
            k8s.patch_namespaced_deployment_scale(
                name="rl-actor", namespace="default",
                body={"spec": {"replicas": count}}
            )
            result = f"Scaled to {count} actors"
        except Exception as e:
            result = f"Scale request sent (K8s: {e})"
        
        return [TextContent(type="text", text=result)]
    
    elif name == "get_config":
        config_info = {
            'env_name': training_state['env_name'],
            'num_actors': training_state['num_actors'],
            'hyperparameters': training_state['config'],
            'is_running': training_state['is_running']
        }
        return [TextContent(type="text", text=json.dumps(config_info, indent=2))]
    
    elif name == "list_environments":
        envs = [
            {"name": "CartPole-v1", "type": "Classic Control", "difficulty": "Easy"},
            {"name": "LunarLander-v2", "type": "Box2D", "difficulty": "Medium"},
            {"name": "Acrobot-v1", "type": "Classic Control", "difficulty": "Easy"},
            {"name": "MountainCar-v0", "type": "Classic Control", "difficulty": "Hard (sparse reward)"},
            {"name": "Pendulum-v1", "type": "Classic Control", "difficulty": "Medium (continuous)"},
        ]
        return [TextContent(type="text", text=json.dumps(envs, indent=2))]
    
    return [TextContent(type="text", text=f"Unknown tool: {name}")]


def metrics_consumer_thread():
    """Background thread to consume training metrics"""
    consumer = KafkaConsumer(
        'metrics.training',
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_deserializer=lambda v: json.loads(v.decode('utf-8')),
        auto_offset_reset='latest',
        group_id='mcp-metrics'
    )
    
    for message in consumer:
        training_state['metrics'].append(message.value)


async def main():
    # Start metrics consumer
    thread = threading.Thread(target=metrics_consumer_thread, daemon=True)
    thread.start()
    
    # Run MCP server
    from mcp.server.stdio import stdio_server
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
