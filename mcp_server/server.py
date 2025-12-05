import json
import asyncio
from mcp.server import Server
from mcp.types import Tool, TextContent
import core

app = Server("rl-training-server")

core.start_metrics_consumer()


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="start_training",
            description="Initialize and launch a distributed reinforcement learning training session using the PPO algorithm. "
                        "This tool spawns multiple actor processes that collect experience trajectories in parallel from the specified OpenAI Gym environment. "
                        "The learner process aggregates experiences, computes policy gradients, and updates the shared neural network weights. "
                        "Supports both discrete and continuous action spaces. "
                        "Training progress and metrics are streamed via Kafka for real-time monitoring.",
            inputSchema={
                "type": "object",
                "properties": {
                    "env_name": {"type": "string", "description": "Gym environment name"},
                    "num_actors": {"type": "integer", "default": 4}
                },
                "required": ["env_name"]
            }
        ),
        Tool(
            name="stop_training",
            description="Gracefully terminate all active training processes including actor pods, the learner process, and any associated Kubernetes resources. "
                        "This performs a clean shutdown by allowing in-flight experience batches to complete, saving the current model checkpoint to persistent storage, "
                        "flushing all pending metrics to Kafka, and deallocating GPU/CPU resources. "
                        "Use this before scaling down infrastructure or switching environments.",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="get_metrics",
            description="Retrieve comprehensive training metrics aggregated from all actor processes. "
                        "Returns episode rewards (mean, min, max, std), policy loss, value function loss, entropy bonus, KL divergence from previous policy, "
                        "explained variance, frames per second throughput, total environment steps completed, and wall-clock training time. "
                        "Metrics are computed over a rolling window of recent episodes for stability.",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="set_hyperparam",
            description="Dynamically modify a hyperparameter during active training without restarting the session. "
                        "Changes propagate to all actor and learner processes on the next optimization step. "
                        "Useful for implementing learning rate schedules, adjusting exploration-exploitation tradeoffs via entropy coefficient, "
                        "or tuning PPO's clipping ratio based on observed KL divergence. "
                        "Changes are logged and can be correlated with metric shifts.",
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
            description="Horizontally scale the number of actor pods in the Kubernetes cluster to adjust training throughput. "
                        "More actors increase environment sample collection rate, enabling faster policy updates but consuming more compute resources. "
                        "Scaling is performed via Kubernetes HPA with zero-downtime rolling updates. "
                        "New actors automatically sync with the latest policy weights and begin contributing experiences immediately.",
            inputSchema={
                "type": "object",
                "properties": {"count": {"type": "integer", "minimum": 1, "maximum": 16}},
                "required": ["count"]
            }
        ),
        Tool(
            name="get_config",
            description="Fetch the complete current training configuration including all hyperparameters "
                        "(learning rate, gamma, lambda for GAE, clip ratio, entropy coefficient, value loss coefficient), "
                        "network architecture details (hidden layer sizes, activation functions), environment settings, "
                        "batch and minibatch sizes, number of PPO epochs per update, gradient clipping thresholds, "
                        "and Kubernetes resource allocations for actors and learner.",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="list_environments",
            description="Enumerate all available OpenAI Gym environments that are compatible with this distributed training framework. "
                        "Returns environment IDs grouped by category (Classic Control, Box2D, MuJoCo, Atari), "
                        "along with observation space dimensions, action space type and size, reward range, and maximum episode length. "
                        "Helps in selecting appropriate environments for benchmarking or experimentation.",
            inputSchema={"type": "object", "properties": {}}
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    
    if name == "start_training":
        result = core.start_training(
            arguments.get('env_name', 'CartPole-v1'),
            arguments.get('num_actors', 4)
        )
    
    elif name == "stop_training":
        result = core.stop_training()
    
    elif name == "get_metrics":
        data = core.get_metrics()
        if not data['summary']:
            return [TextContent(type="text", text="No metrics yet. Training may not have started.")]
        result = data['summary']
    
    elif name == "set_hyperparam":
        result = core.set_hyperparam(arguments['key'], arguments['value'])
    
    elif name == "scale_actors":
        result = core.scale_actors(arguments['count'])
    
    elif name == "get_config":
        result = core.get_config()
    
    elif name == "list_environments":
        result = core.list_environments()
    
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]
    
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def main():
    from mcp.server.stdio import stdio_server
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
