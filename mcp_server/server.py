import json
import asyncio
from mcp.server import Server
from mcp.types import Tool, TextContent
import core

app = Server("rl-training-server")

# Start background consumer
core.start_metrics_consumer()


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="start_training",
            description="Start distributed RL training with specified environment and number of actors",
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
            description="Stop all training processes",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="get_metrics",
            description="Get current training metrics",
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
                "properties": {"count": {"type": "integer", "minimum": 1, "maximum": 16}},
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
