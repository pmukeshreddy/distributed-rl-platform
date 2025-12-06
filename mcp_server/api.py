from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from typing import List
import core

app = FastAPI(title="RL Training Platform API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

connected_websockets: List[WebSocket] = []

# Start background consumer
core.start_metrics_consumer()


@app.get("/")
async def root():
    return {"status": "ok", "service": "RL Training Platform"}


@app.get("/api/status")
async def get_status():
    return core.get_status()


@app.post("/api/training/start")
async def start_training(env_name: str = "CartPole-v1", num_actors: int = 4):
    return core.start_training(env_name, num_actors)


@app.post("/api/training/stop")
async def stop_training():
    return core.stop_training()


@app.get("/api/metrics")
async def get_metrics():
    return core.get_metrics()


@app.get("/api/metrics/history")
async def get_metrics_history(limit: int = 100):
    return {"history": list(core.training_state['metrics'])[-limit:]}


@app.get("/api/actors")
async def get_actor_metrics():
    return {"actors": list(core.training_state['actor_metrics'])}


@app.post("/api/config/hyperparam")
async def set_hyperparam(key: str, value: float):
    return core.set_hyperparam(key, value)


@app.post("/api/scale")
async def scale_actors(count: int):
    return core.scale_actors(count)


@app.get("/api/config")
async def get_config():
    return core.get_config()


@app.get("/api/environments")
async def list_environments():
    return {"environments": core.list_environments()}


@app.websocket("/ws/metrics")
async def websocket_metrics(websocket: WebSocket):
    await websocket.accept()
    connected_websockets.append(websocket)
    
    try:
        last_sent = 0
        metrics = core.training_state['metrics']
        while True:
            if len(metrics) > last_sent:
                new_metrics = list(metrics)[last_sent:]
                await websocket.send_json({"type": "metrics", "data": new_metrics})
                last_sent = len(metrics)
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        connected_websockets.remove(websocket)


from fastapi.responses import FileResponse
from pathlib import Path

@app.get("/ui")
async def serve_ui():
    return FileResponse(Path(__file__).parent.parent / "frontend" / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
