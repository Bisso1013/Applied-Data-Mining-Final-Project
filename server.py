import os
import json
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from main import app as langgraph_app

os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

server = FastAPI()
SESSIONS_FILE = Path("chat_sessions.json")


def load_sessions() -> dict:
    if SESSIONS_FILE.exists():
        try:
            return json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_sessions(sessions: dict):
    SESSIONS_FILE.write_text(json.dumps(sessions, indent=2), encoding="utf-8")


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "web_session"


@server.get("/", response_class=HTMLResponse)
async def root():
    with open("index.html", encoding="utf-8") as f:
        return f.read()


@server.get("/chats")
async def list_chats():
    sessions = load_sessions()
    return sorted(sessions.values(), key=lambda x: x.get("updated_at", ""), reverse=True)


@server.delete("/chats/{thread_id}")
async def delete_chat(thread_id: str):
    sessions = load_sessions()
    sessions.pop(thread_id, None)
    save_sessions(sessions)
    return {"ok": True}


@server.post("/chat")
async def chat(request: ChatRequest):
    config = {"configurable": {"thread_id": request.thread_id}}
    inputs = {"messages": [HumanMessage(content=request.message)]}

    final_reply = "I'm sorry, I encountered an error processing that request."
    active_node = "system"

    try:
        for event in langgraph_app.stream(inputs, config):
            for node_name, state_update in event.items():
                active_node = node_name
                if isinstance(state_update.get("messages"), list) and state_update["messages"]:
                    latest = state_update["messages"][-1]
                    if hasattr(latest, "content"):
                        final_reply = latest.content
                    elif isinstance(latest, dict) and "content" in latest:
                        final_reply = latest["content"]
                elif isinstance(state_update.get("messages"), str):
                    final_reply = state_update["messages"]
                elif "response" in state_update:
                    final_reply = str(state_update["response"])
    except Exception as e:
        final_reply = f"An error occurred: {str(e)}"
        active_node = "error"

    # Persist session to disk
    sessions = load_sessions()
    now = datetime.now().isoformat()
    if request.thread_id not in sessions:
        sessions[request.thread_id] = {
            "thread_id": request.thread_id,
            "title": request.message[:60],
            "messages": [],
            "created_at": now,
        }
    sessions[request.thread_id]["messages"].append(
        {"role": "user", "content": request.message}
    )
    sessions[request.thread_id]["messages"].append(
        {"role": "assistant", "content": final_reply, "node": active_node}
    )
    sessions[request.thread_id]["updated_at"] = now
    save_sessions(sessions)

    return {"reply": final_reply, "node": active_node}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(server, host="0.0.0.0", port=8001)
