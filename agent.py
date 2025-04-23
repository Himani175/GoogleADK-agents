import os
import re
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types

# Load environment variables
load_dotenv()

# FastAPI app instance
app = FastAPI()

# MongoDB setup
try:
    mongo_uri = os.getenv("MONGO_URI", "mongodb+srv://vishal:Kv8kszGpTDCX2eZ9@cluster0.011x4.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
    client = MongoClient(mongo_uri)
    db = client["Atithi"]
    namespace_collection = db["Namespace"]
    context_collection = db["Context"]
except Exception as e:
    raise RuntimeError(f"Error connecting to MongoDB: {e}")

session_service = InMemorySessionService()

def rename_agent(name):
    return re.sub(r'\W|^(?=\d)', '_', name)

def create_agent(apiKey):
    namespace = namespace_collection.find_one({"apiKey": apiKey})
    if not namespace:
        raise HTTPException(status_code=404, detail="Namespace not found")

    namespace_id = namespace["_id"]
    agent_data = db.Agent.find_one({"namespaceId": namespace_id, "isActive": True})

    if not agent_data:
        raise HTTPException(status_code=404, detail="No active agents found")

    agent_name = rename_agent(agent_data["name"])

    context_document = context_collection.find_one({"agentId": agent_data["_id"]})
    if not context_document:
        raise HTTPException(status_code=404, detail="Context not found")

    context_content = context_document["content"]
    full_instruction = f"{agent_data['basePrompt'].strip()}\n\nContext:\n{context_content.strip()}"

    assistant_agent = Agent(
        name=agent_name,
        model="gemini-2.0-flash",
        description=agent_data["role"],
        instruction=full_instruction
    )

    return assistant_agent

async def call_agent_async(query: str, user_id: str, session_id: str, runner: Runner):
    content = types.Content(role='user', parts=[types.Part(text=query)])
    final_response_text = "Agent did not produce a final response."

    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
        if event.is_final_response():
            if event.content and event.content.parts:
                final_response_text = event.content.parts[0].text
            elif event.actions and event.actions.escalate:
                final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
            break

    return final_response_text

# Request body model
class QueryRequest(BaseModel):
    apiKey: str
    user_id: str
    session_id: str
    query: str

@app.post("/chat")
async def ask_agent(request: QueryRequest):
    agent = create_agent(request.apiKey)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # This session will persist (within server lifetime)
    session = session_service.create_session(
        app_name="chatbot",
        user_id=request.user_id,
        session_id=request.session_id
    )

    runner = Runner(
        agent=agent,
        app_name="chatbot",
        session_service=session_service
    )

    response = await call_agent_async(agent, request.query, request.user_id, request.session_id, runner)
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)