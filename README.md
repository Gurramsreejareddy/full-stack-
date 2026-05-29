import os
import json
import asyncio
from typing import AsyncGenerator, Dict, Any, List
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, HttpUrl

# LangChain / LangGraph
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

app = FastAPI(title="Video Analytics RAG Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global in-memory VectorDB for demo purposes (persistent path can be passed for production)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = Chroma(embedding_function=embeddings)

# --- Schemas ---
class ProcessRequest(BaseModel):
    youtube_url: HttpUrl
    instagram_url: HttpUrl

class ChatPayload(BaseModel):
    message: str
    history: List[Dict[str, str]] = []

# --- Mock Extraction Layer (Replace with actual yt-dlp / scrapers) ---
async def extract_video_data(url: str, platform: str) -> Dict[str, Any]:
    """
    Simulates highly dynamic, robust extraction. 
    In production, wrap yt-dlp and a scraping API (like Apify) here.
    """
    await asyncio.sleep(1.5) # Simulate network lag
    
    if "youtube" in platform.lower():
        return {
            "video_id": "A",
            "creator": "TechReviewer Pro",
            "follower_count": 850000,
            "views": 250000,
            "likes": 18000,
            "comments": 1200,
            "duration": 58,
            "upload_date": "2026-05-15",
            "transcript": "Hey guys! First 5 seconds is crucial. Today I am showing you the hidden feature of this phone that nobody talks about. The screen panel is completely custom made. This is why it outperforms everything else on the market.",
            "hook": "Hey guys! First 5 seconds is crucial. Today I am showing you the hidden feature..."
        }
    else:
        return {
            "video_id": "B",
            "creator": "CreativeShorts",
            "follower_count": 1200000,
            "views": 500000,
            "likes": 15000,
            "comments": 600,
            "duration": 45,
            "upload_date": "2026-05-20",
            "transcript": "Look at this device. It looks pretty normal right? Well it isn't. Watch until the end to see what happens when I drop it. Most people buy this for the aesthetics, but the build quality is actually mediocre.",
            "hook": "Look at this device. It looks pretty normal right? Well it isn't."
        }

# --- API Endpoints ---
@app.post("/api/process")
async def process_videos(payload: ProcessRequest):
    try:
        # 1. Concurrent Extraction
        yt_task = extract_video_data(str(payload.youtube_url), "youtube")
        ig_task = extract_video_data(str(payload.instagram_url), "instagram")
        yt_data, ig_data = await asyncio.gather(yt_task, ig_task)
        
        # 2. Compute Engagement Rates
        for data in [yt_data, ig_data]:
            data["engagement_rate"] = round(((data["likes"] + data["comments"]) / data["views"]) * 100, 2)
            
        # 3. Text Chunking & Meta Tagging
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
        documents = []
        
        for data in [yt_data, ig_data]:
            chunks = text_splitter.split_text(data["transcript"])
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "video_id": data["video_id"],
                        "chunk_index": i,
                        "creator": data["creator"],
                        "engagement_rate": data["engagement_rate"],
                        "views": data["views"],
                        "follower_count": data["follower_count"],
                        "hook": data["hook"]
                    }
                )
                documents.append(doc)
                
        # Clear database and re-index freshly parsed URLs (dynamic rule requirement)
        # Note: In production, switch this to a multi-tenant or session-isolated collection
        global vector_store
        vector_store = Chroma.from_documents(documents, embedding_function=embeddings)
        
        return {
            "status": "success",
            "meta_summary": {
                "A": {k: v for k, v in yt_data.items() if k != 'transcript'},
                "B": {k: v for k, v in ig_data.items() if k != 'transcript'}
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def chat_stream(payload: ChatPayload):
    """
    Exposes an SSE stream providing answers, execution states, and direct chunk citations.
    """
    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            # 1. Retrieve Context with citations
            query = payload.message
            retrieved_docs = vector_store.similarity_search(query, k=4)
            
            citations = []
            context_str = ""
            for doc in retrieved_docs:
                vid = doc.metadata.get("video_id")
                c_idx = doc.metadata.get("chunk_index")
                citations.append({"video_id": vid, "chunk_index": c_idx})
                
                context_str += f"\n[Video ID: {vid} | Creator: {doc.metadata.get('creator')} | ER: {doc.metadata.get('engagement_rate')}%]\nContent: {doc.page_content}\n"

            # Send the citations metadata first down the pipeline stream
            yield f"data: {json.dumps({'type': 'citations', 'data': citations})}\n\n"
            await asyncio.sleep(0.01)

            # 2. LangGraph State Flow Integration
            # Build execution node
            async def call_llm(state: Dict[str, Any]) -> Dict[str, Any]:
                llm = ChatOpenAI(model="gpt-4o", streaming=True, temperature=0.2)
                sys_msg = SystemMessage(
                    content=(
                        "You are an elite social media growth engineer. Analyze the two videos based ONLY on the provided context.\n"
                        f"Retrieved Video Context:\n{context_str}\n\n"
                        "When comparing metrics or strategies (like hooks or engagement rates), be precise and cite the facts provided. "
                        "If you do not know or if it is not in the context, explain that missing data constraints prevent an answer."
                    )
                )
                
                messages = [sys_msg]
                for msg in state["history"]:
                    if msg["role"] == "user":
                        messages.append(HumanMessage(content=msg["content"]))
                    else:
                        messages.append(AIMessage(content=msg["content"]))
                        
                messages.append(HumanMessage(content=state["current_message"]))
                
                # Streaming token execution loop
                full_response = ""
                async for chunk in llm.astream(messages):
                    if chunk.content:
                        full_response += chunk.content
                        yield f"data: {json.dumps({'type': 'token', 'data': chunk.content})}\n\n"
                
                # Return updated state modifier 
                return {"messages": [AIMessage(content=full_response)]}

            # Define Simple LangGraph Engine
            workflow = StateGraph(state_schema=Dict[str, Any])
            workflow.add_node("llm_node", call_llm)
            workflow.set_entry_point("llm_node")
            workflow.add_edge("llm_node", END)
            graph = workflow.compile()

            # Execute graph generator flow
            initial_state = {
                "current_message": query,
                "history": payload.history,
                "messages": []
            }
            
            async for output in graph.astream(initial_state):
                # The tokens are already streamed out dynamically via the inner llm loop generator
                pass
                
            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main.py", host="0.0.0.0", port=8000, reload=True)