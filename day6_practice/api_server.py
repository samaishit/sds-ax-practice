# api_server.py - 종합 시나리오 앱을 API로 노출
# 이 실습에만 필요한 패키지입니다: pip install fastapi uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from final_scenario import build_app, get_text
from contextlib import asynccontextmanager

graph_app = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global graph_app
    graph_app = await build_app()
    yield

api = FastAPI(lifespan=lifespan)

class Query(BaseModel):
    question: str

@api.post("/chat")
async def chat(q: Query):
    result = await graph_app.ainvoke({"messages": [HumanMessage(q.question)]})
    answer_text = get_text(result["messages"][-1])
    
    # JSONResponse를 사용하여 charset=utf-8을 명시적으로 전달
    return JSONResponse(
        content={"answer": answer_text},
        headers={"Content-Type": "application/json; charset=utf-8"}
    )

# 실행: uvicorn api_server:api --port 8000
# 테스트: Invoke-RestMethod -Uri http://localhost:8000/chat -Method Post -ContentType "application/json" -Body '{"question": "출장 식비 한도 얼마야?"}'