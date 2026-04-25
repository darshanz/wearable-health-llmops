import os
from fastapi import FastAPI

from healthllm.api.schemas import (
    HealthResponse,
    ReadinessPredictionRequest,
    ReadinessPredictionResponse,
)
from healthllm.llm_client import get_ollama_llm
from healthllm.predict import predict_readiness
from fastapi.middleware.cors import CORSMiddleware


 

MODEL_NAME = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")

app = FastAPI(title="Wearable Health LLMOps API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


llm = get_ollama_llm(model=MODEL_NAME)



@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="ok")


@app.post("/predict/readiness", response_model=ReadinessPredictionResponse)
def predict_readiness_endpoint(payload: ReadinessPredictionRequest):
    row = payload.model_dump()

    result = predict_readiness(
        row=row,
        llm=llm,
        few_shot_examples=None,
    )

    return ReadinessPredictionResponse(
        predicted_readiness=result["predicted_readiness"],
        parse_success=result["parse_success"],
        prompt_version=result["prompt_version"],
        model_name=MODEL_NAME,
        error_message=result["error_message"],
        raw_response=result["raw_response"],
    )
