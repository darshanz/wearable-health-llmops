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
from healthllm.rf_model import load_rf_artifact, predict_readiness_rf


 

MODEL_NAME = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")

app = FastAPI(title="Wearable Health LLMOps API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# LLM
llm = get_ollama_llm(model=MODEL_NAME)

# Random Forest
rf_artifact = load_rf_artifact()



@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="ok")


@app.post("/predict/readiness", response_model=ReadinessPredictionResponse)
def predict_readiness_endpoint(payload: ReadinessPredictionRequest):
    row = payload.model_dump()

    model_type = row.pop("model_type")
    if model_type == "random_forest":
        result = predict_readiness_rf(row=row, artifact=rf_artifact)
        model_name = "random_forest_baseline"
    else:
        result = predict_readiness(
            row=row,
            llm=llm,
            few_shot_examples=None,
        )
        model_name = MODEL_NAME

    return ReadinessPredictionResponse(
        predicted_readiness=result["predicted_readiness"],
        parse_success=result["parse_success"],
        prompt_version=result["prompt_version"],
        model_name=model_name,
        error_message=result["error_message"],
        raw_response=result["raw_response"],
    )
