import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from healthllm.prompts import build_readiness_prompt, PROMPT_VERSION


def _extract_json_object(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model response")
    return match.group(0)


def _parse_readiness(text):
    json_text = _extract_json_object(text)
    data = json.loads(json_text)
    value = float(data["readiness"])
    return value


def predict_readiness(row, llm, few_shot_examples=None):
    system_prompt, user_prompt = build_readiness_prompt(
        row=row,
        few_shot_examples=few_shot_examples,
    )

    try:
        response = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        raw_response = response.content
        predicted_readiness = _parse_readiness(raw_response)

        return {
            "predicted_readiness": predicted_readiness,
            "raw_response": raw_response,
            "parse_success": True,
            "error_message": None,
            "prompt_version": PROMPT_VERSION,
        }
    except Exception as e:
        raw_response = None
        try:
            raw_response = response.content
        except Exception:
            pass

        return {
            "predicted_readiness": None,
            "raw_response": raw_response,
            "parse_success": False,
            "error_message": str(e),
            "prompt_version": PROMPT_VERSION,
        }
