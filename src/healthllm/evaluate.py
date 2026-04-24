import pandas as pd
from sklearn.metrics import mean_absolute_error

from healthllm.predict import predict_readiness


def run_readiness_evaluation(eval_df, llm, few_shot_examples=None, target_col="readiness"):
    results = []

    for _, row in eval_df.iterrows():
        pred = predict_readiness(
            row=row,
            llm=llm,
            few_shot_examples=few_shot_examples,
        )

        results.append(
            {
                "participant_id": row.get("participant_id"),
                "date": row.get("date"),
                "actual_readiness": row[target_col],
                **pred,
            }
        )

    return pd.DataFrame(results)


def compute_readiness_metrics(results_df):
    valid_df = results_df[
        results_df["parse_success"] & results_df["predicted_readiness"].notna()
    ].copy()

    metrics = {
        "num_rows": len(results_df),
        "num_valid_predictions": len(valid_df),
        "parse_failure_rate": 1 - results_df["parse_success"].mean(),
        "mae": None,
    }

    if not valid_df.empty:
        metrics["mae"] = mean_absolute_error(
            valid_df["actual_readiness"],
            valid_df["predicted_readiness"],
        )

    return metrics