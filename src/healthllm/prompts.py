FEATURE_COLS = [
    "steps_daily",
    "sleep_minutes",
    "time_in_bed",
    "sleep_efficiency",
    "hr_mean",
    "hr_min",
    "hr_max",
    "hr_std",
    "resting_heart_rate",
    "calories_daily",
    "mood",
]

PROMPT_VERSION = "readiness_v1"


def _format_value(value):
    if value is None:
        return "null"
    try:
        if value != value:
            return "null"
    except Exception:
        pass
    return value


def format_feature_block(row, feature_cols=FEATURE_COLS):
    lines = []
    for col in feature_cols:
        lines.append(f"- {col}: {_format_value(row.get(col))}")
    return "\n".join(lines)


def format_few_shot_example(row, target_col="readiness", feature_cols=FEATURE_COLS):
    features = format_feature_block(row, feature_cols=feature_cols)
    target_value = row[target_col]
    return (
        "Example:\n"
        f"{features}\n"
        f'Output: {{"{target_col}": {float(target_value):.1f}}}'
    )


def build_readiness_prompt(row, few_shot_examples=None, feature_cols=FEATURE_COLS):
    system_prompt = (
        "You predict a user's daily readiness score from wearable features. "
        "Return only valid JSON with one key: readiness. "
        'Example format: {"readiness": 6.4}'
    )

    parts = [
        "Predict the user's readiness score from 0 to 10.",
        "Use one decimal place.",
    ]

    if few_shot_examples:
        example_text = "\n\n".join(
            format_few_shot_example(example, target_col="readiness", feature_cols=feature_cols)
            for example in few_shot_examples
        )
        parts.append(example_text)

    parts.append("Now predict for this row:")
    parts.append(format_feature_block(row, feature_cols=feature_cols))

    user_prompt = "\n\n".join(parts)
    return system_prompt, user_prompt
