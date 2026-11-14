
import os
import time

from dotenv import load_dotenv
from google import genai

load_dotenv()

# Try the lighter model first, then fall back if it is unavailable.
MODEL_NAMES = [
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
]


def generate_explanation(
    url: str,
    features: dict,
    prediction: str,
    decision_score: float,
) -> str:
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Check your project-root .env file."
        )

    selected_features = {
        key: features.get(key)
        for key in [
            "url_len",
            "is_ip",
            "subdom_cnt",
            "digit_ratio",
            "special_cnt",
            "is_https",
            "entropy",
            "path_len",
            "query_len",
        ]
        if key in features
    }

    prompt = f"""
You are URLSight's URL security explanation assistant.

Explain the result of an existing machine-learning classifier.

Classifier prediction: {prediction}
LinearSVC decision score: {decision_score:.4f}
Selected URL characteristics: {selected_features}

Explain:
1. What the classifier predicted.
2. Which URL characteristics may be relevant.
3. What the user should do next.

Rules:
- Do not change or override the classifier's prediction.
- A negative score represents the legitimate class; a positive score
  represents the phishing class for this model.
- Do not claim that you visited or inspected the website.
- Do not invent website reputation, content, or external scan results.
- Do not claim URL characteristics prove that a site is malicious.
- Explain that legitimate predictions do not guarantee safety.
- Keep the response concise and easy to understand.
"""

    client = genai.Client(api_key=api_key)
    errors = []

    for model_name in MODEL_NAMES:
        # Retry temporary server errors with increasing delays.
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )

                if response.text:
                    return response.text

                raise RuntimeError("Gemini returned an empty response.")

            except Exception as exc:
                error_text = str(exc)
                errors.append(f"{model_name}: {error_text}")

                # Only retry likely temporary errors.
                temporary = any(
                    marker in error_text
                    for marker in [
                        "503",
                        "UNAVAILABLE",
                        "429",
                        "RESOURCE_EXHAUSTED",
                        "500",
                        "INTERNAL",
                        "504",
                        "DEADLINE_EXCEEDED",
                    ]
                )

                if temporary and attempt == 0:
                    time.sleep(2)
                    continue

                # Try the next model for model availability/server issues.
                if any(
                    marker in error_text
                    for marker in [
                        "503",
                        "UNAVAILABLE",
                        "404",
                        "NOT_FOUND",
                        "500",
                        "INTERNAL",
                        "504",
                        "DEADLINE_EXCEEDED",
                    ]
                ):
                    break

                # For authentication, permission, or other errors,
                # stop rather than repeatedly sending the same request.
                raise RuntimeError(
                    f"Gemini request failed for {model_name}: {exc}"
                ) from exc

    raise RuntimeError(
        "Gemini models could not generate an explanation right now. "
        "Your SVM result is still available. Please try again later. "
        + " | ".join(errors[-2:])
    )
