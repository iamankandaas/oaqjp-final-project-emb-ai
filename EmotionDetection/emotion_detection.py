import json
import requests

API_URL = "https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
HEADERS = {
    "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock",
    "Content-Type": "application/json"
}

def emotion_detector(text):
    if not text.strip():
        return {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None
        }

    input_data = {"raw_document": {"text": text}}
    response = requests.post(API_URL, headers=HEADERS, json=input_data)

    if response.status_code == 400:  # Handle empty input error from API
        return {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None
        }

    try:
        response_dict = response.json()
    except json.JSONDecodeError:
        return {"error": "Invalid response from API"}

    emotions = response_dict.get("emotionPredictions", [{}])[0].get("emotion", {})
    dominant_emotion = max(emotions, key=emotions.get) if emotions else None

    return {
        "anger": emotions.get("anger", 0),
        "disgust": emotions.get("disgust", 0),
        "fear": emotions.get("fear", 0),
        "joy": emotions.get("joy", 0),
        "sadness": emotions.get("sadness", 0),
        "dominant_emotion": dominant_emotion
    }
