"""
Flask API for Emotion Detection
"""

from flask import Flask, request, jsonify
from EmotionDetection.emotion_detection import emotion_detector

app = Flask(__name__)

@app.route('/emotionDetector', methods=['POST'])
def emotion_detector_api():
    """
    API endpoint to analyze emotions from a given text input.

    Returns:
        JSON response containing detected emotions.
    """
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "Invalid text! Please try again!"}), 400

    text_to_analyze = data["text"]
    response = emotion_detector(text_to_analyze)

    if response.get("dominant_emotion") is None:
        return jsonify({"error": "Invalid text! Please try again!"}), 400

    return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
