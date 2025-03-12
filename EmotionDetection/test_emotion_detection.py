import unittest
from EmotionDetection.emotion_detection import emotion_detector

class TestEmotionDetector(unittest.TestCase):
    def test_joy(self):
        result = emotion_detector('{"emotion": {"joy": 0.9, "sadness": 0.1}}')
        self.assertEqual(result["dominant_emotion"], "joy")

    def test_anger(self):
        result = emotion_detector('{"emotion": {"anger": 0.8, "joy": 0.2}}')
        self.assertEqual(result["dominant_emotion"], "anger")

    def test_fear(self):
        result = emotion_detector('{"emotion": {"fear": 0.7, "joy": 0.3}}')
        self.assertEqual(result["dominant_emotion"], "fear")

    def test_sadness(self):
        result = emotion_detector('{"emotion": {"sadness": 0.6, "joy": 0.4}}')
        self.assertEqual(result["dominant_emotion"], "sadness")

    def test_disgust(self):
        result = emotion_detector('{"emotion": {"disgust": 0.5, "joy": 0.4}}')
        self.assertEqual(result["dominant_emotion"], "disgust")

if __name__ == "__main__":
    unittest.main()
