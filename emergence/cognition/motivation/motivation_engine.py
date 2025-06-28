import random

class MotivationEngine:
    def __init__(self):
        self.activities = ["reflect", "dream", "simulate", "ignore"]

    def select_action(self, emotion_vector):
        dominant_emotion = max(emotion_vector, key=emotion_vector.get)
        # Simple heuristic mapping
        if dominant_emotion in ["awe", "curiosity"]:
            return "dream"
        elif dominant_emotion in ["fear", "anger"]:
            return "simulate"
        elif dominant_emotion in ["sadness", "confusion"]:
            return "reflect"
        else:
            return random.choice(self.activities)
