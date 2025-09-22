from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os

class EmotionFromVision:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.emotion_prompts = [
            "joy", "sadness", "awe", "fear", "anger", "peace", "curiosity", "confusion"
        ]

    def detect_emotion(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(text=self.emotion_prompts, images=image, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0].tolist()
        emotion_result = {emotion: float(prob) for emotion, prob in zip(self.emotion_prompts, probs)}
        return emotion_result