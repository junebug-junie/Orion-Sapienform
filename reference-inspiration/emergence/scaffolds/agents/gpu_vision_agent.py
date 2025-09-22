import torch
import torchvision.transforms as T
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os

class GPUVisionAgent:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        print(f"[GPUVisionAgent] Initialized on {self.device}")

    def analyze_image(self, image_path, prompts):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(text=prompts, images=image, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        result = {prompt: prob.item() for prompt, prob in zip(prompts, probs[0])}
        print(f"[GPUVisionAgent] Results: {result}")
        return result

if __name__ == "__main__":
    agent = GPUVisionAgent()
    test_img = os.path.join("test_images", "bird.jpg")
    prompts = ["a bird", "a cat", "a dog"]
    agent.analyze_image(test_img, prompts)