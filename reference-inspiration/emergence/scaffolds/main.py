from core.engine import EmergentAISystem
from perception.vision import detect_bird
from perception.audio import detect_cat
from core.trigger_router import run_router
from agents.gpu_vision_agent import GPUVisionAgent
from perception.emotion_from_vision import EmotionFromVision
from cognition.motivation.motivation_engine import MotivationEngine
from memory.rdf_encoder import encode_emotion_rdf

def run_gpu_test():
    print("[Main] Running GPU Vision Agent Test")
    agent = GPUVisionAgent()
    image_path = "test_images/bird.jpg"
    prompts = ["a bird", "a cat", "a tree"]
    result = agent.analyze_image(image_path, prompts)
    print("[Main] GPU result:", result)
    
if __name__ == "__main__":
    print("[Main] Starting router and simulating inputs")
    run_router()
    detect_bird()
    detect_cat()
    run_gpu_test()


    image_path = "test_images/bird.jpg"

    efv = EmotionFromVision()
    emotion_vector = efv.detect_emotion(image_path)
    print("[EmotionFromVision]", emotion_vector)

    motivator = MotivationEngine()
    action = motivator.select_action(emotion_vector)
    print("[MotivationEngine] Selected Action:", action)

    rdf_triples = encode_emotion_rdf(emotion_vector, image_path)
    for triple in rdf_triples:
        print("[RDF]", triple)


from agents.gpu_vision_agent import GPUVisionAgent
from core.redis_pubsub import RedisPubSub
from memory.rdf_encoder import RDFEncoder
from narrative.insight_builder import build_insight
from narrative.collapse_mirror import record_collapse

redis_pub = RedisPubSub()

# Simulated vision task
agent = GPUVisionAgent()
image_path = "test_images/bird.jpg"
prompts = ["a bird", "a cat", "a tree"]
result = agent.analyze_image(image_path, prompts)

# Publish to Redis
redis_pub.publish({
    "type": "vision",
    "source": "gpu_vision",
    "image": image_path,
    "labels": result
})

# Simulated RDF ingestion listener
rdf = RDFEncoder()

def process_message(msg):
    if msg["type"] == "vision":
        rdf.reset()
        rdf.encode_vision_result(
            subject=msg["source"],
            image_uri=msg["image"],
            label_scores=msg["labels"]
        )
        insight = build_insight(msg)
        collapse = record_collapse(msg)
        print("[RDF]", rdf.serialize())
        print("[Insight]", insight)
        print("[Collapse]", collapse)

redis_pub.listen(process_message)



$$$$$$$$$$$$$$$$$$$


from agents.gpu_vision_agent import GPUVisionAgent, run_gpu_vision_test
from core.redis_pubsub import RedisPubSub
from memory.rdf_encoder import RDFEncoder
from narrative.insight_builder import build_insight
from narrative.collapse_mirror import record_collapse


def run_pipeline():
    redis_pub = RedisPubSub()
    agent = GPUVisionAgent()
    image_path = "test_images/bird.jpg"
    prompts = ["a bird", "a cat", "a tree"]
    result = agent.analyze_image(image_path, prompts)
    redis_pub.publish({
        "type": "vision",
        "source": "gpu_vision",
        "image": image_path,
        "labels": result
    })

    rdf = RDFEncoder()

    def process_message(msg):
        if msg["type"] == "vision":
            rdf.reset()
            rdf.encode_vision_result(
                subject=msg["source"],
                image_uri=msg["image"],
                label_scores=msg["labels"]
            )
            insight = build_insight(msg)
            collapse = record_collapse(msg)
            print("[RDF]", rdf.serialize())
            print("[Insight]", insight)
            print("[Collapse]", collapse)

    redis_pub.listen(process_message)

if __name__ == "__main__":
    run_pipeline()

