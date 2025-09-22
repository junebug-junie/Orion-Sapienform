def encode_emotion_rdf(emotion_vector, image_path):
    triples = []
    for emotion, score in emotion_vector.items():
        triples.append((image_path, "hasEmotion", f"{emotion}:{score:.2f}"))
    return triples
