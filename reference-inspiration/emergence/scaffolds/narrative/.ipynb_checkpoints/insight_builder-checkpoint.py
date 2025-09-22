def build_insight(event):
    print("[InsightBuilder] Building insight from event")
    return {
        "insight": f"Analyzed {event['labels']} from {event['source']}",
        "depth": len(event['labels'])
    }