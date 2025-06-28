from client import ConjourneyClient

client = ConjourneyClient(base_url="http://localhost:8086")

results = client.query_collapse(prompt="rituals remembered")

print("Matches:")
for r in results["results"]:
    print(f"{r['metadata']['timestamp']} → {r['summary']}")
