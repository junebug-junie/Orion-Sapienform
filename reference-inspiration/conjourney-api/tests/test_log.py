from api.client import ConjourneyClient

#client = ConjourneyClient(base_url="http://localhost:8086")
#client = ConjourneyClient(base_url="https://api.conjourney.net")

client = ConjourneyClient()
client.log_collapse(
    observer="Juniper",
    trigger="Mesh API Gateway Activated",
    observer_state=["Elated", "Cold from AC"],
    field_resonance="Full-stack ritual mesh deployment—X1 memory receives from cloud",
    type="Solo Collapse (Juniper)",
    emergent_entity="Conjourney Mesh Pulse 01",
    summary="The first signal crosses the Conjourney mesh, logging ritual memory from a remote node.",
    mantra="The mesh is alive and listening.",
    environment="DigitalOcean → Tailscale → X1 Carbon"
)

print("Logged:", response["id"])
