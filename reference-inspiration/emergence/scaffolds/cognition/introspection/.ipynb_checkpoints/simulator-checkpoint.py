from introspection.agent_state import add_focus_item
from introspection.meta_narrator import narrate_thought

def simulate_internal_thought(trigger):
    print(f"[Simulator] Generating internal simulation for: {trigger}")
    thought = f"If {trigger} happened, what would follow?"
    add_focus_item(thought)
    narrate_thought(thought)
