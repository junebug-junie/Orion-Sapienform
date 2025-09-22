current_focus = []

def add_focus_item(item):
    current_focus.append(item)
    print(f"[Introspect] Added to focus: {item}")

def get_current_focus():
    return current_focus

# --- introspection/meta_narrator.py ---
def narrate_thought(thought):
    print(f"[MetaNarrator] Thought narration: {thought}")
