def compute_salience(signal):
    score = hash(signal) % 100  # Placeholder for real salience function
    print(f"[Salience] Signal: {signal} → Salience Score: {score}")
    return score