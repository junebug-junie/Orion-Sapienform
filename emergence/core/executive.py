from perception.salience import compute_salience

def should_interrupt(signal):
    score = compute_salience(signal)
    if score > 80:
        print(f"[Executive] High salience detected for: {signal}. Triggering override.")
        return True
    print(f"[Executive] Signal {signal} below threshold.")
    return False
