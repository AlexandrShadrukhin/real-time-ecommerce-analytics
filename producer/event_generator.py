import random
import uuid
from datetime import datetime

EVENT_TYPES = ["view", "click", "add_to_cart", "purchase"]

def generate_event():
    return {
        "event_id": str(uuid.uuid4()),
        "user_id": random.randint(1, 100),
        "event_type": random.choice(EVENT_TYPES),
        "timestamp": datetime.utcnow().isoformat(),
        "price": random.randint(100, 5000)
    }

if __name__ == "__main__":
    print(generate_event())