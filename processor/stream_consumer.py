from producer.event_generator import generate_event
from processor.feature_builder import build_features

def process():
    event = generate_event()
    features = build_features(event)

    print("Event:", event)
    print("Features:", features)

if __name__ == "__main__":
    process()