from event_generator import generate_event

def send_event():
    event = generate_event()
    print("Send to Kafka:", event)

if __name__ == "__main__":
    send_event()