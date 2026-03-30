def build_features(event):
    return {
        "user_id": event["user_id"],
        "is_click": int(event["event_type"] == "click"),
        "is_view": int(event["event_type"] == "view"),
        "is_cart": int(event["event_type"] == "add_to_cart"),
        "price": event["price"]
    }