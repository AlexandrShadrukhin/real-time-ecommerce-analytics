def predict(features):
    score = 0.1

    if features["is_cart"]:
        score += 0.5
    if features["is_click"]:
        score += 0.2
    if features["is_view"]:
        score += 0.1

    return min(score, 0.95)