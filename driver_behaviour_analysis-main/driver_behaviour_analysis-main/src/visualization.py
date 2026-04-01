import cv2 as cv


def draw(frame, tracks, results):
    warming_up = isinstance(results, dict) and results.get("status") == "warming_up"

    results_map = {}
    if not warming_up:
        results_map = {r["id"]: r for r in results}

    colors = {
        "Smooth Driver": (0, 200, 0),       # green
        "Moderate Driver": (0, 165, 255),   # orange
        "Risky Driver": (0, 0, 220),        # red
    }

    for track in tracks:
        tid = track["id"]
        x1, y1, x2, y2 = track["bbox"]

        result = results_map.get(tid)

        if result:
            label = result["label"]
            score = result["score"]
            color = colors.get(label, (200, 200, 200))
        else:
            label = "Analyzing..."
            score = ""
            color = (200, 200, 200)  # gray

        cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        text = f"#{tid} | {label}" + (f" | Score: {score}" if score != "" else "")
        text_y = max(y1 - 10, 15)
        cv.putText(frame, text, (x1, text_y), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Overlay warming-up notice
    if warming_up:
        cv.putText(frame, "Warming up — collecting data...",
                   (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    return frame


