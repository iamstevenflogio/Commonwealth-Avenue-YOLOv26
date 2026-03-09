import cv2
import numpy as np
from roboflow import Roboflow
import math

VIDEO_FILE = "test-inbound.mp4"

#rf = Roboflow(api_key="SvDHthV3LGvz5DlAKaMt")
rf = Roboflow(api_key="SvDHthV3LGvz5DlAKaMt")

project = rf.workspace().project("commonwealth-ave")
#model = project.version("3").model  # class "Cars"
model = project.version("5").model

cap = cv2.VideoCapture(VIDEO_FILE)
fps_in  = cap.get(cv2.CAP_PROP_FPS)
width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("zone_count.mp4", fourcc, fps_in, (width, height))

# polygon as you tuned it
inbound_polygon = np.array([
    (int(width * 0.26), int(height * 0.60)),   # left top
    (int(width * 0.72), int(height * 0.60)),   # right top
    (int(width * 1.72), int(height * 1.99)),   # right bottom
    (int(width * -0.10), int(height * 0.99)),  # left bottom
], dtype=np.int32)

frame_idx = 0

# set of centers we have already counted (x, y)
seen_centers = []
total_cars = 0
DIST_THRESH = 30  # pixels; how close is "same car"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = model.predict(frame, confidence=0.4)
    preds = result.json()["predictions"]

    cv2.polylines(frame, [inbound_polygon], True, (0, 255, 255), 2)

    cars_in_zone_now = 0  # instantaneous occupancy (still useful to see)

    for p in preds:
        if p["class"] != "Cars":
            continue

        x, y = p["x"], p["y"]
        w, h = p["width"], p["height"]

        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        center = (int(x), int(y))
        cv2.circle(frame, center, 3, (0, 0, 255), -1)

        inside = cv2.pointPolygonTest(inbound_polygon, center, False)
        if inside >= 0:
            cars_in_zone_now += 1

            # check if this car is "new" (not near any previously seen center)
            is_new = True
            for sx, sy in seen_centers:
                if math.hypot(center[0] - sx, center[1] - sy) < DIST_THRESH:
                    is_new = False
                    break

            if is_new:
                total_cars += 1
                seen_centers.append(center)

    cv2.putText(frame, f"Cars in zone: {cars_in_zone_now}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.putText(frame, f"Total cars passed: {total_cars}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    out.write(frame)
    frame_idx += 1
    print(f"Frame {frame_idx}, preds: {len(preds)}, in-zone: {cars_in_zone_now}, total: {total_cars}")

    if frame_idx >= 500:
        print("Stopping early at frame 500 for inspection")
        break

cap.release()
out.release()
print("Saved zone_count.mp4, total cars passed:", total_cars)
