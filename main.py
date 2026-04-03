import cv2
import numpy as np
from ultralytics import YOLO


VIDEO_PATH = "Inbound.mp4"
OUTPUT_PATH = "output_inbound.mp4"
MODEL_PATH = "yolo26n.pt"  


POLYGON = np.array([
    [662, 454],
    [-200, 1026],
    [1860, 1036],
    [1194, 460]
], dtype=np.int32)


# COCO class IDs in Ultralytics/COCO
TARGET_CLASSES = {
    2: "car",
    3: "motorcycle",
    7: "truck"
}


model = YOLO(MODEL_PATH)


cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")


fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))


cv2.namedWindow("Polygon ROI", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Polygon ROI", 960, 540)

while True:
    ret, frame = cap.read()
    if not ret:
        break


    results = model.track(frame, persist=True, conf=0.20, imgsz=960, verbose=False)
    annotated = frame.copy()


    cv2.polylines(annotated, [POLYGON], isClosed=True, color=(0, 255, 255), thickness=2)


    if results and results[0].boxes is not None:
        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy().astype(int)
        cls_ids = boxes.cls.cpu().numpy().astype(int)


        track_ids = None
        if boxes.id is not None:
            track_ids = boxes.id.cpu().numpy().astype(int)


        for i, (x1, y1, x2, y2) in enumerate(xyxy):
            cls_id = cls_ids[i]


            if cls_id not in [2, 3, 7]:
                continue


            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)


            inside = cv2.pointPolygonTest(POLYGON, (cx, cy), False)
            if inside < 0:
                continue


            label = ["car", "motorcycle", "truck"][ [2, 3, 7].index(cls_id) ]
            if track_ids is not None:
                label = f"{label} ID:{track_ids[i]}"


            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(annotated, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


    cv2.imshow("Polygon ROI", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()