import os
import json
import time
from datetime import datetime

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import requests

USE_SNAPSHOT_MODE = False
STREAM_URL = "http://192.168.0.170:8080/video"
SNAPSHOT_URL = "http://192.168.0.170:8080/shot.jpg"

MODEL_PATH = "G:/PestVision_1/models/best.pt"

N8N_WEBHOOK_URL = "http://localhost:5678/webhook-test/pest-event"

CAMERA_ID = "top_camera_1"
PLANT_ID = "plant_1"

OUT_FULL_DIR = "G:/PestVision_1/data/snapshots/full"
OUT_CROP_DIR = "G:/PestVision_1/data/snapshots/crop"
OUT_PAYLOAD_DIR = "G:/PestVision_1/data/payloads"

CONF_THRESHOLD = 0.45
MAX_EVENTS_PER_MIN = 12
CROP_MARGIN = 0.30

MAX_BOX_WIDTH_RATIO = 0.40
MAX_BOX_HEIGHT_RATIO = 0.40

RECONNECT_DELAY = 1.0
SNAPSHOT_TIMEOUT = 2.0
POST_TIMEOUT = 6.0

def ensure_dirs():
    os.makedirs(OUT_FULL_DIR, exist_ok=True)
    os.makedirs(OUT_CROP_DIR, exist_ok=True)
    os.makedirs(OUT_PAYLOAD_DIR, exist_ok=True)

def ts_compact():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ts_iso():
    return datetime.now().isoformat(timespec="seconds")

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def crop_with_margin(frame, x1, y1, x2, y2, margin=0.30, square=True):
    
    h, w = frame.shape[:2]
    bw = x2 - x1
    bh = y2 - y1

    pad_x = int(bw * margin)
    pad_y = int(bh * margin)

    cx1 = clamp(x1 - pad_x, 0, w - 1)
    cy1 = clamp(y1 - pad_y, 0, h - 1)
    cx2 = clamp(x2 + pad_x, 0, w - 1)
    cy2 = clamp(y2 + pad_y, 0, h - 1)

    if square:
        # make crop square
        crop_w = cx2 - cx1
        crop_h = cy2 - cy1
        diff = abs(crop_w - crop_h)
        if crop_w > crop_h:
            cy1 = clamp(cy1 - diff // 2, 0, h - 1)
            cy2 = clamp(cy2 + diff - diff // 2, 0, h - 1)
        else:
            cx1 = clamp(cx1 - diff // 2, 0, w - 1)
            cx2 = clamp(cx2 + diff - diff // 2, 0, w - 1)

    if cy2 <= cy1 or cx2 <= cx1:
        return None

    return frame[cy1:cy2, cx1:cx2].copy()

def save_image(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)

def fetch_snapshot():
    try:
        r = requests.get(SNAPSHOT_URL, timeout=SNAPSHOT_TIMEOUT)
        if r.status_code != 200:
            return None
        arr = np.frombuffer(r.content, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

def send_to_n8n(payload, full_path, crop_path):
    if not N8N_WEBHOOK_URL:
        return False, "Webhook URL missing."
    try:
        with open(full_path, "rb") as f_full, open(crop_path, "rb") as f_crop:
            files = {
                "full_image": ("full.jpg", f_full, "image/jpeg"),
                "crop_image": ("crop.jpg", f_crop, "image/jpeg"),
            }
            data = {"metadata": json.dumps(payload, ensure_ascii=False)}
            r = requests.post(N8N_WEBHOOK_URL, data=data, files=files, timeout=POST_TIMEOUT)
            return r.ok, f"HTTP {r.status_code}"
    except Exception as e:
        return False, str(e)

def main():
    global USE_SNAPSHOT_MODE
    ensure_dirs()

    print("[INFO] Loading transfer-learned model:", MODEL_PATH)
    model = YOLO(MODEL_PATH)

    if torch.cuda.is_available():
        try:
            model.to("cuda:0")
            print("[INFO] Using GPU:", torch.cuda.get_device_name(0))
        except Exception as e:
            print("[WARN] Could not move model to GPU:", e)
    else:
        print("[INFO] GPU not available, using CPU.")

    cap = None
    if not USE_SNAPSHOT_MODE:
        print("[INFO] Opening MJPEG stream...")
        cap = cv2.VideoCapture(STREAM_URL, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        time.sleep(0.5)

    last_event_times = []
    reconnect_attempts = 0

    print("[INFO] Detector running. Press ESC to quit.\n")

    while True:
        if USE_SNAPSHOT_MODE:
            frame = fetch_snapshot()
            if frame is None:
                print("[WARN] snapshot fetch failed, retrying...")
                time.sleep(0.2)
                continue
        else:
            if cap is None or not cap.isOpened():
                print("[WARN] stream closed, reconnecting...")
                reconnect_attempts += 1
                time.sleep(RECONNECT_DELAY)
                try:
                    cap = cv2.VideoCapture(STREAM_URL, cv2.CAP_FFMPEG)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    cap = None
                if cap is None or not cap.isOpened() and reconnect_attempts > 5:
                    print("[WARN] switching to snapshot mode (stream unreliable).")
                    USE_SNAPSHOT_MODE = True
                    continue

            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

        try:
            results = model.predict(source=frame, conf=CONF_THRESHOLD, verbose=False)
        except Exception as e:
            print("[ERROR] model.predict failed:", e)
            time.sleep(0.2)
            continue

        dets = None
        try:
            dets = results[0].boxes
        except Exception:
            dets = None

        if dets is not None and len(dets) > 0:
            try:
                best = max(dets, key=lambda d: float(d.conf[0]))
                x1, y1, x2, y2 = map(int, best.xyxy[0].tolist())
                conf = float(best.conf[0])
            except Exception:
                best = None
                conf = 0.0

            if best is not None:
                H, W = frame.shape[:2]
                bw = (x2 - x1) / W
                bh = (y2 - y1) / H

                if bw > MAX_BOX_WIDTH_RATIO or bh > MAX_BOX_HEIGHT_RATIO:
                    cv2.putText(frame, "IGNORED LARGE OBJECT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                else:
                    now = time.time()
                    last_event_times = [t for t in last_event_times if now - t < 60]
                    if len(last_event_times) < MAX_EVENTS_PER_MIN:
                        last_event_times.append(now)
                        ts = ts_compact()
                        full_path = os.path.join(OUT_FULL_DIR, f"full_{ts}.jpg")
                        crop_path = os.path.join(OUT_CROP_DIR, f"crop_{ts}.jpg")

                        save_image(frame, full_path)
                        crop_img = crop_with_margin(frame, x1, y1, x2, y2, margin=CROP_MARGIN)
                        if crop_img is None or crop_img.size == 0:
                            print("[WARN] crop empty, skipping save/send.")
                        else:
                            save_image(crop_img, crop_path)

                            payload = {
                                "event_type": "pest_detected",
                                "timestamp": ts_iso(),
                                "camera_id": CAMERA_ID,
                                "plant_id": PLANT_ID,
                                "detector": "yolo11_pest",
                                "confidence": conf,
                                "bbox": [x1, y1, x2, y2],
                                "snapshot_full_path": full_path,
                                "snapshot_crop_path": crop_path
                            }

                            payload_file = os.path.join(OUT_PAYLOAD_DIR, f"payload_{ts}.json")
                            with open(payload_file, "w", encoding="utf-8") as f:
                                json.dump(payload, f, indent=2, ensure_ascii=False)

                            ok, msg = send_to_n8n(payload, full_path, crop_path)
                            print(f"[EVENT] sent={ok} | {msg} | saved: {full_path}")

                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.putText(frame, f"pest {conf:.2f}", (x1, max(0, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.imshow("Pest Detector (YOLO11)", frame)
        if cv2.waitKey(1) == 27:
            break

    if cap and cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
