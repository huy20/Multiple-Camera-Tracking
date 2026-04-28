import time

import cv2
import numpy as np
import torch
import torchreid
from ultralytics import YOLO


class PersonTracker:
    def __init__(self, model_path="yolo11s.pt", reid_weights="osnet_x0_75_msmt17.pth", tracker_config="custom_tracker.yaml"):
        self.model = YOLO(model_path, task="detect")
        self.tracker_config = tracker_config

        # ===== LOAD OSNET DIRECTLY =====
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.reid_model = torchreid.models.build_model(
            name="osnet_x0_75",
            num_classes=1000,
            pretrained=False
        )

        torchreid.utils.load_pretrained_weights(self.reid_model, reid_weights)

        self.reid_model.to(self.device)
        self.reid_model.eval()
        self.reid_model = self.reid_model.float()

        # ===== NORMALIZATION =====
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

        # ===== ReID GLOBAL DB =====
        self.embedding_db = {}
        self.last_seen = {}
        self.next_global_id = 0
        self.track_memory = {}

        self.MIN_HITS = 3     # frames to confirm new ID
        self.MAX_MISS = 10    # frames to forget track

        self.SIM_THRESHOLD = 0.7
        self.TTL = 30
        self.MAX_DB = 1000

        # ===== UI =====
        self.COLOR_PALETTE = np.array([
            (255, 50, 50), (50, 255, 50), (50, 50, 255),
            (255, 255, 50), (50, 255, 255), (255, 50, 255),
            (255, 150, 50), (150, 50, 255), (50, 150, 255), (150, 255, 50)
        ], dtype=np.uint8)

    # ===== EMBEDDING =====
    def get_embedding(self, frame, box):
        x1, y1, x2, y2 = map(int, box)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        # resize (OSNet expects 256x128 HxW)
        img = cv2.resize(crop, (128, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std

        img = np.transpose(img, (2, 0, 1))  # CHW
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.reid_model(img)

        feat = feat / feat.norm(p=2, dim=1, keepdim=True)

        return feat.cpu().numpy().flatten()

    # ===== MATCHING =====
    def match_embedding(self, emb):
        now = time.time()

        # cleanup
        to_delete = [gid for gid, t in self.last_seen.items() if now - t > self.TTL]
        for gid in to_delete:
            self.embedding_db.pop(gid, None)
            self.last_seen.pop(gid, None)

        if len(self.embedding_db) == 0:
            gid = self.next_global_id
            self.embedding_db[gid] = emb
            self.last_seen[gid] = now
            self.next_global_id += 1
            return gid

        ids = list(self.embedding_db.keys())
        embs = np.array(list(self.embedding_db.values()))

        sims = embs @ emb / (np.linalg.norm(embs, axis=1) * np.linalg.norm(emb) + 1e-6)

        best_idx = np.argmax(sims)
        best_score = sims[best_idx]
        best_id = ids[best_idx]
        
        print(best_id, best_score)

        if best_score > self.SIM_THRESHOLD:
            self.embedding_db[best_id] = 0.75 * self.embedding_db[best_id] + 0.25 * emb
            self.last_seen[best_id] = now
            return best_id
        else:
            gid = self.next_global_id
            self.embedding_db[gid] = emb
            self.last_seen[gid] = now
            self.next_global_id += 1

            if len(self.embedding_db) > self.MAX_DB:
                oldest = min(self.last_seen, key=self.last_seen.get)
                self.embedding_db.pop(oldest)
                self.last_seen.pop(oldest)

            return gid

    # ===== TRACK =====
    def track(self, source_url, imgsz=480, conf=0.3):
        results = self.model.track(
            source=source_url,
            conf=conf,
            imgsz=imgsz,
            stream=True,
            tracker=self.tracker_config,
            persist=True,
            classes=[0]
        )

        for r in results:
            frame = r.orig_img
            current_detections = []

            if r.boxes is not None and r.boxes.id is not None:
                data = r.boxes.data.cpu().numpy()

                for row in data:
                    if len(row) == 7:
                        x1, y1, x2, y2, track_id, score, cls_id = row
                    else:
                        x1, y1, x2, y2, track_id, score = row

                    if score < 0.4:
                        continue

                    t_id = int(track_id)
                    box = [int(x1), int(y1), int(x2), int(y2)]

                    emb = self.get_embedding(frame, box)

                    if emb is not None:
                        global_id = self.match_embedding(emb)
                    else:
                        global_id = t_id

                    current_detections.append({
                        "id": t_id,
                        "global_id": global_id,
                        "bbox": box,
                        "conf": float(score),
                        "base_point": (int((x1 + x2) / 2), int(y2))
                    })

                    color = self.COLOR_PALETTE[global_id % len(self.COLOR_PALETTE)].tolist()

                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 3)
                    cv2.putText(frame, f"GID:{global_id}", (box[0], box[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            yield frame, current_detections