import cv2
import faiss
import numpy as np
import torch
import torchreid
from ultralytics import YOLO

class PersonTracker:
    def __init__(self, model_path="yolo11s.pt", reid_weights="osnet_x0_75_msmt17.pth", tracker_config="custom_tracker.yaml"):
        self.model = YOLO(model_path, task="detect")
        self.tracker_config = tracker_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ===== LOAD OSNET =====
        self.reid_model = torchreid.models.build_model(name="osnet_x0_75", num_classes=1000, pretrained=False)
        torchreid.utils.load_pretrained_weights(self.reid_model, reid_weights)
        self.reid_model.to(self.device)
        self.reid_model.eval()
        self.reid_model = self.reid_model.float()

        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

        # ===== FAISS & REID LOGIC =====
        self.dim = 512
        self.index = faiss.IndexFlatIP(self.dim)
        self.id_map = []
        self.next_global_id = 0
        
        # Strategy 2: Stricter threshold for better matching
        self.SIM_THRESHOLD = 0.9
        self.trackid_to_global = {}
        
        # Strategy 2: Buffer to wait for stable frames before fingerprinting
        self.track_history = {} 

        self.COLOR_PALETTE = np.array([
            (255, 50, 50), (50, 255, 50), (50, 50, 255),
            (255, 255, 50), (50, 255, 255), (255, 50, 255),
            (255, 150, 50), (150, 50, 255), (50, 150, 255), (150, 255, 50)
        ], dtype=np.uint8)

    # ===== STRATEGY 3: OVERLAP FILTER =====
    def get_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0: return 0
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea)

    def get_embedding(self, frame, box):
        x1, y1, x2, y2 = map(int, box)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0: return None
        
        img = cv2.resize(crop, (128, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = np.transpose(img, (2, 0, 1))
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.reid_model(img)
        if isinstance(feat, (list, tuple)): feat = feat[0]
        feat = feat.view(feat.size(0), -1)
        feat = feat / feat.norm(p=2, dim=1, keepdim=True)
        return feat.cpu().numpy().flatten().astype(np.float32)

    def track(self, source_url, imgsz=480, conf=0.3):
        results = self.model.track(source=source_url, conf=conf, imgsz=imgsz, stream=True, tracker=self.tracker_config, persist=True, classes=[0])

        for r in results:
            frame = r.orig_img
            current_detections = []
            if r.boxes is not None and r.boxes.id is not None:
                data = r.boxes.data.cpu().numpy()
                all_boxes = data[:, :4]

                for row in data:
                    x1, y1, x2, y2, track_id, score, _ = row if len(row)==7 else (*row, 0)
                    t_id = int(track_id)
                    box = [int(x1), int(y1), int(x2), int(y2)]

                    if t_id in self.trackid_to_global:
                        global_id = self.trackid_to_global[t_id]
                    else:
                        # ===== STRATEGY 3: CHECK FOR OCCLUSION =====
                        is_occluded = False
                        for other_box in all_boxes:
                            if np.array_equal(box, other_box.astype(int)): continue
                            if self.get_iou(box, other_box) > 0.15: # 15% overlap threshold
                                is_occluded = True
                                break

                        # ===== STRATEGY 2: TEMPORAL BUFFER (Wait for 5 stable frames) =====
                        if not is_occluded:
                            self.track_history[t_id] = self.track_history.get(t_id, 0) + 1
                            
                            if self.track_history[t_id] >= 5: # Stable for 5 frames
                                emb = self.get_embedding(frame, box)
                                if emb is not None:
                                    # Normalize and search
                                    emb = emb / (np.linalg.norm(emb) + 1e-6)
                                    global_id = None
                                    
                                    if self.index.ntotal > 0:
                                        D, I = self.index.search(emb.reshape(1, -1), 1)
                                        if D[0][0] > self.SIM_THRESHOLD:
                                            global_id = self.id_map[int(I[0][0])]

                                    if global_id is None:
                                        global_id = self.next_global_id
                                        self.next_global_id += 1
                                        self.id_map.append(global_id)
                                        self.index.add(emb.reshape(1, -1))
                                    
                                    self.trackid_to_global[t_id] = global_id
                                else:
                                    global_id = t_id # Fallback
                            else:
                                global_id = t_id # Pending buffer
                        else:
                            global_id = t_id # Occluded, don't fingerprint yet

                    current_detections.append({
                        "id": t_id, "global_id": global_id, "bbox": box,
                        "base_point": (int((x1 + x2) / 2), int(y2))
                    })
                    
                    color = self.COLOR_PALETTE[global_id % len(self.COLOR_PALETTE)].tolist()
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 3)
                    cv2.putText(frame, f"GID:{global_id}", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            yield frame, current_detections
