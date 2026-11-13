"""COCO detections and guideline review flags for the CVAT agent."""

from io import BytesIO
from math import ceil


COCO_LABELS = {
    "person": "pedestrian", "bicycle": "bicycle", "car": "car",
    "motorcycle": "motorcycle", "bus": "bus", "train": "train", "truck": "truck",
    "traffic light": "traffic light", "stop sign": "traffic sign",
}


def intersection(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def tile_starts(length, window):
    distance = length - window
    if not distance:
        return [0]
    steps = ceil(distance / (window * 0.75))
    return [round(distance * step / steps) for step in range(steps + 1)]


def merge_detections(detections):
    kept = []
    for name, score, points in sorted(detections, key=lambda item: item[1], reverse=True):
        area = intersection(points, points)
        if area <= 0:
            continue
        duplicate = False
        for other_name, _, other in kept:
            if other_name != name:
                continue
            shared = intersection(points, other)
            other_area = intersection(other, other)
            if shared / (area + other_area - shared) >= 0.5 or shared / min(area, other_area) >= 0.85:
                duplicate = True
                break
        if not duplicate:
            kept.append((name, score, points))
    return kept


def guideline_proposals(detections, width, height, confidence, class_names=COCO_LABELS):
    boxes, issues = [], []
    cycles = [p for name, score, p in detections if name in {"bicycle", "motorcycle"} and score >= confidence]

    def flag(kind, label, points, reason):
        location = f"{label} {tuple(round(p) for p in points)} px"
        issues.append({"type": kind, "location": location, "reason": reason})

    for i, (name, score, points) in enumerate(detections):
        label = class_names[name]
        x1, y1, x2, y2 = points
        if score < confidence:
            flag("LOW_VISIBILITY", label, points, f"Điểm YOLO {score:.2f}; kiểm tra có đối tượng thật không.")
            continue
        if name == "person" and label == "pedestrian":
            center = (x1 + x2) / 2
            if any(bike[0] <= center <= bike[2] and intersection(points, bike) > 0 for bike in cycles):
                flag("UNCERTAIN_CLASS", label, points, "Người chồng lên xe hai bánh; phân biệt pedestrian và rider.")
                continue

        normalized = [round(x1 * 1000 / width), round(y1 * 1000 / height),
                      round(x2 * 1000 / width), round(y2 * 1000 / height)]
        if normalized[0] >= normalized[2] or normalized[1] >= normalized[3]:
            flag("UNCERTAIN_BOUNDARY", label, points, "Hộp quá nhỏ để biểu diễn trên thang tọa độ của tool.")
            continue
        truncated = normalized[0] == 0 or normalized[1] == 0 or normalized[2] == 1000 or normalized[3] == 1000
        if truncated:
            flag("ATTRIBUTE_CHECK", label, points, "Hộp chạm mép ảnh; kiểm tra vật thể thực sự bị cắt.")
        area = (x2 - x1) * (y2 - y1)
        if any(j != i and other_score >= confidence and intersection(points, other) >= 0.2 * area
               for j, (_, other_score, other) in enumerate(detections)):
            flag("ATTRIBUTE_CHECK", label, points, "Hộp chồng lấn đáng kể; kiểm tra occluded và biên nhìn thấy.")
        boxes.append({"label": label, "bbox_2d": normalized, "score": round(score, 4),
                      "occluded": False, "truncated": truncated})
    return {"boxes": boxes, "issues": issues}


class YoloDetector:
    def __init__(self, weights, confidence=0.30, *, class_names=COCO_LABELS):
        from ultralytics import YOLO

        self.model = YOLO(str(weights), task="detect")
        self.confidence = confidence
        self.class_names = class_names

    def detect(self, image, size):
        result = self.model.predict(image, imgsz=size, device="cpu",
                                    conf=min(self.confidence, 0.15), verbose=False)[0]
        boxes = result.boxes
        return [(result.names[int(cls)], score, points) for cls, score, points in
                zip(boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist())
                if result.names[int(cls)] in self.class_names]

    def predict_image(self, image):
        width, height = image.size
        detections = self.detect(image, 960)
        tile_w, tile_h = min(768, width), min(512, height)
        for top in tile_starts(height, tile_h):
            for left in tile_starts(width, tile_w):
                if (tile_w, tile_h) == image.size:
                    continue
                right, bottom = left + tile_w, top + tile_h
                tile = image.crop((left, top, right, bottom))
                for name, score, (x1, y1, x2, y2) in self.detect(tile, 768):
                    if ((left and x1 <= 2) or (top and y1 <= 2)
                            or (right < width and x2 >= tile_w - 2)
                            or (bottom < height and y2 >= tile_h - 2)):
                        continue
                    detections.append((name, score, [x1 + left, y1 + top, x2 + left, y2 + top]))
        return guideline_proposals(merge_detections(detections), width, height, self.confidence, self.class_names)

    def predict(self, data, frame):
        from PIL import Image

        with Image.open(BytesIO(data)) as image:
            if image.size != (frame["width"], frame["height"]):
                raise ValueError("Kích thước ảnh tải về khác metadata của CVAT.")
            return self.predict_image(image.convert("RGB"))
