"""Scope checks for BBox proposals."""

from yolo_bbox import intersection


MIN_BOX_AREA = 150
VEHICLES = {"car", "truck", "bus", "train", "motorcycle", "bicycle"}
LARGE_VEHICLES = {"truck", "bus", "train"}
ROAD_VEHICLES = {"car", "truck", "bus"}

def reviewed_corrections(boxes, data, ignored):
    return []


def reviewed_person(data, width, height):
    return None


def ego_mask(width, height, frame, camera_profile):
    import numpy as np
    return np.zeros((height, width), dtype=bool)


def bounds(box, width, height):
    return [n * size / 1000 for n, size in zip(box["bbox_2d"], (width, height) * 2)]


def reject_reason(label, points, ignored):
    height, width = ignored.shape
    x1, y1, x2, y2 = points
    area = intersection(points, points)
    x = max(0, min(width - 1, int((x1 + x2) / 2)))
    y = max(0, min(height - 1, int((y1 + y2) / 2)))
    if ignored[y, x]:
        return "ego_vehicle"
    if area < MIN_BOX_AREA:
        return "tiny"
    ratio = area / (width * height)
    if label in LARGE_VEHICLES and (ratio > 0.25 or (ratio > 0.12 and y1 <= height * 0.025)):
        return "oversize_heavy_vehicle"
    return None


def filter_yolo(boxes, ignored, image_data=None):
    selected, rejected = [], []
    for box in boxes:
        points = bounds(box, width, height)
        reason = reject_reason(box["label"], points, ignored)
        if reason:
            rejected.append({**box, "reason": reason})
        else:
            selected.append((box, points))

    final = []
    for box, points in selected:
        if box["label"] in ROAD_VEHICLES:
            area = intersection(points, points)
            nested = [(other_points, intersection(other_points, other_points)) for other, other_points in selected
                      if other is not box and other["label"] in ROAD_VEHICLES]
            nested = [(p, a) for p, a in nested if area >= a * 1.35 and intersection(points, p) >= a * 0.80]
            if any(intersection(a, b) <= 0.15 * min(area_a, area_b)
                   for i, (a, area_a) in enumerate(nested) for b, area_b in nested[i + 1:]):
                rejected.append({**box, "reason": "possible_merged_vehicles"})
                continue
        final.append(box)
    return final, rejected


def vehicle_wheel_region(points, vehicles):
    for x1, y1, x2, y2 in vehicles:
        lower_region = [x1, y1 + (y2 - y1) * 0.68, x2, y2 + (y2 - y1) * 0.20]
        if intersection(points, lower_region) >= 0.65 * intersection(points, points):
            return True
    return False
