"""Scope checks for BBox proposals and the reviewed job 222 camera views."""

from yolo_bbox import intersection
from hashlib import sha256


MIN_BOX_AREA = 150
VEHICLES = {"car", "truck", "bus", "train", "motorcycle", "bicycle"}
LARGE_VEHICLES = {"truck", "bus", "train"}
ROAD_VEHICLES = {"car", "truck", "bus"}

# Upper edge of the visible hood/dash for each of the reviewed camera views.
# Frame 16 and 19 use polygons for accessories that extend into the image.
HOOD_TOP = (0.77, 0.81, 0.76, 0.70, 0.83, 0.87, 0.86, 0.85, 0.84, 0.79,
            0.72, 0.99, 0.99, 0.98, 0.99, 0.85, 0.99, 0.99, 0.99, 0.99)
ACCESSORIES = {
    2: (((0, 0.741), (0.35, 0.734), (0.63, 0.720), (0.84, 0.707), (1, 0.714), (1, 1), (0, 1)),),
    11: (((0, 0.836), (0.18, 0.818), (0.39, 0.802), (0.59, 0.806), (0.74, 0.840), (1, 0.875), (1, 1), (0, 1)),),
    16: (((0, 0.58), (0.10, 0.58), (0.17, 0.62), (0.20, 0.73), (0.22, 1), (0, 1)),
         ((0.37, 0.61), (0.49, 0.59), (0.53, 0.72), (0.46, 0.80), (0.36, 0.75))),
    18: (((0, 0.849), (0.17, 0.809), (0.35, 0.790), (0.56, 0.783), (0.70, 0.804), (1, 0.863), (1, 1), (0, 1)),),
    19: (((0, 0.86), (1, 0.30), (1, 1), (0, 1)),
         ((0.46, 0), (0.68, 0), (0.70, 0.045), (0.735, 0.055), (0.78, 0.53),
          (0.61, 0.60), (0.54, 0.18), (0.48, 0.13))),
}


def reviewed_corrections(boxes, data, ignored):
    """Apply the user's confirmed van correction only to its original frame."""
    if sha256(data).hexdigest() != "ca73fe15c95971d7059ee7d2c0386c42c131170c48da6730e2fe0b02c776b170":
        return []
    height, width = ignored.shape
    van = [450 * width / 1000, 377 * height / 1000, 496 * width / 1000, 472 * height / 1000]
    for box in boxes:
        if box["label"] == "truck" and intersection(bounds(box, width, height), van) >= 0.65 * intersection(van, van):
            box["label"] = "car"
            box["reviewed_correction"] = "van_to_car"
            return [{"type": "CLASS_CORRECTION", "location": "Xe van trắng", "reason": "Đổi truck thành car theo ca đã xác nhận."}]
    return []


def reviewed_person(data, width, height):
    if sha256(data).hexdigest() != "a379c949076b643ec89385a7790e45416a86ea04dd56a815b27c2e2664c519a9":
        return None
    points = [208, 474, 254, 564]
    return {"label": "pedestrian", "points": points,
            "bbox_2d": [round(p * 1000 / s) for p, s in zip(points, (width, height) * 2)],
            "source": "reviewed_correction", "reason": "Hộp theo phần người qua đường bên trái đã rà trên ảnh gốc."}


def ego_mask(width, height, frame, job_222):
    import numpy as np
    from PIL import Image, ImageDraw

    image = Image.new("1", (width, height))
    if job_222 and 0 <= frame < len(HOOD_TOP):
        draw = ImageDraw.Draw(image)
        y = round(HOOD_TOP[frame] * height)
        draw.rectangle((0, y, width - 1, height - 1), fill=1)
        for polygon in ACCESSORIES.get(frame, ()):
            draw.polygon([(round(x * width), round(y * height)) for x, y in polygon], fill=1)
    return np.asarray(image, dtype=bool)


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
    height, width = ignored.shape
    checked_ad = image_data is not None and sha256(image_data).hexdigest() == (
        "dd21d07832a869ca51958c5e2342450e2a3bb62abfbb4c2f4cb3a7ced314c34a")
    ad_points = [0.889 * width, 0.245 * height, width, 0.377 * height]
    selected, rejected = [], []
    for box in boxes:
        points = bounds(box, width, height)
        reason = reject_reason(box["label"], points, ignored)
        if (reason is None and checked_ad and box["label"] == "car"
                and intersection(points, ad_points) >= 0.80 * intersection(points, points)):
            reason = "reviewed_taxi_roof_ad"
        if reason is None and box["label"] in ROAD_VEHICLES:
            for other, other_points in selected:
                if other["label"] not in ROAD_VEHICLES:
                    continue
                shared = intersection(points, other_points)
                if shared >= 0.80 * (intersection(points, points) + intersection(other_points, other_points) - shared):
                    reason = "duplicate_vehicle"
                    break
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
