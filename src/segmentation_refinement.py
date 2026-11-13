"""Refine segmentation under glare and apply verified boundaries to example frames."""

from hashlib import sha256


def night_exposure(image):
    import cv2
    import numpy as np
    from PIL import Image

    lab = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2LAB)
    light = lab[:, :, 0]
    if np.median(light) >= 78:
        return None
    smoothed = cv2.bilateralFilter(light, 5, 9, 3)
    values = np.arange(256) / 255
    # Lift shadows but limit the boost in already bright headlights and reflections.
    adjusted = values + 0.20 * np.sqrt(values) * (1 - values) ** 2
    lab[:, :, 0] = cv2.LUT(smoothed, np.rint(adjusted * 255).astype(np.uint8))
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


def deskew(image, source_hash):
    return None


def restore_deskewed(mask, rotated, inverse, margin, ignored, class_ids, void):
    import cv2
    import numpy as np

    height, width = mask.shape
    restored = cv2.warpAffine(rotated, inverse, (width, height), flags=cv2.INTER_NEAREST, borderValue=void)
    supported = np.isin(restored, [class_ids[name] for name in ("road", "building", "wall", "pole")])
    candidate = supported & ~ignored & (restored != mask) & ((mask == void) | (margin < 0.22))
    mask[candidate] = restored[candidate]
    return int(np.count_nonzero(candidate))


def fill_vehicle_mask_holes(mask, boxes, ignored, class_ids, void):
    import cv2
    import numpy as np

    height, width = mask.shape
    count = 0
    for box in boxes:
        label = box["label"]
        if label not in {"car", "truck", "bus", "train"}:
            continue
        x1, y1, x2, y2 = [round(v * s / 1000) for v, s in zip(box["bbox_2d"], (width, height) * 2)]
        crop = mask[y1:y2, x1:x2]
        if not crop.size:
            continue
        _, components, stats, _ = cv2.connectedComponentsWithStats((crop != class_ids[label]).astype(np.uint8))
        for idx, (x, y, w, h, area) in enumerate(stats[1:], 1):
            if x == 0 or y == 0 or x + w == crop.shape[1] or y + h == crop.shape[0] or area > 0.025 * crop.size:
                continue
            region = components == idx
            if np.all(crop[region] == void) and not np.any(ignored[y1:y2, x1:x2][region]):
                crop[region] = class_ids[label]
                count += int(area)
    return count


def reviewed_surfaces(mask, source_hash, ignored, class_ids):
    return {}


def frame_fingerprint(data):
    return sha256(data).hexdigest()
