"""Luminance correction and conservative repair of shadowed road masks."""

def relight_pavement(image):
    import cv2
    import numpy as np
    from PIL import Image

    lab = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2LAB)
    luminance = lab[:, :, 0].astype(np.float32)
    pavement = luminance[int(image.height * 0.42):int(image.height * 0.74),
                         int(image.width * 0.10):int(image.width * 0.90)]
    if not pavement.size:
        return None
    low, middle, high = np.percentile(pavement, (15, 50, 85))
    # On uniformly dark scenes, lifting the signal mainly amplifies sensor noise.
    if middle < 64 or high - low < 42:
        return None
    scale = max(2, min(image.size) / 34)
    illumination = cv2.GaussianBlur(luminance, (0, 0), scale)
    target = np.clip(np.percentile(pavement, 68), 115, 180)
    gain = np.clip((target / np.maximum(illumination, 20)) ** 0.65, 0.80, 1.55)
    lab[:, :, 0] = np.rint(np.clip(luminance * gain, 0, 255)).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)), gain >= 1.08


def repair_shadowed_road(mask, relit_road, lifted, top_ids, margin, ignored, object_boxes, class_ids, void):
    import numpy as np

    height, width = mask.shape
    road = class_ids["road"]
    current_road = mask == road
    positions = np.arange(width, dtype=np.int32)[None, :]
    left = np.maximum.accumulate(np.where(current_road, positions, -width), axis=1)
    right = np.minimum.accumulate(np.where(current_road, positions, 2 * width)[:, ::-1], axis=1)[:, ::-1]
    between_road = (left < positions) & (right > positions) & ((right - left) <= round(0.34 * width))
    stable_edge = np.isin(mask, [class_ids[name] for name in ("wall", "fence", "pole", "sidewalk")]) & (margin >= 0.24)
    road_support = (mask == void) | (top_ids[0] == road) | (top_ids[1] == road)
    candidate = (relit_road >= 0.72) & lifted & between_road & road_support & ~stable_edge & ~ignored
    candidate[:round(height * 0.38)] = False
    for box in object_boxes:
        x1, y1, x2, y2 = [round(value * size / 1000)
                          for value, size in zip(box["bbox_2d"], (width, height) * 2)]
        candidate[y1:y2, x1:x2] = False
    mask[candidate] = road
    return int(np.count_nonzero(candidate))


def reviewed_frame2(mask, source, ignored, class_ids):
    return {"road": 0, "wall": 0}
