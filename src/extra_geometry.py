"""Convert detected regions and human keypoints to CVAT geometry."""

import re
from xml.etree import ElementTree

import cv2
import numpy as np


def encode_mask(region, left, top):
    flat = region.astype(bool, copy=False).ravel()
    ends = np.r_[0, np.flatnonzero(flat[1:] != flat[:-1]) + 1, flat.size]
    runs = np.diff(ends).tolist()
    if flat[0]:
        runs.insert(0, 0)
    height, width = region.shape
    return runs + [left, top, left + width - 1, top + height - 1]


def decode_mask(points, width, height):
    if len(points) < 5 or any(type(v) not in (int, float) or not 0 <= v <= width * height or int(v) != v for v in points):
        raise ValueError("Mask CVAT có tọa độ hoặc RLE không hợp lệ.")
    *runs, left, top, right, bottom = map(int, points)
    if (not (0 <= left <= right < width and 0 <= top <= bottom < height)
            or sum(runs) != (right - left + 1) * (bottom - top + 1) or not any(runs[1::2])):
        raise ValueError("Mask CVAT không khớp khung pixel.")
    return np.repeat(np.arange(len(runs)) % 2, runs).reshape(bottom - top + 1, right - left + 1), left, top


def thin(region):
    image = np.pad(region.astype(bool), 1)
    center = image[1:-1, 1:-1]
    neighbors = (image[:-2, 1:-1], image[:-2, 2:], image[1:-1, 2:], image[2:, 2:],
                 image[2:, 1:-1], image[2:, :-2], image[1:-1, :-2], image[:-2, :-2])
    while True:
        changed = False
        for first in (True, False):
            count = sum(neighbors, start=np.zeros(center.shape, dtype=np.uint8))
            transitions = sum((~a & b for a, b in zip(neighbors, (*neighbors[1:], neighbors[0]))),
                              start=np.zeros(center.shape, dtype=np.uint8))
            n, _, e, _, s, _, w, _ = neighbors
            blocked = (n & e & s) | (e & s & w) if first else (n & e & w) | (n & s & w)
            remove = center & (count >= 2) & (count <= 6) & (transitions == 1) & ~blocked
            if remove.any():
                center[remove] = False
                changed = True
        if not changed:
            return center


def centerlines(region, left, top):
    region = region.astype(bool, copy=False)
    ys, xs = np.nonzero(thin(region))
    pixels = set(zip(xs.tolist(), ys.tolist()))
    graph = {}
    for x, y in sorted(pixels):
        graph[x, y] = [(x + dx, y + dy) for dx, dy in ((-1, 0), (0, -1), (0, 1), (1, 0),
                       (-1, -1), (-1, 1), (1, -1), (1, 1))
                       if (x + dx, y + dy) in pixels and (not dx or not dy or
                          ((x + dx, y) not in pixels and (x, y + dy) not in pixels))]
    visited = set()
    for start in sorted(graph, key=lambda pixel: len(graph[pixel]) == 2):
        for end in graph[start]:
            if (start, end) in visited:
                continue
            path = [start]
            prev = start
            while (prev, end) not in visited:
                visited.update(((prev, end), (end, prev)))
                if end == start:
                    break
                path.append(end)
                if len(graph[end]) != 2:
                    break
                nxt = next(p for p in graph[end] if p != prev)
                prev, end = end, nxt
            if len(path) < 4:
                continue
            raw = np.asarray(path, dtype=np.int32)
            simpler = cv2.approxPolyDP(raw, 1.5, False).reshape(-1, 2)
            x, y, width, height = cv2.boundingRect(raw)
            raster = np.zeros((height, width), dtype=np.uint8)
            cv2.polylines(raster, [simpler - (x, y)], False, 1)
            points = raw if np.any(raster & ~region[y:y+height, x:x+width]) else simpler
            yield (points + (left, top)).ravel().tolist()


COCO_JOINTS = ("nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
               "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
               "left_knee", "right_knee", "left_ankle", "right_ankle")


def joint_indices(label):
    children = label.get("sublabels") or []
    indices = {}
    for child in children:
        name = re.sub(r"[\s-]+", "_", child["name"].strip().lower())
        if name not in COCO_JOINTS or child.get("type", "points") != "points":
            raise ValueError(f"Skeleton {label['name']} có điểm {child['name']} ngoài bộ COCO Pose.")
        indices[child["id"]] = COCO_JOINTS.index(name)
    if not indices or len(set(indices.values())) != len(children):
        raise ValueError(f"Skeleton {label['name']} cần các điểm COCO Pose có tên riêng trên CVAT.")
    return indices


def skeleton_edges(label):
    try:
        root = ElementTree.fromstring(f"<root>{label.get('svg') or ''}</root>")
    except ElementTree.ParseError:
        return []
    nodes = {node.get("data-node-id"): node.get("data-label-id") for node in root.iter()
             if node.get("data-node-id") and node.get("data-label-id")}
    edges = []
    for node in root.iter():
        start = nodes.get(node.get("data-node-from"), "")
        end = nodes.get(node.get("data-node-to"), "")
        if start.isdecimal() and end.isdecimal():
            edges.append((int(start), int(end)))
    return edges


class HumanPose:
    def __init__(self, weights):
        if not weights.is_file():
            raise ValueError(f"Thiếu model pose tại {weights}.")
        from ultralytics import YOLO
        self.model = YOLO(str(weights), task="pose")

    def predict(self, image):
        result = self.model.predict(image, imgsz=960, device="cpu", verbose=False)[0]
        if result.keypoints is not None:
            yield from result.keypoints.data.cpu().tolist()
