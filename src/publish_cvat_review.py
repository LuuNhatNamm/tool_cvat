"""Publish verified rectangles and segmentation polygons to CVAT."""

import argparse
from collections import Counter, defaultdict
import json
from math import isfinite
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from cvat_bbox_agent import (AgentError, CVAT, JOB_ID, LABELS, MAX_BATCH_IMAGES, ROOT, box_points,
                            check_assignee, overlap)
from extra_geometry import decode_mask, joint_indices
from local_annotation_review import PALETTE, VOID
from polygon_geometry import editable_polygons


def polygon_shapes(item, folder, labels, mapping=None, names=tuple(PALETTE)):
    number, width, height = (item[key] for key in ("frame", "width", "height"))
    mapping = mapping if mapping is not None else {name: name for name in names}
    allowed = {i for i, name in enumerate(names) if name in mapping}
    if "polygons" in item:
        regions = ((shape["class_id"], shape["points"]) for shape in item["polygons"])
    else:
        mask_path = folder / item["mask"]
        if mask_path.resolve().parent != folder.resolve():
            raise AgentError(f"Mask frame {number} nằm ngoài thư mục review.")
        with Image.open(mask_path) as image:
            if image.mode != "P" or image.size != (width, height):
                raise AgentError(f"Mask frame {number} sai chế độ hoặc kích thước.")
            ids = np.asarray(image)
        if any(i >= len(names) and i != VOID for i in np.unique(ids)):
            raise AgentError(f"Mask frame {number} chứa ID ngoài guideline.")
        regions = editable_polygons(ids, allowed)
    result = []
    for class_id, points in regions:
        if type(class_id) is not int or not 0 <= class_id < len(names):
            raise AgentError(f"Polygon frame {number} chứa ID ngoài guideline.")
        if class_id not in allowed:
            continue
        if (len(points) < 6 or len(points) % 2 or any(
                type(v) not in (int, float) or not isfinite(v) or not 0 <= v <= size
                for v, size in zip(points, (width, height) * (len(points) // 2)))):
            raise AgentError(f"Polygon frame {number} có tọa độ không hợp lệ.")
        result.append(dict(type="polygon", frame=number, label_id=labels[mapping[names[class_id]]]["id"],
                           points=points, source="auto", occluded=False, attributes=[]))
    return result


def label_map(cvat, mode="both", guideline=None):
    labels = {}
    bbox = set(guideline.bbox.values() if guideline else LABELS) if mode != "segmentation" else set()
    semantic = set(guideline.segmentation.values() if guideline else PALETTE) if mode != "bbox" else set()
    extras = guideline.extra if guideline else {}
    if not extras and ((mode != "segmentation" and not bbox) or (mode != "bbox" and not semantic)):
        raise AgentError("Guideline thiếu ánh xạ cho phạm vi lưu đã chọn.")
    required = bbox | semantic | {name for mapping in extras.values() for name in mapping.values()}
    for item in cvat.list("/api/labels", job_id=cvat.job_id):
        if item["name"] in required and not (item.get("parent") or item.get("parent_id")):
            if item["name"] in labels:
                raise AgentError(f"Nhãn {item['name']} bị trùng trong cấu hình job.")
            labels[item["name"]] = item
    if required - labels.keys():
        raise AgentError("Job thiếu nhãn: " + ", ".join(sorted(required - labels.keys())))
    for name, item in labels.items():
        accepted = {kind for kind, mapping in extras.items() if name in mapping.values()}
        if name in bbox:
            accepted.add("rectangle")
        if name in semantic:
            accepted.add("polygon")
        kind = item.get("type", "any")
        if (accepted != {kind} and (kind != "any" or "skeleton" in accepted)):
            raise AgentError(f"Kiểu shape của nhãn {name} chưa phù hợp.")
        if kind == "skeleton":
            joint_indices(item)
    return labels


def shape_key(shape):
    elements = tuple(sorted((element["label_id"], bool(element.get("outside")),
                             () if element.get("outside") else tuple(element["points"]))
                            for element in shape.get("elements", ())))
    return shape["frame"], shape["type"], shape["label_id"], tuple(shape.get("points") or ()), elements


def extra_shapes(item, labels, guideline):
    if not guideline or not guideline.extra:
        return []
    width, height, frame = (item[key] for key in ("width", "height", "frame"))
    allowed = {kind: set(mapping.values()) for kind, mapping in guideline.extra.items()}

    def valid_points(points, minimum):
        return (len(points) >= minimum and len(points) % 2 == 0 and all(
            type(value) in (float, int) and 0 <= value <= size
            for value, size in zip(points, (width, height) * (len(points) // 2))))

    result = []
    for candidate in item.get("extra_shapes", []):
        kind, name, points = (candidate[key] for key in ("type", "label", "points"))
        if name not in allowed.get(kind, ()):
            raise AgentError(f"Shape {name} ngoài lựa chọn đã xem trước.")
        label = labels[name]
        shape = dict(type=kind, label_id=label["id"], frame=frame, points=points, source="auto",
                     occluded=False, outside=False, attributes=[])
        if kind == "mask":
            decode_mask(points, width, height)
        elif kind == "polyline" and (not valid_points(points, 4) or len(set(zip(points[::2], points[1::2]))) < 2):
            raise AgentError(f"Polyline của nhãn {name} có tọa độ không hợp lệ.")
        elif kind == "skeleton":
            elements = candidate["elements"]
            if points or Counter(p["label_id"] for p in elements) != Counter(joint_indices(label).keys()):
                raise AgentError(f"Skeleton của nhãn {name} không khớp cấu trúc CVAT.")
            shape["elements"] = []
            for element in elements:
                coords = element["points"]
                if len(coords) != 2 or not valid_points(coords, 2) or type(element["outside"]) is not bool:
                    raise AgentError(f"Điểm Skeleton của nhãn {name} không hợp lệ.")
                shape["elements"].append(dict(type="points", frame=frame, source="auto", occluded=False,
                                              attributes=[], **element))
        result.append(shape)
    return result


def rectangle_points(box, item):
    if "points_px" not in box:
        return box_points(box, item)
    points = box["points_px"]
    if len(points) != 4 or any(type(value) is not int for value in points):
        raise AgentError("Rectangle SAM 3 cần bốn tọa độ pixel nguyên.")
    x1, y1, x2, y2 = points
    width, height = item["width"], item["height"]
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise AgentError("Tọa độ Rectangle SAM 3 nằm ngoài ảnh hoặc có diện tích bằng 0.")
    touches_edge = x1 == 0 or y1 == 0 or x2 == width or y2 == height
    if type(box["occluded"]) is not bool or box["truncated"] is not touches_edge:
        raise AgentError("Thuộc tính Rectangle SAM 3 không khớp hình học của hộp.")
    return points


def prepare(folder, frames, labels, existing, mode="both", destination=None, guideline=None, *,
            polygon_names=tuple(PALETTE), allow_unmarked_truncation=False):
    report = json.loads((folder / "review.json").read_text(encoding="utf-8"))
    if destination is not None and report.get("destination") != destination:
        raise AgentError("Bản review thuộc job khác.")
    if report.get("customer_guideline") != (guideline.metadata() if guideline else None):
        raise AgentError("Bản review dùng guideline khác với cấu hình đang chạy.")
    if report["class_ids"] != list(polygon_names) or report["void_id"] != VOID:
        raise AgentError("Taxonomy của file review không khớp guideline.")
    items = {f["frame"]: f for f in report["frames"]}
    if len(items) != len(report["frames"]) or items.keys() != frames.keys():
        raise AgentError("Bản review không khớp phạm vi frame trên CVAT.")
    if existing["tracks"] and (mode != "segmentation" or (guideline and guideline.extra)):
        raise AgentError("Job có track. Cần đối chiếu hộp nội suy trước khi ghi.")
    existing_by_frame = defaultdict(list)
    for shape in existing["shapes"]:
        existing_by_frame[shape["frame"]].append(shape)
    semantic_names = guideline.segmentation.values() if guideline else PALETTE
    semantic_ids = {labels[name]["id"] for name in semantic_names} if mode != "bbox" else set()
    to_add, to_update, notes = [], [], []
    for number, item in items.items():
        if any(item[k] != frames[number][k] for k in ("width", "height", "name")):
            raise AgentError(f"Frame {number} không khớp ảnh đã review.")
        current = existing_by_frame[number]
        previous = {shape_key(shape) for shape in current}
        proposed = extra_shapes(item, labels, guideline)
        proposed_keys = {shape_key(shape) for shape in proposed}
        conflicting = {(shape["type"], shape["label_id"]) for shape in current if shape_key(shape) not in proposed_keys}
        for shape in proposed:
            key = shape_key(shape)
            if key in previous:
                continue
            if (shape["type"], shape["label_id"]) in conflicting:
                notes.append((number, "skip_existing_extra"))
            else:
                to_add.append(shape)
        if semantic_ids:
            current_polygons = {shape_key(s) for s in current if s["type"] == "polygon" and s["label_id"] in semantic_ids}
            if any(s["type"] == "mask" and shape_key(s) not in proposed_keys for s in current):
                notes.append((number, "skip_existing_segmentation"))
            else:
                polygons = polygon_shapes(item, folder, labels, guideline.segmentation if guideline else None, polygon_names)
                if current_polygons - {shape_key(s) for s in polygons}:
                    notes.append((number, "skip_existing_segmentation"))
                else:
                    to_add.extend(s for s in polygons if shape_key(s) not in current_polygons)
        if mode == "segmentation":
            continue
        rectangles = [s for s in current if s["type"] == "rectangle" and len(s["points"]) == 4 and not s.get("outside")]
        for box in item["boxes"]:
            if guideline and box["label"] not in guideline.bbox:
                continue
            label = labels[guideline.bbox[box["label"]] if guideline else box["label"]]
            points = rectangle_points(box, item)
            hits = [s for s in rectangles if overlap(points, s["points"]) >= 0.55]
            if any(s["label_id"] == label["id"] for s in hits):
                continue
            if hits:
                old = hits[0]
                if (not guideline and len(hits) == 1 and box.get("reviewed_correction") == "van_to_car"
                        and old["label_id"] == labels["truck"]["id"] and old.get("source") == "auto"
                        and not old.get("attributes") and overlap(points, old["points"]) >= 0.85):
                    to_update.append({**old, "label_id": label["id"]})
                else:
                    notes.append((number, "skip_cross_label_bbox"))
                continue
            attrs = [a for a in label.get("attributes", []) if a["name"] == "truncated" and a["input_type"] == "checkbox"]
            if box["truncated"] and not attrs and not allow_unmarked_truncation:
                notes.append((number, "skip_missing_truncated"))
                continue
            shape = dict(type="rectangle", frame=number, label_id=label["id"], points=points, source="auto",
                         occluded=box["occluded"], attributes=[{"spec_id": a["id"], "value": str(box["truncated"]).lower()} for a in attrs])
            to_add.append(shape)
            rectangles.append(shape)
    return to_add, to_update, notes


def publish(cvat, endpoint, existing, add, update, *, progress=print):
    version = existing["version"]
    confirmed = {}
    batches = [("update", update)] if update else []
    batches.extend(("create", add[i:i + 300]) for i in range(0, len(add), 300))
    for action, batch in batches:
        response = cvat.api(endpoint, method="PATCH", params={"action": action},
                            payload={"version": version, "shapes": batch, "tags": [], "tracks": []})
        returned = response["shapes"]
        if Counter(map(shape_key, returned)) != Counter(map(shape_key, batch)):
            raise AgentError("CVAT trả về kết quả khác bản ghi. Đọc lại job trước khi thử lại.")
        confirmed.update({s["id"]: shape_key(s) for s in returned})
        version = response.get("version", version)
        progress(f"CVAT xác nhận {action} {len(returned)} shape.")
    after = cvat.api(endpoint) if confirmed else existing
    server_keys = {s["id"]: shape_key(s) for s in after["shapes"]}
    if any(server_keys.get(i) != key for i, key in confirmed.items()):
        raise AgentError("Kết quả đọc lại chưa khớp các shape vừa ghi.")
    return dict(Counter(s["type"] for s in after["shapes"]))


def main():
    for stream in (sys.stdout, sys.stderr):
        if not stream.isatty() and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, default=ROOT / "semantic-review")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    try:
        cvat = CVAT()
        job = cvat.job()
        check_assignee(cvat, job)
        all_frames = cvat.frames(job)
        report = json.loads((args.review / "review.json").read_text(encoding="utf-8"))
        selected = [f["frame"] for f in report["frames"]]
        if not 1 <= len(selected) <= MAX_BATCH_IMAGES or any(n not in all_frames for n in selected):
            raise AgentError("Bản review cần chứa từ 1 đến 10 ảnh thuộc job đã cấu hình.")
        frames = {n: all_frames[n] for n in selected}
        labels = label_map(cvat)
        endpoint = f"/api/jobs/{JOB_ID}/annotations/"
        existing = cvat.api(endpoint)
        add, update, notes = prepare(args.review, frames, labels, existing)
        print(f"Job {JOB_ID}: thêm {dict(Counter(s['type'] for s in add))}, đổi nhãn {len(update)}. "
              f"Cần rà: {dict(Counter(reason for _, reason in notes))}.", flush=True)
        if notes:
            print("Frame cần rà:", notes, flush=True)
        if not args.apply:
            return 0
        totals = publish(cvat, endpoint, existing, add, update, progress=lambda line: print(line, flush=True))
        print("Đã đọc lại CVAT và xác nhận:", totals, flush=True)
    except (AgentError, OSError, ValueError, TypeError, KeyError) as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
