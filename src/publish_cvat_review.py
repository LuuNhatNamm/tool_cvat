"""Publish the verified local BBox and semantic masks to the assigned CVAT job."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from cvat_bbox_agent import AgentError, CVAT, JOB_ID, LABELS, ROOT, box_points, check_assignee, overlap
from local_annotation_review import PALETTE, VOID


def mask_points(binary):
    ys, xs = np.nonzero(binary)
    if not len(xs):
        return []
    left, top, right, bottom = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    pixels = binary[top:bottom + 1, left:right + 1].ravel(order="C")
    boundaries = np.flatnonzero(pixels[1:] != pixels[:-1]) + 1
    runs = np.diff(np.concatenate(([0], boundaries, [pixels.size]))).tolist()
    if pixels[0]:
        runs.insert(0, 0)
    return runs + [left, top, right, bottom]


def mask_shapes(item, folder, labels, mapping=None):
    number, width, height = (item[key] for key in ("frame", "width", "height"))
    mask_path = folder / item["mask"]
    if mask_path.parent.resolve() != folder.resolve():
        raise AgentError(f"Mask frame {number} nằm ngoài thư mục review.")
    with Image.open(mask_path) as image:
        if image.mode != "P" or image.size != (width, height):
            raise AgentError(f"Mask frame {number} sai chế độ hoặc kích thước.")
        ids = np.asarray(image)
    classes = np.unique(ids)
    if any(i >= len(PALETTE) and i != VOID for i in classes):
        raise AgentError(f"Mask frame {number} chứa ID ngoài guideline.")
    names = list(PALETTE)
    mapping = mapping if mapping is not None else {name: name for name in names}
    return [dict(type="mask", frame=number, label_id=labels[mapping[names[i]]]["id"],
                 points=mask_points(ids == i), source="auto", occluded=False, attributes=[])
            for i in classes if i != VOID and names[i] in mapping]


def label_map(cvat, mode="both", guideline=None):
    labels = {}
    bbox = set(guideline.bbox.values() if guideline else LABELS) if mode != "segmentation" else set()
    semantic = set(guideline.segmentation.values() if guideline else PALETTE) if mode != "bbox" else set()
    if (mode != "segmentation" and not bbox) or (mode != "bbox" and not semantic):
        raise AgentError("Guideline thiếu ánh xạ cho phạm vi lưu đã chọn.")
    required = bbox | semantic
    for item in cvat.list("/api/labels", job_id=cvat.job_id):
        if item["name"] in required:
            if item["name"] in labels:
                raise AgentError(f"Nhãn {item['name']} bị trùng trong cấu hình job.")
            labels[item["name"]] = item
    if required - labels.keys():
        raise AgentError("Job thiếu nhãn: " + ", ".join(sorted(required - labels.keys())))
    for name, item in labels.items():
        accepted = {"any"}
        if name in bbox:
            accepted.add("rectangle")
        if name in semantic:
            accepted.add("mask")
        if item.get("type", "any") not in accepted or (name in bbox and name in semantic and item.get("type", "any") != "any"):
            raise AgentError(f"Kiểu shape của nhãn {name} chưa phù hợp.")
    return labels


def shape_key(shape):
    return shape["frame"], shape["type"], shape["label_id"], tuple(shape["points"])


def prepare(folder, frames, labels, existing, mode="both", destination=None, guideline=None):
    report = json.loads((folder / "review.json").read_text(encoding="utf-8"))
    if destination is not None and report.get("destination") != destination:
        raise AgentError("Bản review thuộc job khác.")
    if report.get("customer_guideline") != (guideline.metadata() if guideline else None):
        raise AgentError("Bản review dùng guideline khác với cấu hình đang chạy.")
    if report["class_ids"] != list(PALETTE) or report["void_id"] != VOID:
        raise AgentError("Taxonomy của file review không khớp guideline.")
    items = {f["frame"]: f for f in report["frames"]}
    if len(items) != len(report["frames"]) or items.keys() != frames.keys():
        raise AgentError("Bản review không khớp phạm vi frame trên CVAT.")
    if existing["tracks"] and mode != "segmentation":
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
        if semantic_ids:
            masks = mask_shapes(item, folder, labels, guideline.segmentation if guideline else None)
            mask_keys = {shape_key(s) for s in masks}
            current_masks = {shape_key(s) for s in current if s["type"] == "mask"}
            has_semantic_polygons = any(s["type"] == "polygon" and s["label_id"] in semantic_ids for s in current)
            if current_masks - mask_keys or has_semantic_polygons:
                notes.append((number, "skip_existing_segmentation"))
            else:
                to_add.extend(s for s in masks if shape_key(s) not in current_masks)
        if mode == "segmentation":
            continue
        rectangles = [s for s in current if s["type"] == "rectangle" and len(s["points"]) == 4 and not s.get("outside")]
        for box in item["boxes"]:
            if guideline and box["label"] not in guideline.bbox:
                continue
            label = labels[guideline.bbox[box["label"]] if guideline else box["label"]]
            points = box_points(box, item)
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
            if box["truncated"] and not attrs:
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
    batches.extend(("create", add[i:i + 30]) for i in range(0, len(add), 30))
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
    parser.add_argument("--review", type=Path, default=ROOT / "review-job-222-seg-refined")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    try:
        cvat = CVAT()
        job = cvat.job()
        check_assignee(cvat, job)
        frames = cvat.frames(job)
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
