"""Build the CVAT review from the selected labels and local model results."""

from base64 import b64encode
from html import escape
from io import BytesIO
import json

import cv2
import numpy as np
from PIL import Image

from annotation_scope import bounds, ego_mask, filter_yolo
from cvat_bbox_agent import YOLO_WEIGHTS, ROOT
from drivable_area import AREA_LABELS, BOUNDARY_PROMPTS, DIVIDER_PROMPT, ROAD_PROMPT, driving_lanes
from extra_geometry import HumanPose, centerlines, decode_mask, encode_mask, joint_indices, skeleton_edges
from lane_markings import filter_double_white
from local_annotation_review import CLASS_IDS, MODEL, SAM3_MODEL, VOID, Segmenter
from polygon_geometry import outlines
from sam3_concepts import Sam3Concepts
from yolo_bbox import COCO_LABELS, YoloDetector, intersection


SAM3_BBOX = "sam3_bbox"
SAM3_MIXED = "sam3_mixed"
POSE_MODEL = ROOT / "models" / "yolo11n-pose.pt"


def make_review(images, output, destination, mode, guideline, prompts, emit, *,
                bbox_model=YOLO_WEIGHTS, sam3_model=SAM3_MODEL, pose_model=POSE_MODEL, labels=None,
                semantic_model=MODEL):
    names = tuple(prompts)
    polygon_names = tuple(guideline.segmentation) if mode == SAM3_MIXED else names
    polygon_ids = {prompt: index for index, prompt in enumerate(polygon_names)}
    vegetation_prompt = next((prompt for prompt, label in guideline.segmentation.items()
                              if label == "vegetation" and prompt in polygon_ids), None) if mode != "bbox" else None
    use_semantic_vegetation = bool(vegetation_prompt and (semantic_model / "config.json").is_file())
    area_prompts = {prompt: name for prompt, name in guideline.segmentation.items()
                    if name in AREA_LABELS} if mode == SAM3_MIXED else {}
    direct_prompt = next((prompt for prompt, name in area_prompts.items() if name == "area/direct"), None)
    alternative_prompt = next((prompt for prompt, name in area_prompts.items() if name == "area/alternative"), None)
    other_inputs = guideline.bbox.keys() | guideline.extra.get("mask", {}).keys() | guideline.extra.get("polyline", {}).keys()
    inference_prompts = dict.fromkeys(prompt for prompt in names if prompt not in area_prompts or prompt in other_inputs)
    if use_semantic_vegetation and vegetation_prompt not in other_inputs:
        inference_prompts.pop(vegetation_prompt, None)
    if area_prompts:
        inference_prompts.update(dict.fromkeys((ROAD_PROMPT, *BOUNDARY_PROMPTS)))
    if alternative_prompt:
        inference_prompts[DIVIDER_PROMPT] = None
    if mode == "bbox":
        emit("log", "Đang nạp model YOLO…")
        source_names = {name: COCO_LABELS.get(name, name) for name in guideline.bbox}
        detector = YoloDetector(bbox_model, guideline.bbox_min_score, class_names=source_names)
        source_for = {label: name for name, label in source_names.items()}
    elif inference_prompts:
        emit("log", "Đang nạp model SAM 3…")
        detector = Sam3Concepts(sam3_model)
    vegetation_model = None
    if use_semantic_vegetation:
        emit("log", "Đang nạp SegFormer cho Polygon vegetation…")
        vegetation_model = Segmenter(semantic_model)
    pose_labels = guideline.extra.get("skeleton", {}).values()
    joints = {name: joint_indices(labels[name]) for name in pose_labels}
    pose = HumanPose(pose_model) if joints else None
    frames, originals = [], {}
    for number, name, mime, data in images:
        emit("log", f"Đang xử lý frame CVAT {number}…")
        with Image.open(BytesIO(data)) as file:
            image = file.convert("RGB")
        width, height = image.size
        ignored = ego_mask(width, height, number, False)
        item = {"frame": number, "name": name, "width": width, "height": height,
                "boxes": [], "issues": [], "polygons": [], "extra_shapes": []}
        if mode == "bbox":
            prediction = detector.predict_image(image)
            boxes, _ = filter_yolo(prediction["boxes"], ignored)
            for box in boxes:
                points = bounds(box, width, height)
                if intersection(points, points) >= guideline.bbox_min_area_px:
                    item["boxes"].append({**box, "label": source_for[box["label"]]})
            item["issues"] = prediction["issues"]
        elif inference_prompts:
            road = np.zeros((height, width), dtype=bool) if area_prompts else None
            white = np.zeros_like(road) if area_prompts else None
            boundaries, divider_lines = [], []
            for prompt, masks in detector.predict(image, inference_prompts, guideline.mask_min_score):
                rectangle = prompt in guideline.bbox and mode in (SAM3_BBOX, SAM3_MIXED)
                polygon = (prompt in guideline.segmentation and prompt not in area_prompts
                           and not (use_semantic_vegetation and prompt == vegetation_prompt)
                           and mode in ("segmentation", SAM3_MIXED))
                mask_label = guideline.extra.get("mask", {}).get(prompt)
                line_label = guideline.extra.get("polyline", {}).get(prompt)
                for mask in masks:
                    if mask.shape != (height, width):
                        raise ValueError(f"SAM 3 trả về kích thước vùng khác ảnh ở frame {number}.")
                    region = (np.asarray(mask, dtype=bool) & ~ignored).astype(np.uint8)
                    ys, xs = np.nonzero(region)
                    if not xs.size:
                        continue
                    left, top = int(xs.min()), int(ys.min())
                    right, bottom = int(xs.max()) + 1, int(ys.max()) + 1
                    crop = region[top:bottom, left:right]
                    if area_prompts and prompt == ROAD_PROMPT:
                        road |= region != 0
                    if area_prompts and prompt == BOUNDARY_PROMPTS[0]:
                        white |= region != 0
                    geometry = area_prompts and (prompt in BOUNDARY_PROMPTS or (alternative_prompt and prompt == DIVIDER_PROMPT))
                    lines = list(centerlines(crop, left, top)) if line_label or geometry else []
                    if area_prompts and prompt in BOUNDARY_PROMPTS:
                        boundaries.extend(lines)
                    if alternative_prompt and prompt == DIVIDER_PROMPT:
                        divider_lines.extend(lines)
                    if rectangle and (right - left) * (bottom - top) >= guideline.bbox_min_area_px:
                        item["boxes"].append({"label": prompt, "points_px": [left, top, right, bottom],
                                              "occluded": False, "truncated": left == 0 or top == 0
                                              or right == width or bottom == height})
                    if polygon:
                        for points in outlines(crop, left, top):
                            item["polygons"].append({"class_id": polygon_ids[prompt], "points": points})
                    if mask_label:
                        item["extra_shapes"].append(dict(type="mask", label=mask_label, points=encode_mask(crop, left, top)))
                    if line_label:
                        item["extra_shapes"].extend(dict(type="polyline", label=line_label, points=points)
                                                    for points in lines)
                targets = [mapping[prompt] for enabled, mapping in
                           ((rectangle, guideline.bbox), (polygon, guideline.segmentation)) if enabled]
                targets += [target for target in (mask_label, line_label) if target]
                emit("log", f"Frame {number} · {', '.join(dict.fromkeys(targets)) or prompt} · {len(masks)} vùng SAM 3.")
            if area_prompts:
                direct, alternative = driving_lanes(road, boundaries, ignored, white, divider_lines)
            for prompt, lane, reason in (
                    (direct_prompt, direct if area_prompts else None, "Thiếu hai biên rõ cho làn hiện tại. Cần rà trên CVAT."),
                    (alternative_prompt, alternative if area_prompts else None, "Chưa đủ dấu hiệu của làn kề cùng chiều có thể chuyển vào. Cần rà trên CVAT.")):
                if not prompt:
                    continue
                if lane is None:
                    item["issues"].append({"location": area_prompts[prompt], "reason": reason})
                else:
                    item["polygons"].extend({"class_id": polygon_ids[prompt], "points": points}
                                            for points in outlines(lane, 0, 0))
        if vegetation_prompt and not any(p["class_id"] == polygon_ids[vegetation_prompt] for p in item["polygons"]):
            if vegetation_model is not None:
                vegetation = (vegetation_model.predict(image, max(0.55, guideline.mask_min_score))
                              == CLASS_IDS["vegetation"]) & ~ignored
                _, components, stats, _ = cv2.connectedComponentsWithStats(vegetation.astype(np.uint8), connectivity=4)
                for idx, (left, top, w, h, area) in enumerate(stats.tolist()[1:], 1):
                    if area < max(12, width * height / 30000):
                        continue
                    region = (components[top:top + h, left:left + w] == idx).astype(np.uint8)
                    item["polygons"].extend({"class_id": polygon_ids[vegetation_prompt], "points": points}
                                            for points in outlines(region, left, top))
            recovered = sum(p["class_id"] == polygon_ids[vegetation_prompt] for p in item["polygons"])
            if recovered:
                emit("log", f"Frame {number} · SegFormer bổ sung {recovered} Polygon vegetation.")
            else:
                item["issues"].append({"location": "vegetation", "reason": "Chưa có đề xuất Polygon đủ tin cậy."})
        if pose:
            for keypoints in pose.predict(image):
                for name, indices in joints.items():
                    elements = []
                    for label_id, index in indices.items():
                        x, y, confidence = keypoints[index]
                        visible = (confidence >= 0.5 and 0 <= x < width and 0 <= y < height
                                   and not ignored[int(y), int(x)])
                        elements.append(dict(label_id=label_id, points=[round(x, 2), round(y, 2)] if visible else [0, 0],
                                             outside=not visible))
                    if any(not point["outside"] for point in elements):
                        item["extra_shapes"].append(dict(type="skeleton", label=name, points=[], elements=elements))
        item["extra_shapes"], rejected = filter_double_white(item["extra_shapes"], width)
        if rejected:
            message = f"Đưa {rejected} đề xuất lane/double white vào review vì thiếu cặp song song rõ hoặc trùng vạch đơn."
            item["issues"].append({"location": "lane/double white", "reason": message})
            emit("log", f"Frame {number} · {message}")
        frames.append(item)
        originals[number] = data, mime
        emit("log", f"Frame {number} · {len(item['boxes'])} BBox · {len(item['polygons'])} Polygon · "
                    f"{len(item['extra_shapes'])} Polyline/Mask/Skeleton.")
        emit("step", None)

    output.mkdir(parents=True)
    (output / "review.json").write_text(json.dumps({"destination": destination,
        "class_ids": polygon_names, "void_id": VOID, "frames": frames,
        "customer_guideline": guideline.metadata()}, ensure_ascii=False, indent=2), encoding="utf-8")
    return frames, originals


def write_review_html(output, frames, originals, labels, shapes, mode, job_url, skipped, *, reference=None):
    titles = {"bbox": "YOLO BBox", SAM3_BBOX: "SAM 3 BBox", "segmentation": "SAM 3 Polygon",
              SAM3_MIXED: "SAM 3 và Pose"}
    title = titles[mode]
    by_id = {label["id"]: label for label in labels.values()}
    by_frame = {item["frame"]: [] for item in frames}
    for shape in shapes:
        by_frame[shape["frame"]].append(shape)
    sections = []
    for item in frames:
        number = item["frame"]
        data, mime = originals[number]
        width, height = item["width"], item["height"]
        picture = b64encode(data).decode("ascii")
        drawing = [f'<image width="{width}" height="{height}" href="data:{mime};base64,{picture}"/>']
        for shape in sorted(by_frame[number], key=lambda shape: shape["type"] == "rectangle"):
            label = by_id[shape["label_id"]]
            color = label.get("color", "#00ff86")
            if not isinstance(color, str) or len(color) != 7 or color[0] != "#" or any(
                    ch not in "0123456789abcdefABCDEF" for ch in color[1:]):
                color = "#00ff86"
            name = escape(label["name"])
            points = shape["points"]
            if shape["type"] == "rectangle":
                x1, y1, x2, y2 = points
                drawing.append(f'<rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" stroke="{color}"/>')
                drawing.append(f'<text x="{x1}" y="{max(14, y1-3)}" fill="{color}">{name}</text>')
            elif shape["type"] == "mask":
                mask, x, y = decode_mask(points, width, height)
                rgba = Image.new("RGBA", (mask.shape[1], mask.shape[0]), color)
                rgba.putalpha(Image.fromarray((mask * 120).astype(np.uint8)))
                png = BytesIO()
                rgba.save(png, format="PNG")
                encoded = b64encode(png.getvalue()).decode("ascii")
                drawing.append(f'<image x="{x}" y="{y}" width="{rgba.width}" height="{rgba.height}" '
                               f'href="data:image/png;base64,{encoded}"><title>{name}</title></image>')
            elif shape["type"] == "skeleton":
                visible = {element["label_id"]: element["points"] for element in shape["elements"] if not element["outside"]}
                for start, end in skeleton_edges(label):
                    if start in visible and end in visible:
                        x1, y1 = visible[start]
                        x2, y2 = visible[end]
                        drawing.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                                       f'stroke="{color}" stroke-width="2"/>')
                for element in shape["elements"]:
                    if not element["outside"]:
                        x, y = element["points"]
                        child = next(child for child in label["sublabels"] if child["id"] == element["label_id"])
                        drawing.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{color}" stroke="black">'
                                       f'<title>{name} · {escape(child["name"])}</title></circle>')
            elif shape["type"] == "polyline":
                coords = " ".join(f"{x:g},{y:g}" for x, y in zip(points[::2], points[1::2]))
                drawing.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2">'
                               f'<title>{name}</title></polyline>')
            else:
                coords = " ".join(f"{x:g},{y:g}" for x, y in zip(points[::2], points[1::2]))
                drawing.append(f'<polygon points="{coords}" fill="{color}" fill-opacity=".55"><title>{name}</title></polygon>')
        issues = "".join(f"<li>{escape(i['location'])} · {escape(i['reason'])}</li>" for i in item["issues"])
        sections.append(f'<section><h2>Frame {number} · {escape(item["name"])}</h2>'
                        f'<p>{len(by_frame[number])} shape sẵn sàng lưu</p>'
                        f'<svg viewBox="0 0 {width} {height}">{"".join(drawing)}</svg><ul>{issues}</ul></section>')
    notes = "".join(f"<li>{escape(note)}</li>" for note in skipped)
    guideline_html = ""
    if reference:
        guideline_html = (f'<p>Guideline: {escape(reference.name)}</p>'
                          '<details><summary>Đọc nội dung guideline để rà kết quả</summary>'
                          f'<pre style="white-space:pre-wrap">{escape(reference.text)}</pre></details>')
    html = ('<!doctype html><html lang="vi"><meta charset="utf-8"><title>Bản xem trước CVAT</title>'
            '<style>body{font:16px system-ui;max-width:1280px;margin:auto;padding:1rem}'
            'svg{width:100%;display:block}rect{fill:none;stroke-width:2}'
            'text{stroke:#000;stroke-width:3;paint-order:stroke;font:bold 16px system-ui}</style>'
            f'<h1>Bản xem trước {title}</h1><p><a href="{escape(job_url)}">Mở job CVAT</a> · '
            f'{len(shapes)} shape sẵn sàng lưu</p><ul>{notes}</ul>' + guideline_html
            + "\n".join(sections) + "</html>\n")
    (output / "index.html").write_text(html, encoding="utf-8")
