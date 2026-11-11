"""Review tiled YOLO detections and Cityscapes semantic masks locally."""

import argparse
import base64
from html import escape
from io import BytesIO
import json
from pathlib import Path
import re
import sys

from cvat_bbox_agent import ROOT, YOLO_WEIGHTS, frame_preview
from annotation_scope import (VEHICLES, bounds, ego_mask, filter_yolo, reject_reason,
                              reviewed_corrections, reviewed_person, vehicle_wheel_region)
from shadow_segmentation import relight_pavement, repair_shadowed_road, reviewed_frame2
from segmentation_refinement import (deskew, fill_vehicle_mask_holes, frame_fingerprint,
                                     night_exposure, restore_deskewed, reviewed_surfaces)
from yolo_bbox import YoloDetector, intersection, tile_starts


MODEL = ROOT / "models" / "segformer-b0-cityscapes"
PALETTE = {
    "road": "804080", "sidewalk": "F423E8", "building": "464646", "wall": "66669C",
    "fence": "BE9999", "pole": "999999", "traffic_light": "FAAA1E", "traffic_sign": "DCDC00",
    "vegetation": "6B8E23", "terrain": "98FB98", "sky": "4682B4", "person": "DC143C",
    "rider": "FF0000", "car": "00008E", "truck": "000046", "bus": "003C64",
    "train": "005064", "motorcycle": "0000E6", "bicycle": "770B20",
}
VOID = 255
BBOX_CUES = {"person": "pedestrian", "rider": "rider", "traffic_light": "traffic light",
             "traffic_sign": "traffic sign"}
PRIORITY_FRAMES = (2, 4, 6, 11)
STRUCTURAL_CLASSES = {name: list(PALETTE).index(name) for name in ("building", "pole")}
CLASS_IDS = {name: i for i, name in enumerate(PALETTE)}


def exposure_views(image):
    import cv2
    import numpy as np
    from PIL import Image

    rgb = np.asarray(image)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    luminance = lab[:, :, 0]
    enhanced = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8)).apply(luminance)
    lab[:, :, 0] = cv2.addWeighted(luminance, 0.45, enhanced, 0.55, 0)
    contrast = Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))
    # Lift shadows in dark frames and tone down highlights in bright frames.
    gamma = 0.78 if np.median(luminance) < 100 else 1.18
    exposure = image.point([round(255 * (value / 255) ** gamma) for value in range(256)] * 3)
    night = night_exposure(image)
    return contrast, night if night is not None else exposure


class Segmenter:
    def __init__(self, checkpoint):
        import torch
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

        self.torch = torch
        torch.set_num_threads(min(torch.get_num_threads(), 4))
        self.processor = SegformerImageProcessor.from_pretrained(checkpoint, local_files_only=True)
        self.model = SegformerForSemanticSegmentation.from_pretrained(checkpoint, local_files_only=True).eval()
        self.names = [self.model.config.id2label[i].replace(" ", "_") for i in range(self.model.config.num_labels)]
        if tuple(self.names) != tuple(PALETTE):
            raise ValueError("Checkpoint không khớp thứ tự 19 lớp trong guideline.")

    def logits(self, image):
        inputs = self.processor(images=image, do_resize=False, return_tensors="pt")
        with self.torch.inference_mode():
            return self.model(**inputs).logits

    def upsample(self, logits, image):
        return self.torch.nn.functional.interpolate(logits, size=image.size[::-1], mode="bilinear", align_corners=False)

    def classify(self, logits, image, threshold):
        scores, ids = self.upsample(logits, image).softmax(dim=1)[0].topk(2, dim=0)
        labels = ids[0].to(self.torch.uint8)
        labels[(scores[0] < threshold) | (scores[0] - scores[1] < 0.12)] = VOID
        return labels, ids, scores

    def predict(self, image, threshold):
        with self.torch.inference_mode():
            labels, _, _ = self.classify(self.logits(image), image, threshold)
        return labels.cpu().numpy()

    def predict_robust(self, image, threshold):
        torch = self.torch
        with torch.inference_mode():
            original, contrast, exposure = [self.logits(view) for view in (image, *exposure_views(image))]
            fused = 0.5 * original + 0.3 * contrast + 0.2 * exposure
            labels, ids, scores = self.classify(fused, image, threshold)
            votes = torch.zeros_like(labels)
            for logits in (original, contrast, exposure):
                votes += (self.upsample(logits, image)[0].argmax(dim=0) == ids[0]).to(torch.uint8)
            disagreement = votes < 2
            labels[disagreement] = VOID
            return (labels.cpu().numpy(), ids.cpu().numpy(),
                    (scores[0] - scores[1]).cpu().numpy(), disagreement.cpu().numpy())

    def road_probability(self, image):
        with self.torch.inference_mode():
            return self.upsample(self.logits(image), image).softmax(dim=1)[0, CLASS_IDS["road"]].cpu().numpy()


def load_images(source):
    if source.suffix.lower() == ".html" and source.is_file():
        # Read the embedded original JPEG/PNG bytes without executing the HTML.
        source_text = source.read_text(encoding="utf-8")
        pattern = r'<h2>Frame (\d+) · ([^<]*)</h2>\s*<svg[^>]*>\s*<image[^>]*href="data:(image/(?:jpeg|png|webp));base64,([A-Za-z0-9+/=]+)"'
        from html import unescape

        for match in re.finditer(pattern, source_text):
            number, name, mime, encoded = match.groups()
            yield int(number), unescape(name), mime, base64.b64decode(encoded, validate=True)
    else:
        paths = sorted(source.iterdir()) if source.is_dir() else [source]
        types = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
        for number, path in enumerate(p for p in paths if p.is_file() and p.suffix.lower() in types):
            yield number, path.name, types[path.suffix.lower()], path.read_bytes()


def segmentation_box_cues(mask, categories):
    import cv2
    import numpy as np

    height, width = mask.shape
    scale = np.array([width, height, width, height]) / 1000
    cues = []
    for category in categories:
        label = BBOX_CUES[category]
        category_id = list(PALETTE).index(category)
        _, _, stats, _ = cv2.connectedComponentsWithStats((mask == category_id).astype(np.uint8), connectivity=8)
        for x, y, w, h, area in stats[1:]:
            if area < 12 or min(w, h) < 3:
                continue
            points = [int(x), int(y), int(x + w), int(y + h)]
            cues.append({"label": label, "bbox_2d": [int(round(p / s)) for p, s in zip(points, scale)],
                         "points": points, "reason": "Vùng semantic có thể gộp nhiều đối tượng. Kiểm tra BBox từng instance."})
    return cues


def detail_cues(segmenter, image, threshold, ignored):
    import numpy as np
    from PIL import Image, ImageEnhance

    width, height = image.size
    tile_w, tile_h = min(768, width), min(448, height)
    dark = np.asarray(image.convert("L").resize((32, 18))).mean() < 75
    detections = []
    structure_votes = {category: np.zeros((height, width), dtype=np.uint8) for category in STRUCTURAL_CLASSES}
    for top in tile_starts(height, tile_h):
        for left in tile_starts(width, tile_w):
            right, bottom = left + tile_w, top + tile_h
            tile = image.crop((left, top, right, bottom))
            if dark:
                tile = ImageEnhance.Contrast(tile.point(lambda n: round(255 * (n / 255) ** 0.78))).enhance(1.1)
            enlarged = tile.resize((round(tile_w * 1.45), round(tile_h * 1.45)), Image.Resampling.BICUBIC)
            mask = segmenter.predict(enlarged, threshold)
            # Return component bounds in original tile pixels, without creating new semantic classes.
            mask = np.asarray(Image.fromarray(mask).resize(tile.size, Image.Resampling.NEAREST)).copy()
            mask[ignored[top:bottom, left:right]] = VOID
            y0, y1 = (8 if top else 0), (tile_h - 8 if bottom < height else tile_h)
            x0, x1 = (8 if left else 0), (tile_w - 8 if right < width else tile_w)
            for category, class_id in STRUCTURAL_CLASSES.items():
                structure_votes[category][top + y0:top + y1, left + x0:left + x1] += (mask[y0:y1, x0:x1] == class_id)
            for cue in segmentation_box_cues(mask, BBOX_CUES):
                x1, y1, x2, y2 = cue["points"]
                if ((left and x1 <= 3) or (top and y1 <= 3)
                        or (right < width and x2 >= tile_w - 3)
                        or (bottom < height and y2 >= tile_h - 3)):
                    continue
                cue["points"] = [x1 + left, y1 + top, x2 + left, y2 + top]
                cue["bbox_2d"] = [round(p * 1000 / size) for p, size in zip(cue["points"], image.size * 2)]
                cue["source"] = "semantic_detail"
                detections.append(cue)
    return detections, structure_votes


def recover_structures(mask, top_ids, margin, disagreement, votes, ignored):
    import numpy as np

    recovered = {}
    for category, class_id in STRUCTURAL_CLASSES.items():
        support = votes[category]
        agrees = (top_ids[0] == class_id) | ((top_ids[1] == class_id) & (margin < 0.22))
        candidate = ~ignored & ~disagreement & (mask != class_id)
        # Recover from tiles only when the full image also supports that class.
        candidate &= ((mask == VOID) & agrees & (support >= 1)) | ((margin < 0.16) & agrees & (support >= 2))
        mask[candidate] = class_id
        recovered[category] = int(np.count_nonzero(candidate))
    return recovered


def night_signal_cues(image, ignored):
    import cv2
    import numpy as np

    rgb = np.asarray(image)
    height, width = ignored.shape
    if rgb[::12, ::12].mean() >= 75:
        return []
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lights = cv2.inRange(hsv, (40, 120, 170), (105, 255, 255))
    lights[round(height * 0.48):] = 0
    lights = cv2.morphologyEx(lights, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    lights[ignored] = 0
    _, _, stats, _ = cv2.connectedComponentsWithStats(lights, connectivity=8)
    cues = []
    for x, y, w, h, area in stats[1:]:
        if area < 35 or not 150 <= w * h <= 3500 or not 0.45 <= w / h <= 2.4:
            continue
        points = [int(x), int(y), int(x + w), int(y + h)]
        cues.append({"label": "traffic light", "points": points,
                     "bbox_2d": [round(p * 1000 / s) for p, s in zip(points, (width, height) * 2)],
                     "source": "night_signal", "reason": "Cụm sáng xanh trong ảnh tối. Xác minh đầu đèn và biên thực tế."})
    return cues


def filter_cues(cues, boxes, ignored):
    height, width = ignored.shape
    reference = [(box["label"], bounds(box, width, height)) for box in boxes]
    vehicles = [p for label, p in reference if label in VEHICLES]
    reviewed = [(c["label"], c["points"]) for c in cues if c.get("source") == "reviewed_correction"]
    selected, rejected = [], []
    for cue in sorted(cues, key=lambda c: (c.get("source") == "reviewed_correction",
                                           intersection(c["points"], c["points"])), reverse=True):
        label, points = cue["label"], cue["points"]
        reason = reject_reason(label, points, ignored)
        center_x, center_y = (points[0] + points[2]) / 2, (points[1] + points[3]) / 2
        if cue.get("source") != "reviewed_correction" and any(
                label == other_label and x1 <= center_x <= x2 and y1 <= center_y <= y2
                for other_label, (x1, y1, x2, y2) in reviewed):
            continue
        if (reason is None and cue.get("source") != "reviewed_correction"
                and label in {"rider", "pedestrian"} and vehicle_wheel_region(points, vehicles)):
            reason = "person_in_vehicle_wheel_region"
        if reason:
            rejected.append({**cue, "reason": reason})
            continue
        area = intersection(points, points)
        if any(label == other_label and intersection(points, other) >= 0.55 * min(area, intersection(other, other))
               for other_label, other in reference):
            continue
        selected.append(cue)
        reference.append((label, points))
    return selected, rejected


def save_mask(image, mask, output, stem):
    import numpy as np
    from PIL import Image

    palette = [tuple(bytes.fromhex(value)) for value in PALETTE.values()]
    lut = np.zeros((256, 3), dtype=np.uint8)
    lut[:len(palette)] = palette
    lut[VOID] = (255, 255, 255)
    indexed = Image.fromarray(mask).convert("P")
    indexed.putpalette(lut.ravel().tolist())
    indexed.save(output / f"{stem}-mask.png", transparency=VOID)
    rgb = np.asarray(image).astype(np.float32)
    mixed = rgb * 0.45 + lut[mask] * 0.55
    mixed[mask == VOID] = rgb[mask == VOID]
    Image.fromarray(mixed.astype(np.uint8)).save(output / f"{stem}-overlay.png")


def render_frame(item, data, mime, stem):
    preview = frame_preview(item, data, mime)
    extra = []
    for cue in item["semantic_box_cues"]:
        x1, y1, x2, y2 = cue["points"]
        extra.append(f'<rect class="cue" x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}"/>')
        label = cue["label"] + (" · đã rà" if cue.get("source") == "reviewed_correction" else "")
        extra.append(f'<text class="cue-label" x="{x1}" y="{max(14, y1-4)}">{escape(label)}</text>')
    preview = preview.replace("</svg>", "".join(extra) + "</svg>")
    uncertainty = item["uncertain_pixels"] / (item["width"] * item["height"])
    ignored = item["ignored_pixels"] / (item["width"] * item["height"])
    scope = item.get("out_of_scope_pixels", 0) / (item["width"] * item["height"])
    scope_note = f" Ngoài phạm vi guideline: {scope:.1%}." if scope else ""
    lighting = item["lighting_disagreement_pixels"] / (item["width"] * item["height"])
    building = item["structural_pixels"]["building"]
    pole = item["structural_pixels"]["pole"]
    road_repair = item["shadow_road_pixels"]
    manual = item["reviewed_surface_pixels"]
    checked = ", ".join(f"{label}: {pixels:,} px" for label, pixels in manual.items() if pixels)
    checked = f" Hiệu chỉnh theo biên đã rà trên ảnh gốc: {checked}." if checked else ""
    return (f'<section id="frame-{item["frame"]}">' + preview +
            f'<p>Pixel cần review: {uncertainty:.1%}. Vùng xe mình bỏ qua: {ignored:.1%}.{scope_note} '
            f'Hộp đã loại: {len(item["excluded_boxes"])}. <a href="{stem}-mask.png">Mask PNG</a></p>'
            f'<p>Bất đồng khi đổi sáng: {lighting:.1%}. Building: {building:,} px. Pole: {pole:,} px.</p>'
            f'<p>Bù vùng đường trong bóng: {road_repair:,} px.{checked}</p>'
            f'<p>Bù lỗ hổng trong thân xe: {item["vehicle_hole_pixels"]:,} px. '
            f'Bổ sung sau cân góc ảnh: {item["deskew_pixels"]:,} px.</p>'
            f'<img src="{stem}-overlay.png" alt="Overlay segmentation frame {item["frame"]}" loading="lazy"></section>')


def main():
    for stream in (sys.stdout, sys.stderr):
        if not stream.isatty() and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "bbox-job-222-yolo.html")
    parser.add_argument("--output", type=Path, default=ROOT / "review-job-222-seg-refined")
    parser.add_argument("--checkpoint", type=Path, default=MODEL)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--frame", type=int, help="chỉ xử lý frame này trong nguồn ảnh")
    parser.add_argument("--mask-confidence", type=float, default=0.55)
    parser.add_argument("--camera-profile", choices=("auto", "job-222", "none"), default="auto")
    args = parser.parse_args()
    if (args.limit is not None and args.limit < 1) or not 0 < args.mask_confidence <= 1:
        parser.error("Kiểm tra giá trị --limit và --mask-confidence.")
    if args.output.exists():
        parser.error("Chọn thư mục --output mới.")
    if not args.source.exists():
        parser.error(f"Không tìm thấy nguồn ảnh {args.source}.")
    if not args.checkpoint.is_dir():
        parser.error(f"Thiếu checkpoint SegFormer tại {args.checkpoint}.")

    job_222 = args.camera_profile == "job-222" or (
        args.camera_profile == "auto" and args.source.resolve() == ROOT / "bbox-job-222-yolo.html")
    try:
        generate_review(load_images(args.source), args.output, checkpoint=args.checkpoint, job_222=job_222,
                        frame=args.frame, limit=args.limit, mask_confidence=args.mask_confidence)
        print(args.output / "index.html")
    except ValueError as exc:
        parser.error(str(exc))


def generate_review(images, output, *, checkpoint=MODEL, job_222=False, frame=None, limit=None,
                    mask_confidence=0.55, progress=print, destination=None, guideline=None):
    import numpy as np
    from PIL import Image

    segmenter = Segmenter(checkpoint)
    detector = YoloDetector(YOLO_WEIGHTS, guideline.bbox_min_score if guideline else 0.30)
    if guideline:
        mask_confidence = guideline.mask_min_score
    output.mkdir(parents=True)
    frames, sections = [], {}
    for number, name, mime, data in images:
        if frame is not None and number != frame:
            continue
        with Image.open(BytesIO(data)) as file:
            image = file.convert("RGB")
        width, height = image.size
        ignored = ego_mask(width, height, number, job_222)
        detection = detector.predict_image(image)
        if not guideline:
            detection["issues"].extend(reviewed_corrections(detection["boxes"], data, ignored))
        detection["boxes"], excluded = filter_yolo(detection["boxes"], ignored, None if guideline else data)
        mask, top_ids, margin, disagreement = segmenter.predict_robust(image, mask_confidence)
        mask[ignored] = VOID
        detailed, votes = detail_cues(segmenter, image, mask_confidence, ignored)
        recovered = recover_structures(mask, top_ids, margin, disagreement, votes, ignored) if not guideline else {}
        shadow_road = 0
        relit = relight_pavement(image) if not guideline else None
        if relit is not None:
            relit_image, lifted = relit
            shadow_road = repair_shadowed_road(mask, segmenter.road_probability(relit_image), lifted,
                                               top_ids, margin, ignored, detection["boxes"], CLASS_IDS, VOID)
        source_hash = frame_fingerprint(data)
        rotation = deskew(image, source_hash) if not guideline else None
        deskew_pixels = 0
        if rotation is not None:
            straight, inverse = rotation
            deskew_pixels = restore_deskewed(mask, segmenter.predict(straight, mask_confidence),
                                             inverse, margin, ignored, CLASS_IDS, VOID)
        vehicle_holes = 0
        reviewed_surface = {}
        if not guideline:
            vehicle_holes = fill_vehicle_mask_holes(mask, detection["boxes"], ignored, CLASS_IDS, VOID)
            reviewed_surface = reviewed_frame2(mask, data, ignored, CLASS_IDS)
            reviewed_surface.update(reviewed_surfaces(mask, source_hash, ignored, CLASS_IDS))
        raw_cues = segmentation_box_cues(mask, BBOX_CUES)
        raw_cues.extend(detailed)
        raw_cues.extend(night_signal_cues(image, ignored))
        person = reviewed_person(data, width, height) if not guideline else None
        if person:
            raw_cues.append(person)
        cues, excluded_cues = filter_cues(raw_cues, detection["boxes"], ignored)
        scope_pixels = 0
        if guideline:
            def allowed(box):
                points = bounds(box, width, height)
                return (box["label"] in guideline.bbox
                        and intersection(points, points) >= guideline.bbox_min_area_px)

            excluded.extend({**b, "reason": "customer_guideline_scope"} for b in detection["boxes"] if not allowed(b))
            detection["boxes"] = [b for b in detection["boxes"] if allowed(b)]
            cues = [b for b in cues if allowed(b)]
            excluded_pixels = (mask != VOID) & ~np.isin(mask, [CLASS_IDS[n] for n in guideline.segmentation])
            scope_pixels = int(np.count_nonzero(excluded_pixels))
            mask[excluded_pixels] = VOID
        stem = f"frame-{number:06d}"
        save_mask(image, mask, output, stem)
        item = {"frame": number, "name": name, "width": width, "height": height, **detection,
                "semantic_box_cues": cues, "mask": f"{stem}-mask.png",
                "ignored_pixels": int(np.count_nonzero(ignored)),
                "uncertain_pixels": int(np.count_nonzero((mask == VOID) & ~ignored)) - scope_pixels,
                "out_of_scope_pixels": scope_pixels,
                "lighting_disagreement_pixels": int(np.count_nonzero(
                    disagreement & ~ignored & (mask == VOID) & (~excluded_pixels if guideline else True))),
                "shadow_road_pixels": shadow_road, "reviewed_surface_pixels": reviewed_surface,
                "vehicle_hole_pixels": vehicle_holes, "deskew_pixels": deskew_pixels,
                "structural_pixels": {category: int(np.count_nonzero(mask == idx))
                                      for category, idx in STRUCTURAL_CLASSES.items()},
                "structural_recovered_pixels": recovered,
                "excluded_boxes": excluded + excluded_cues,
                "present_classes": [segmenter.names[i] for i in np.unique(mask) if i != VOID]}
        frames.append(item)
        display = item
        if guideline:
            display = {**item, **{key: [{**box, "label": guideline.bbox[box["label"]]} for box in item[key]]
                                  for key in ("boxes", "semantic_box_cues")}}
        sections[number] = render_frame(display, data, mime, stem)
        progress(f"Frame {number} · {len(item['boxes'])} BBox · {len(cues)} gợi ý · "
                 f"{len(item['present_classes'])} lớp segmentation.")
        if len(frames) == limit:
            break
    if not frames:
        raise ValueError("Nguồn không có ảnh hoặc frame được chọn.")
    priority = {number: i for i, number in enumerate(PRIORITY_FRAMES)} if job_222 else {}
    ordered = sorted(frames, key=lambda f: (priority.get(f["frame"], len(priority)),
                                           -f["uncertain_pixels"] / (f["width"] * f["height"]), f["frame"]))
    order = [frame["frame"] for frame in ordered]
    report = {"class_ids": list(PALETTE), "palette": PALETTE, "void_id": VOID,
              "camera_profile": "job-222" if job_222 else "none", "review_order": order, "frames": frames}
    if destination is not None:
        report["destination"] = destination
    if guideline is not None:
        report["customer_guideline"] = guideline.metadata()
    (output / "review.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    legend = " ".join(f'<span style="border-left:12px solid #{PALETTE[label]};padding-left:4px">{escape(name)}</span>'
                      for label, name in (guideline.segmentation.items() if guideline else ((n, n) for n in PALETTE)))
    rules = (f'<p>Guideline {escape(guideline.source)} · Điểm YOLO tối thiểu {guideline.bbox_min_score:g} '
             f'· Diện tích hộp tối thiểu {guideline.bbox_min_area_px} px².</p>' if guideline else "")
    mask_note = ("Vùng ngoài phạm vi và vùng thiếu chắc chắn dùng ID 255 trong mask." if guideline else
                 "Lượt cân bằng sáng cục bộ tìm lại vùng đường trong bóng. Vùng thiếu chắc chắn và vùng xe mình dùng ID 255 trong mask.")
    html = ('<!doctype html><html lang="vi"><meta charset="utf-8"><title>Kiểm tra BBox và segmentation</title>'
            '<style>body{font:16px system-ui;max-width:1280px;margin:auto;padding:1rem}'
            'img,svg{width:100%;display:block}rect{fill:none;stroke:#00ff86;stroke-width:2}'
            'text{fill:#00ff86;stroke:#000;stroke-width:3;paint-order:stroke;font:bold 16px system-ui}'
            '.cue{stroke:#ffd84b;stroke-dasharray:5 3}.cue-label{fill:#ffd84b}span{display:inline-block;margin:3px}</style>'
            '<h1>Kiểm tra BBox và segmentation</h1><p>Hộp xanh là BBox đạt ngưỡng YOLO. Hộp vàng nét đứt là gợi ý từ '
            'mask, tín hiệu màu ban đêm hoặc ca đã rà. ' + mask_note + '</p>'
            '<p>Thứ tự review ' + ' · '.join(f'<a href="#frame-{n}">{n}</a>' for n in order) + '</p>'
            + rules + f'<p>{legend}</p>' + "\n".join(sections[n] for n in order) + "</html>\n")
    (output / "index.html").write_text(html, encoding="utf-8")
    return report


if __name__ == "__main__":
    main()
