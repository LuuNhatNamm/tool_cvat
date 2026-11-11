"""Read the supported, structured part of a customer annotation guideline."""

from dataclasses import dataclass
import json
from pathlib import Path
import re

from yolo_bbox import COCO_LABELS


@dataclass(frozen=True)
class CustomerGuideline:
    source: str
    bbox: dict
    segmentation: dict
    bbox_min_score: float = 0.30
    bbox_min_area_px: int = 150
    mask_min_score: float = 0.55

    def metadata(self):
        return {"file": self.source, "bbox": self.bbox, "segmentation": self.segmentation,
                "bbox_min_score": self.bbox_min_score, "bbox_min_area_px": self.bbox_min_area_px,
                "mask_min_score": self.mask_min_score}


def load_guideline(path, semantic_classes):
    path = Path(path)
    if path.suffix.lower() != ".md":
        raise ValueError("Guideline cần là file Markdown .md.")
    document = path.read_text(encoding="utf-8-sig")
    blocks = re.findall(r"^```cvat-guideline[ \t]*\r?\n(.*?)^```[ \t]*$", document, re.MULTILINE | re.DOTALL)
    if len(blocks) != 1:
        raise ValueError("Guideline cần đúng một khối cvat-guideline. Xem mẫu trong thư mục data.")

    def unique_object(pairs):
        result = dict(pairs)
        if len(result) != len(pairs):
            raise ValueError("Guideline có khóa JSON hoặc nhãn nguồn bị trùng.")
        return result

    config = json.loads(blocks[0], object_pairs_hook=unique_object)
    supported = {"version", "bbox", "segmentation", "bbox_min_score", "bbox_min_area_px", "mask_min_score"}
    if not isinstance(config, dict) or type(config.get("version")) is not int or config["version"] != 1:
        raise ValueError("Guideline cần version = 1.")
    unknown = config.keys() - supported
    if unknown:
        raise ValueError("Quy tắc chưa được hỗ trợ: " + ", ".join(sorted(unknown)))
    mappings = {}
    for key, classes in (("bbox", set(COCO_LABELS.values())), ("segmentation", set(semantic_classes))):
        mapping = config.get(key, {})
        if not isinstance(mapping, dict):
            raise ValueError(f"{key} phải là bảng nhãn nguồn → tên nhãn CVAT.")
        missing = mapping.keys() - classes
        if missing:
            raise ValueError(f"Model chưa hỗ trợ {key}: " + ", ".join(sorted(missing)))
        if any(not isinstance(v, str) or not v or v.strip() != v or any(ord(c) < 32 for c in v)
               for v in mapping.values()):
            raise ValueError(f"Tên nhãn đích trong {key} phải là chuỗi hợp lệ.")
        if len(set(mapping.values())) != len(mapping):
            raise ValueError(f"Mỗi nhãn đích trong {key} cần một nhãn nguồn riêng.")
        mappings[key] = mapping
    if not any(mappings.values()):
        raise ValueError("Guideline cần ít nhất một ánh xạ nhãn.")
    bbox_score, area, mask_score = (config.get(key, fallback) for key, fallback in
                                    (("bbox_min_score", 0.30), ("bbox_min_area_px", 150), ("mask_min_score", 0.55)))
    if type(bbox_score) not in (int, float) or not 0.30 <= bbox_score <= 1:
        raise ValueError("bbox_min_score phải nằm trong khoảng 0.30–1.")
    if type(area) is not int or area < 150:
        raise ValueError("bbox_min_area_px phải là số nguyên từ 150.")
    if type(mask_score) not in (int, float) or not 0 < mask_score <= 1:
        raise ValueError("mask_min_score phải lớn hơn 0 và không quá 1.")
    return CustomerGuideline(path.name, **mappings, bbox_min_score=bbox_score,
                             bbox_min_area_px=area, mask_min_score=mask_score)
