"""Read the supported, structured part of a customer annotation guideline."""

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from xml.etree import ElementTree
from zipfile import ZipFile

from yolo_bbox import COCO_LABELS


@dataclass(frozen=True)
class CustomerGuideline:
    source: str
    bbox: dict
    segmentation: dict
    bbox_min_score: float = 0.30
    bbox_min_area_px: int = 150
    mask_min_score: float = 0.55
    extra: dict = field(default_factory=dict)

    def metadata(self):
        return {"file": self.source, "bbox": self.bbox, "segmentation": self.segmentation,
                "bbox_min_score": self.bbox_min_score, "bbox_min_area_px": self.bbox_min_area_px,
                "mask_min_score": self.mask_min_score, **({"extra": self.extra} if self.extra else {})}


@dataclass(frozen=True)
class GuidelineReference:
    name: str
    text: str
    shapes: dict[str, str]
    settings: CustomerGuideline | None = None
    model_inputs: dict[str, dict[str, str]] | None = None


def read_reference(path, semantic_classes):
    path = Path(path)
    if path.suffix.lower() == ".md":
        text = path.read_text(encoding="utf-8-sig")
        if not re.search(r"^```cvat-guideline\b", text, re.M):
            return GuidelineReference(path.name, text, {})
        config = read_config(text)
        if type(config.get("version")) is int and config["version"] == 2:
            return parse_v2(path.name, text, config)
        return GuidelineReference(path.name, text, {}, parse_v1(path.name, config, semantic_classes))
    if path.suffix.lower() != ".docx":
        raise ValueError("Chọn guideline .docx hoặc .md.")
    with ZipFile(path) as archive:
        root = ElementTree.fromstring(archive.read("word/document.xml"))
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

    def paragraph(node):
        return "".join(part.text or "" for part in node.findall(".//w:t", ns)).strip()

    text = "\n".join(filter(None, (paragraph(node) for node in root.findall(".//w:p", ns))))
    shapes = {}
    kinds = {"rectangle / bounding box": "rectangle", "rectangle": "rectangle", "bounding box": "rectangle",
             "polygon": "polygon", "polyline": "polyline", "mask": "mask", "skeleton": "skeleton"}
    for table in root.findall(".//w:tbl", ns):
        rows = [[" ".join(filter(None, (paragraph(p) for p in cell.findall("w:p", ns))))
                 for cell in row.findall("w:tc", ns)] for row in table.findall("w:tr", ns)]
        if not rows or "CVAT shape" not in rows[0] or "Class áp dụng" not in rows[0]:
            continue
        shape_column, label_column = (rows[0].index(column) for column in ("CVAT shape", "Class áp dụng"))
        for cells in rows[1:]:
            if len(cells) <= max(shape_column, label_column):
                continue
            shape = kinds.get(cells[shape_column].casefold())
            if not shape:
                raise ValueError(f"Kiểu shape trong guideline chưa được nhận ra: {cells[shape_column]}.")
            for name in map(str.strip, cells[label_column].split(",")):
                if name:
                    if name in shapes and shapes[name] != shape:
                        raise ValueError(f"Guideline quy định nhiều kiểu shape cho nhãn {name}.")
                    shapes[name] = shape
    return GuidelineReference(path.name, text, shapes)


def load_guideline(path, semantic_classes):
    path = Path(path)
    if path.suffix.lower() != ".md":
        raise ValueError("Guideline cần là file Markdown .md.")
    return parse_v1(path.name, read_config(path.read_text(encoding="utf-8-sig")), semantic_classes)


def read_config(document):
    blocks = re.findall(r"^```cvat-guideline[ \t]*\r?\n(.*?)^```[ \t]*$", document, re.MULTILINE | re.DOTALL)
    if len(blocks) != 1:
        raise ValueError("Guideline cần đúng một khối cvat-guideline. Xem mẫu trong thư mục data.")

    def unique_object(pairs):
        result = dict(pairs)
        if len(result) != len(pairs):
            raise ValueError("Guideline có khóa JSON hoặc nhãn nguồn bị trùng.")
        return result

    try:
        config = json.loads(blocks[0], object_pairs_hook=unique_object)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON trong khối cvat-guideline lỗi ở dòng {exc.lineno}, cột {exc.colno}.") from exc
    if not isinstance(config, dict):
        raise ValueError("Khối cvat-guideline phải chứa một đối tượng JSON.")
    return config


def parse_v1(name, config, semantic_classes):
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
    return CustomerGuideline(name, **mappings, **read_thresholds(config, sam3_default=0.55))


def read_thresholds(config, sam3_default):
    bbox_score = config.get("bbox_min_score", 0.30)
    area = config.get("bbox_min_area_px", 150)
    mask_score = config.get("mask_min_score", sam3_default)
    if type(bbox_score) not in (int, float) or not 0.30 <= bbox_score <= 1:
        raise ValueError("bbox_min_score phải nằm trong khoảng 0.30–1.")
    if type(area) is not int or area < 150:
        raise ValueError("bbox_min_area_px phải là số nguyên từ 150.")
    if type(mask_score) not in (int, float) or not 0 < mask_score <= 1:
        raise ValueError("mask_min_score phải lớn hơn 0 và không quá 1.")
    return dict(bbox_min_score=bbox_score, bbox_min_area_px=area, mask_min_score=mask_score)


def parse_v2(name, text, config):
    unknown = config.keys() - {"version", "labels", "bbox_min_score", "bbox_min_area_px", "mask_min_score"}
    if unknown:
        raise ValueError("Trường guideline chưa được hỗ trợ: " + ", ".join(sorted(unknown)))
    labels = config.get("labels")
    if not isinstance(labels, dict) or not labels:
        raise ValueError("Guideline version 2 cần bảng labels có ít nhất một nhãn CVAT.")
    shapes, inputs = {}, {}
    for label, entry in labels.items():
        if not label or label != label.strip() or any(ord(c) < 32 for c in label):
            raise ValueError(f"Tên nhãn CVAT không hợp lệ: {label!r}.")
        if not isinstance(entry, dict) or entry.keys() - {"shape", "yolo_class", "sam3_prompt"}:
            raise ValueError(f"Nhãn {label} chỉ nhận shape, yolo_class và sam3_prompt.")
        shape = entry.get("shape")
        if shape not in ("rectangle", "polygon", "polyline", "mask", "skeleton"):
            raise ValueError(f"Nhãn {label} cần shape là rectangle, polygon, polyline, mask hoặc skeleton.")
        models = {key: value for key, value in entry.items() if key != "shape"}
        if (shape == "skeleton" and models) or (shape != "rectangle" and "yolo_class" in models):
            raise ValueError(f"Cấu hình model của nhãn {label} không phù hợp với shape {shape}.")
        for key, value in models.items():
            if not isinstance(value, str) or not value.strip() or any(ord(c) < 32 for c in value):
                raise ValueError(f"Nhãn {label} cần {key} là một dòng chữ có nội dung.")
        shapes[label] = shape
        inputs[label] = {key: value.strip() for key, value in models.items()}
    settings = CustomerGuideline(name, {}, {}, **read_thresholds(config, sam3_default=0.50))
    return GuidelineReference(name, text, shapes, settings, inputs)
