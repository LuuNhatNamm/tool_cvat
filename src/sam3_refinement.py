"""Refine semantic object masks with box prompts and a local SAM 3 checkpoint."""

import math


OBJECT_NAMES = {"pedestrian": "person", "rider": "rider", "car": "car", "truck": "truck",
                "bus": "bus", "train": "train", "motorcycle": "motorcycle", "bicycle": "bicycle",
                "traffic light": "traffic_light", "traffic sign": "traffic_sign"}
VOID = 255


class Sam3Refiner:
    def __init__(self, checkpoint):
        if not checkpoint.is_dir() or not (checkpoint / "config.json").is_file() or not (
                (checkpoint / "model.safetensors").is_file() or (checkpoint / "model.safetensors.index.json").is_file()):
            raise ValueError(f"Thiếu checkpoint SAM 3 tại {checkpoint}. Xem cách tải trong README.")
        try:
            import torch
            from transformers import Sam3TrackerModel, Sam3TrackerProcessor
        except ImportError as exc:
            raise ValueError("Cần Transformers 5.x có Sam3TrackerModel và PyTorch để chạy SAM 3.") from exc

        self.torch = torch
        self.processor = Sam3TrackerProcessor.from_pretrained(checkpoint, local_files_only=True)
        self.model = Sam3TrackerModel.from_pretrained(checkpoint, local_files_only=True, use_safetensors=True).eval()

    def predict(self, image, boxes):
        inputs = self.processor(images=image, input_boxes=[[list(box) for box in boxes]], return_tensors="pt")
        with self.torch.inference_mode():
            output = self.model(pixel_values=inputs["pixel_values"].to(dtype=self.model.dtype), input_boxes=inputs["input_boxes"],
                                multimask_output=False)
        scores = output.iou_scores[0, :, 0].cpu().tolist()
        if len(scores) != len(boxes):
            raise ValueError("SAM 3 trả về số mask không khớp số BBox.")
        for start in range(0, len(boxes), 4):
            masks = self.processor.post_process_masks(output.pred_masks[:, start:start + 4].cpu(),
                                                       inputs["original_sizes"])[0]
            for binary, score in zip(masks[:, 0], scores[start:start + 4]):
                yield binary.numpy(), score


def refine_objects(refiner, image, mask, boxes, ignored, class_ids, *, allowed_classes=None, min_score=0.65):
    import numpy as np

    width, height = image.size
    requests = []
    for box in boxes:
        name = OBJECT_NAMES.get(box["label"])
        if name is None or (allowed_classes is not None and name not in allowed_classes):
            continue
        x1, y1, x2, y2 = [value * size / 1000 for value, size in zip(box["bbox_2d"], (width, height) * 2)]
        bounds = (max(0, math.floor(x1)), max(0, math.floor(y1)), min(width, math.ceil(x2)), min(height, math.ceil(y2)))
        if bounds[2] > bounds[0] and bounds[3] > bounds[1]:
            requests.append((name, bounds))
    if not requests:
        return {"prompted": 0, "accepted": 0, "changed_pixels": 0}

    original = mask.copy()
    protected = np.isin(original, [class_ids[name] for name in {*OBJECT_NAMES.values(), "pole"}])
    candidates = []
    for (name, (x1, y1, x2, y2)), (prediction, score) in zip(
            requests, refiner.predict(image, [bounds for _, bounds in requests]), strict=True):
        binary = np.asarray(prediction, dtype=bool)
        if binary.shape != mask.shape:
            raise ValueError("SAM 3 trả về mask sai kích thước ảnh.")
        if not math.isfinite(score) or score < min_score:
            continue
        region = np.s_[y1:y2, x1:x2]
        cropped = binary[region]
        size = int(np.count_nonzero(cropped))
        if not 0.05 * cropped.size <= size <= 0.98 * cropped.size or size < 0.75 * np.count_nonzero(binary):
            continue
        class_id = class_ids[name]
        eligible = cropped & ~ignored[region] & (~protected[region] | (original[region] == class_id))
        if np.count_nonzero(eligible) < 0.5 * size:
            continue
        candidates.append((score, class_id, region, eligible))

    claimed = np.zeros(mask.shape, dtype=bool)
    accepted = []
    for _, class_id, region, eligible in sorted(candidates, key=lambda item: -item[0]):
        owned = eligible & ~claimed[region]
        if not np.any(owned):
            continue
        mask[region][owned] = class_id
        claimed[region] |= owned
        accepted.append((class_id, region, eligible))

    for class_id, region, eligible in accepted:
        same_class = original[region] == class_id
        if np.count_nonzero(same_class & eligible) >= 0.5 * np.count_nonzero(same_class):
            trim = same_class & ~eligible & ~claimed[region] & ~ignored[region]
            mask[region][trim] = VOID
    return {"prompted": len(requests), "accepted": len(accepted),
            "changed_pixels": int(np.count_nonzero(mask != original))}
