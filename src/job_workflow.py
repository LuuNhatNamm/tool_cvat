"""Read the selected CVAT job, build a review, and save that reviewed batch."""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from cvat_bbox_agent import AgentError, MAX_BATCH_IMAGES, check_assignee
from job_preview import SAM3_BBOX, SAM3_MIXED, make_review, write_review_html
from publish_cvat_review import label_map, prepare, publish, shape_key


NOTES = {"skip_existing_segmentation": "Frame đã có segmentation khác",
         "skip_existing_extra": "Frame đã có shape khác cùng kiểu và nhãn",
         "skip_cross_label_bbox": "BBox trùng vị trí, khác nhãn",
         "skip_missing_truncated": "BBox cần thuộc tính truncated"}


@dataclass
class Review:
    output: Path
    mode: str
    destination: dict
    frames: dict
    labels: dict
    guideline: object
    polygon_names: tuple
    shapes: list


def batch_range(first, last):
    if first < 1 or last < first or last - first + 1 > MAX_BATCH_IMAGES:
        raise ValueError("Nhập khoảng ảnh bắt đầu từ 1, tối đa 10 ảnh mỗi lượt.")
    return first, last


def job_images(cvat, frames):
    for number, info in frames.items():
        data, mime = cvat.api(f"/api/jobs/{cvat.job_id}/data",
                             params={"type": "frame", "number": number, "quality": "original"}, image=True)
        if mime not in {"image/jpeg", "image/png", "image/webp"}:
            raise AgentError(f"Frame {number} trả định dạng {mime}. Tool cần job ảnh 2D.")
        yield number, info["name"], mime, data


def destination(cvat):
    return {"server": cvat.server, "task": cvat.task_id, "job": cvat.job_id, "organization": cvat.organization}


def annotations_endpoint(cvat):
    return f"/api/jobs/{cvat.job_id}/annotations/"


def plan_shapes(review, existing):
    shapes, updates, notes = prepare(review.output, review.frames, review.labels, existing,
                                     review.mode, review.destination, review.guideline,
                                     polygon_names=review.polygon_names, allow_unmarked_truncation=True)
    if updates:
        raise AgentError("Bản review cần cập nhật shape có sẵn. Tải lại job để xem trước.")
    shapes = list({shape_key(shape): shape for shape in shapes}.values())
    for shape in shapes:
        shape["source"] = "manual"
        for element in shape.get("elements", ()):
            element["source"] = "manual"
    messages = [f"Frame {frame} · {NOTES[reason]} ({count})."
                for (frame, reason), count in sorted(Counter(notes).items())]
    return shapes, messages


def preview_job(cvat, mode, output, emit, guideline, prompts, first, last, *, reference=None):
    batch_range(first, last)
    frames = cvat.frames(cvat.job())
    if last > len(frames):
        raise AgentError(f"Job có {len(frames)} ảnh. Chọn khoảng trong danh sách ảnh của job.")
    selected = dict(sorted(frames.items())[first - 1:last])
    shape_mode = "bbox" if mode == SAM3_BBOX else mode
    polygon_names = tuple(prompts)
    if mode == SAM3_MIXED:
        if guideline.bbox and guideline.segmentation:
            shape_mode = "both"
        else:
            shape_mode = "bbox" if guideline.bbox else "segmentation"
        polygon_names = tuple(guideline.segmentation)
    labels = label_map(cvat, shape_mode, guideline)
    review = Review(output, shape_mode, destination(cvat), selected, labels, guideline, polygon_names, [])
    emit("phase", ("Đang tạo bản xem trước trên máy", len(selected)))
    emit("log", f"Ảnh {first}–{last} · Frame CVAT {', '.join(map(str, selected))}.")
    items, originals = make_review(job_images(cvat, selected), output, review.destination,
                                   mode, guideline, prompts, emit, labels=labels)
    emit("phase", ("Đang đối chiếu annotation CVAT và xuất bản xem trước", 0))
    review.shapes, notes = plan_shapes(review, cvat.api(annotations_endpoint(cvat)))
    for message in notes:
        emit("log", message)
    job_url = f"{cvat.server}/tasks/{cvat.task_id}/jobs/{cvat.job_id}"
    write_review_html(output, items, originals, labels, review.shapes, mode, job_url, notes, reference=reference)
    return review


def save_review(cvat, review, emit):
    if destination(cvat) != review.destination:
        raise AgentError("Bản review thuộc job khác. Tạo lại bản xem trước.")
    job = cvat.job()
    check_assignee(cvat, job)
    frames = cvat.frames(job)
    if any(frames.get(n) != info for n, info in review.frames.items()):
        raise AgentError("Ảnh trên CVAT đã thay đổi. Tạo lại bản xem trước.")
    if label_map(cvat, review.mode, review.guideline) != review.labels:
        raise AgentError("Nhãn trên CVAT đã thay đổi. Tạo lại bản xem trước.")
    endpoint = annotations_endpoint(cvat)
    existing = cvat.api(endpoint)
    shapes, notes = plan_shapes(review, existing)
    if any(shape not in review.shapes for shape in shapes):
        raise AgentError("Có shape khác bản đã xem. Tạo lại bản xem trước.")
    existing_keys = {shape_key(shape) for shape in existing["shapes"]}
    pending_keys = {shape_key(shape) for shape in shapes}
    if not {shape_key(shape) for shape in review.shapes} <= existing_keys | pending_keys:
        raise AgentError("Annotation trên CVAT đã thay đổi. Tạo lại bản xem trước.")
    for message in notes:
        emit("log", message)
    publish(cvat, endpoint, existing, shapes, [], progress=lambda message: emit("log", message))
    return len(shapes)
