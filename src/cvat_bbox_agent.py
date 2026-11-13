"""Propose traffic bounding boxes with local models and save them to a CVAT job."""

import argparse
import base64
from collections import defaultdict
import csv
from getpass import getpass
from html import escape
import json
import os
from pathlib import Path
import sys
from time import monotonic
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from annotation_scope import ego_mask, filter_yolo, reviewed_corrections


LABELS = (
    "pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle",
    "bicycle", "traffic light", "traffic sign",
)
ISSUES = (
    "UNCERTAIN_CLASS", "UNCERTAIN_BOUNDARY", "UNCERTAIN_SCOPE",
    "ATTRIBUTE_CHECK", "LOW_VISIBILITY",
)
ROOT = Path(__file__).resolve().parent.parent
GUIDELINE = ROOT / "docs" / "BBOX_ANNOTATION_GUIDELINE.md"
TOKEN_FILE = ROOT / "config" / "access_tokens.csv"
YOLO_WEIGHTS = ROOT / "models" / "yolo11n.pt"
OLLAMA = "http://127.0.0.1:11434"
SERVER = os.getenv("CVAT_SERVER", "").rstrip("/")
ORGANIZATION = os.getenv("CVAT_ORGANIZATION", "")
TASK_ID = int(os.getenv("CVAT_TASK_ID", "0"))
JOB_ID = int(os.getenv("CVAT_JOB_ID", "0"))
DUPLICATE_IOU = 0.85
MAX_BATCH_IMAGES = 10


class AgentError(Exception):
    pass


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def request(base, path, *, headers=None, payload=None, method="GET", image=False, local=False):
    headers = dict(headers or {})
    if payload is not None:
        headers["Content-Type"] = "application/json"
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = Request(base + path, data=body, headers=headers, method=method)
    # Never forward credentials or images to a redirect or a local HTTP proxy.
    handlers = [NoRedirect()]
    if local:
        handlers.append(ProxyHandler({}))
    streaming = local and path == "/api/chat" and payload.get("stream")
    try:
        with build_opener(*handlers).open(req, timeout=900 if streaming else 60) as response:
            if streaming:
                return read_ollama_chat(response)
            data = response.read()
            return (data, response.headers.get_content_type()) if image else json.loads(data)
    except HTTPError as exc:
        hint = ""
        if exc.code in (401, 403) and not local:
            hint = " Kiểm tra token, Organization và quyền trên job."
        if exc.code == 404 and local:
            hint = " Chạy ollama pull cho model đã chọn."
        raise AgentError(f"{method} {base}{path}: HTTP {exc.code}.{hint}") from exc
    except (URLError, TimeoutError) as exc:
        raise AgentError(f"Không kết nối được {base} ({method} {path}): {exc}") from exc


def read_ollama_chat(response):
    fragments = []
    last_update = monotonic()
    for line in response:
        if not line.strip():
            continue
        chunk = json.loads(line)
        if chunk.get("error"):
            raise AgentError(f"Ollama báo lỗi: {chunk['error']}")
        text = chunk.get("message", {}).get("content", "")
        if text:
            fragments.append(text)
            now = monotonic()
            if now - last_update >= 60:
                print("Ollama đang sinh kết quả cho frame này…", flush=True)
                last_update = now
        if chunk.get("done"):
            return {"done_reason": chunk.get("done_reason"), "message": {"content": "".join(fragments)}}
    raise AgentError("Luồng kết quả từ Ollama đã đóng trước khi nhận đủ dữ liệu.")


class CVAT:
    def __init__(self, *, server=SERVER, organization=ORGANIZATION, task_id=TASK_ID, job_id=JOB_ID, token=None):
        if not server or task_id <= 0 or job_id <= 0:
            raise AgentError("Nhập link job trong Job Runner, hoặc đặt CVAT_SERVER, CVAT_TASK_ID và CVAT_JOB_ID khi dùng CLI.")
        self.server, self.organization = server, organization
        self.task_id, self.job_id = task_id, job_id
        legacy = None
        use_local_credentials = token is None and server == SERVER
        if use_local_credentials:
            token = os.getenv("CVAT_ACCESS_TOKEN")
            legacy = os.getenv("CVAT_LEGACY_TOKEN")
        if use_local_credentials and not token and not legacy and TOKEN_FILE.is_file():
            token = read_csv_token(TOKEN_FILE)
        if use_local_credentials and not token and not legacy and sys.stdin is not None and sys.stdin.isatty():
            token = getpass("Nhập CVAT Personal Access Token (ký tự được ẩn): ").strip()
        if not token and not legacy:
            raise AgentError("Đặt CVAT_ACCESS_TOKEN trong PowerShell trước khi chạy.")
        self.headers = {"Authorization": f"Bearer {token}" if token else f"Token {legacy}"}

    def api(self, path, *, params=None, **kwargs):
        query = {**({"org": self.organization} if self.organization else {}), **(params or {})}
        return request(self.server, path + "?" + urlencode(query), headers=self.headers, **kwargs)

    def list(self, path, **filters):
        page = 1
        while True:
            result = self.api(path, params={"page": page, "page_size": 50, **filters})
            yield from result["results"]
            if not result["next"]:
                break
            page += 1

    def job(self):
        job = self.api(f"/api/jobs/{self.job_id}")
        if job["id"] != self.job_id or job["task_id"] != self.task_id:
            raise AgentError(f"CVAT trả về job ngoài phạm vi task {self.task_id}, job {self.job_id}.")
        if job["stage"] != "annotation" or job.get("type", "annotation") != "annotation":
            raise AgentError(f"Job {self.job_id} phải thuộc stage annotation và loại annotation.")
        if job.get("mode", "annotation") != "annotation" or job.get("dimension", "2d") != "2d":
            raise AgentError("Tool hỗ trợ job ảnh 2D ở chế độ annotation.")
        return job

    def frames(self, job):
        meta = self.api(f"/api/jobs/{job['id']}/data/meta")
        frames = range(job["start_frame"], job["stop_frame"] + 1)
        if len(meta["frames"]) != len(frames) or meta.get("included_frames") is not None:
            raise AgentError("Job có cấu trúc frame khác job ảnh liên tục. Cần kiểm tra ánh xạ frame trước khi ghi.")
        deleted = set(meta.get("deleted_frames") or [])
        return {n: info for n, info in zip(frames, meta["frames"]) if n not in deleted}

    def labels(self, job_id):
        labels = {}
        for label in self.list("/api/labels", job_id=job_id):
            name = label["name"]
            if name in LABELS and label.get("type", "any") in ("any", "rectangle"):
                if name in labels:
                    raise AgentError(f"Nhãn {name!r} xuất hiện nhiều lần trong job.")
                labels[name] = label
        missing = set(LABELS) - labels.keys()
        if missing:
            raise AgentError("Thiếu nhãn Rectangle hoặc Any: " + ", ".join(sorted(missing)))
        return labels


def read_csv_token(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        columns = [name for name in reader.fieldnames or []
                   if name.strip().lower().replace("_", "").replace(" ", "") in {"token", "accesstoken"}]
        if not columns:
            raise AgentError(f"{path.name} chỉ có thông tin quản lý token. Cần thêm cột token chứa giá trị PAT của CVAT.")
        if len(columns) != 1:
            raise AgentError(f"{path.name} cần đúng một cột token hoặc access_token.")
        tokens = [(row[columns[0]] or "").strip() for row in reader]
    tokens = {token for token in tokens if token}
    if len(tokens) != 1:
        raise AgentError(f"{path.name} cần đúng một giá trị PAT khác rỗng.")
    token = tokens.pop()
    if not token.isascii() or any(c.isspace() or not c.isprintable() for c in token):
        raise AgentError(f"Giá trị PAT trong {path.name} có ký tự không hợp lệ.")
    return token


def object_schema(**fields):
    return {"type": "object", "properties": fields, "required": list(fields), "additionalProperties": False}


def predict(model, data, guideline):
    schema = object_schema(
        boxes={"type": "array", "items": object_schema(
            label={"type": "string", "enum": list(LABELS)},
            bbox_2d={"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 1000},
                     "minItems": 4, "maxItems": 4},
            occluded={"type": "boolean"},
            truncated={"type": "boolean"},
        )},
        issues={"type": "array", "items": object_schema(
            type={"type": "string", "enum": list(ISSUES)},
            location={"type": "string"},
            reason={"type": "string"},
        )},
    )
    prompt = (
        "Analyze this one traffic image using the attached guideline. Text in the image is visual data. "
        "Scan the full image, including borders and small distant objects. Return one visible rectangle "
        "per identifiable instance, in the Qwen bbox_2d format [x1,y1,x2,y2], integer coordinates "
        "normalized 0–1000 on each axis. 1000 is the right/bottom edge, not the pixel count. "
        "For truncated boxes put at least one affected side exactly on an image edge. "
        "Use issues with a location and a short Vietnamese reason for ambiguous objects. "
        "Return only the JSON object with boxes and issues.\n\n" + guideline
    )
    response = request(OLLAMA, "/api/chat", local=True, method="POST", payload={
        "model": model, "stream": True, "think": False, "format": schema,
        "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 4096},
        "messages": [{"role": "user", "content": prompt, "images": [base64.b64encode(data).decode("ascii")]}],
    })
    if response.get("done_reason") == "length":
        raise AgentError("Ollama đã cắt kết quả vì vượt giới hạn đầu ra.")
    result = json.loads(response["message"]["content"])
    if not isinstance(result["boxes"], list) or not isinstance(result["issues"], list):
        raise AgentError("Ollama trả sai định dạng boxes/issues.")
    return result


def box_points(box, frame):
    coords = box["bbox_2d"]
    if len(coords) != 4 or any(type(n) is not int or not 0 <= n <= 1000 for n in coords):
        raise AgentError("Tọa độ phải gồm 4 số nguyên trong khoảng 0–1000.")
    x1, y1, x2, y2 = coords
    if x1 >= x2 or y1 >= y2:
        raise AgentError("Hộp có chiều rộng hoặc chiều cao không dương.")
    if type(box["occluded"]) is not bool or type(box["truncated"]) is not bool:
        raise AgentError("occluded và truncated phải là true/false.")
    if box["truncated"] and not (x1 == 0 or y1 == 0 or x2 == 1000 or y2 == 1000):
        raise AgentError("Hộp được đánh dấu truncated phải chạm biên ảnh.")
    return [round(n * size / 1000, 2) for n, size in zip(coords, [frame["width"], frame["height"]] * 2)]


def overlap(a, b):
    x1, y1, x2, y2 = a
    u1, v1, u2, v2 = b
    intersection = max(0, min(x2, u2) - max(x1, u1)) * max(0, min(y2, v2) - max(y1, v1))
    union = (x2 - x1) * (y2 - y1) + (u2 - u1) * (v2 - v1) - intersection
    return intersection / union if union > 0 else 0


def shapes_for_plan(plan, frames, labels, current):
    known = defaultdict(list)
    for shape in current["shapes"]:
        if shape["type"] == "rectangle" and len(shape["points"]) == 4 and not shape.get("outside"):
            known[shape["frame"], shape["label_id"]].append(shape["points"])
    rectangle_tracks = [
        track for track in current.get("tracks", [])
        if any(shape["type"] == "rectangle" for shape in track["shapes"])
    ]
    if rectangle_tracks:
        raise AgentError("Job có rectangle track. Cần review để tránh trùng với hộp nội suy.")
    shapes, issues = [], []
    for item in plan["frames"]:
        number = item["frame"]
        frame = frames.get(number)
        if frame is None or any(frame[key] != item[key] for key in ("width", "height", "name")):
            raise AgentError(f"Frame {number} đã thay đổi hoặc nằm ngoài job.")
        for box in item["boxes"]:
            label = labels.get(box.get("label"))
            if label is None:
                raise AgentError(f"Nhãn không hợp lệ ở frame {number}: {box.get('label')!r}.")
            points = box_points(box, frame)
            candidates = known[number, label["id"]]
            if any(overlap(points, candidate) >= DUPLICATE_IOU for candidate in candidates):
                continue
            truncated = next((a for a in label.get("attributes", []) if a["name"] == "truncated"), None)
            attrs = []
            if truncated and truncated["input_type"] == "checkbox":
                attrs.append({"spec_id": truncated["id"], "value": str(box["truncated"]).lower()})
            elif box["truncated"]:
                issues.append(f"ATTRIBUTE_CHECK | Frame {number} | {label['name']} {points} | Job thiếu checkbox truncated.")
                continue
            shapes.append({
                "type": "rectangle", "frame": number, "label_id": label["id"], "points": points,
                "occluded": box["occluded"], "attributes": attrs, "source": "auto",
            })
            candidates.append(points)
    return shapes, issues


def frame_preview(item, image, mime):
    width, height = item["width"], item["height"]
    encoded = base64.b64encode(image).decode("ascii")
    drawing = [f'<image width="{width}" height="{height}" href="data:{mime};base64,{encoded}"/>']
    for box in item["boxes"]:
        x1, y1, x2, y2 = box_points(box, item)
        label = escape(box["label"])
        drawing.append(f'<rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}"/>')
        drawing.append(f'<text x="{x1}" y="{max(14, y1-3)}">{label}</text>')
    issues = "".join(f"<li>{escape(issue['type'])} · {escape(issue['location'])} · {escape(issue['reason'])}</li>"
                     for issue in item["issues"])
    return (f"<h2>Frame {item['frame']} · {escape(item['name'])}</h2>"
            f'<svg viewBox="0 0 {width} {height}">{"".join(drawing)}</svg><ul>{issues}</ul>')


def preview(args, cvat, job, frames):
    output = args.output or Path(f"bbox-job-{JOB_ID}.json")
    html = output.with_suffix(".html")
    if output.suffix.lower() != ".json" or output.exists() or html.exists():
        raise AgentError("Chọn --output .json chưa tồn tại, với file .html cùng tên cũng chưa tồn tại.")
    available = list(frames)
    if args.frame is not None:
        if args.frame not in frames:
            raise AgentError(f"Frame {args.frame} không có trong job.")
        available = available[available.index(args.frame):]
    selected = available if args.all else available[:args.limit]
    if len(selected) > MAX_BATCH_IMAGES:
        raise AgentError("Mỗi lượt tối đa 10 ảnh. Dùng --frame và --limit để chọn nhóm ảnh.")
    if not selected:
        raise AgentError("Job không có frame phù hợp.")
    cvat.labels(job["id"])
    detector = None
    if args.model == "yolo11n":
        from yolo_bbox import YoloDetector

        detector = YoloDetector(YOLO_WEIGHTS, args.confidence)
    else:
        installed = request(OLLAMA, "/api/tags", local=True)
        if args.model not in {m["name"] for m in installed["models"]} or "cloud" in args.model.lower():
            raise AgentError(f"Cần model Ollama đã tải local. Chạy ollama pull {args.model}.")
    plan = {"server": SERVER, "org": ORGANIZATION, "task": TASK_ID, "job": JOB_ID,
            "model": args.model, "frames": []}
    guideline = GUIDELINE.read_text(encoding="utf-8") if detector is None else None
    page = ['<!doctype html><html lang="vi"><meta charset="utf-8"><title>Kiểm tra BBox</title>'
            '<style>body{font:16px system-ui;max-width:1320px;margin:2rem auto;padding:0 1rem}'
            'svg{width:100%;display:block}rect{stroke:#00ff86;stroke-width:2;fill:none}'
            'text{fill:#00ff86;stroke:#000;stroke-width:3;paint-order:stroke;font: bold 16px system-ui}</style>'
            f"<h1>Kiểm tra BBox của job {job['id']}</h1>"]
    if detector is not None:
        page.append("<p>Rà rider, biển báo ngoài biển STOP, vật nhỏ và thuộc tính che khuất trên từng ảnh. "
                    "YOLO COCO thiếu lớp riêng cho các trường hợp này.</p>")
    for number in selected:
        image, mime = cvat.api(f"/api/jobs/{job['id']}/data",
                               params={"type": "frame", "number": number, "quality": "original"}, image=True)
        if mime not in {"image/jpeg", "image/png", "image/webp"}:
            raise AgentError(f"Frame {number} trả định dạng không hỗ trợ: {mime}.")
        item = {"frame": number, **{key: frames[number][key] for key in ("width", "height", "name")}}
        print(f"Frame {number}: đang nhận diện bằng {args.model}…", flush=True)
        result = detector.predict(image, item) if detector is not None else predict(args.model, image, guideline)
        if detector is not None:
            ignored = ego_mask(item["width"], item["height"], number, False)
            result["issues"].extend(reviewed_corrections(result["boxes"], image, ignored))
            result["boxes"], item["excluded_boxes"] = filter_yolo(result["boxes"], ignored, image)
        item.update(boxes=[], issues=result["issues"])
        for box in result["boxes"]:
            try:
                if box["label"] not in LABELS:
                    raise AgentError("Nhãn ngoài taxonomy.")
                box_points(box, item)
            except (AgentError, KeyError, TypeError) as exc:
                item["issues"].append({"type": "UNCERTAIN_BOUNDARY", "location": str(box), "reason": str(exc)})
            else:
                item["boxes"].append(box)
        plan["frames"].append(item)
        page.append(frame_preview(item, image, mime))
        print(f"Frame {number}: {len(item['boxes'])} hộp đề xuất, {len(item['issues'])} Issue.", flush=True)
    with output.open("x", encoding="utf-8") as file:
        json.dump(plan, file, ensure_ascii=False, indent=2)
        file.write("\n")
    with html.open("x", encoding="utf-8") as file:
        file.write("\n".join(page) + "\n</html>\n")
    print(f"Xem trước: {html}\nKết quả đề xuất: {output}")
    return output


def check_assignee(cvat, job):
    if job["state"] == "completed":
        raise AgentError("Job đã completed. Mở lại qua luồng CVAT nếu cần sửa.")
    user = cvat.api("/api/users/self")
    if (job.get("assignee") or {}).get("id") != user["id"]:
        raise AgentError("Tài khoản của token phải là assignee của job để ghi BBox.")


def apply(args, cvat, job, frames):
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if not 1 <= len(plan["frames"]) <= MAX_BATCH_IMAGES:
        raise AgentError("Bản đề xuất cần chứa từ 1 đến 10 ảnh.")
    identity = (SERVER, ORGANIZATION, TASK_ID, JOB_ID)
    if tuple(plan[key] for key in ("server", "org", "task", "job")) != identity:
        raise AgentError("File đề xuất thuộc máy chủ, Organization, task hoặc job khác.")
    check_assignee(cvat, job)
    endpoint = f"/api/jobs/{job['id']}/annotations/"
    labels = cvat.labels(job["id"])
    current = cvat.api(endpoint)
    shapes, issues = shapes_for_plan(plan, frames, labels, current)
    for item in plan["frames"]:
        for issue in item["issues"]:
            print(f"{issue['type']} | Frame {item['frame']} | {issue['location']} | {issue['reason']}")
    for issue in issues:
        print(issue)
    if not shapes:
        print("Không có hộp mới cần ghi.")
        return
    result = cvat.api(endpoint, method="PATCH", params={"action": "create"}, payload={
        "version": current["version"], "shapes": shapes, "tags": [], "tracks": [],
    })
    saved = result.get("shapes", [])
    if len(saved) != len(shapes) or any("id" not in shape for shape in saved):
        raise AgentError("CVAT chưa xác nhận đủ hộp mới. Kiểm tra job trước khi chạy lại.")
    print(f"CVAT xác nhận đã thêm {len(saved)} BBox vào job {job['id']}.")


def main():
    for stream in (sys.stdout, sys.stderr):
        if not stream.isatty() and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=f"Gán BBox local cho CVAT task {TASK_ID}, job {JOB_ID}.")
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("preview", help="nhận diện local và xuất JSON/HTML xem trước")
    inspect.add_argument("--model", default="yolo11n")
    auto = sub.add_parser("run", help="chạy YOLO và ghi BBox vào job đã cấu hình trong một lệnh")
    auto.set_defaults(model="yolo11n")
    for command, default_limit in ((inspect, 1), (auto, 5)):
        command.add_argument("--frame", type=int, help="frame bắt đầu, dùng số frame của CVAT")
        scope = command.add_mutually_exclusive_group()
        scope.add_argument("--limit", type=int, default=default_limit)
        scope.add_argument("--all", action="store_true")
        command.add_argument("--output", type=Path)
        command.add_argument("--confidence", type=float, default=0.30)
    save = sub.add_parser("apply", help="ghi thêm các hộp đã xem trước vào CVAT")
    save.add_argument("--plan", required=True, type=Path)
    args = parser.parse_args()
    if args.command in {"preview", "run"}:
        if not 1 <= args.limit <= MAX_BATCH_IMAGES:
            parser.error("--limit phải từ 1 đến 10")
        if not 0 < args.confidence <= 1:
            parser.error("--confidence phải lớn hơn 0 và không quá 1")
    try:
        cvat = CVAT()
        job = cvat.job()
        frames = cvat.frames(job)
        if args.command == "run":
            check_assignee(cvat, job)
            args.plan = preview(args, cvat, job, frames)
            apply(args, cvat, cvat.job(), frames)
        elif args.command == "preview":
            preview(args, cvat, job, frames)
        else:
            apply(args, cvat, job, frames)
    except (AgentError, OSError, ValueError, KeyError, TypeError, EOFError, ImportError, RuntimeError) as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Đã dừng.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
