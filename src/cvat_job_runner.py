"""Select CVAT labels, preview a batch locally, then save the reviewed shapes."""

from datetime import datetime
from importlib.util import find_spec
import os
from pathlib import Path
from queue import Empty, Queue
import re
from threading import Thread
import tkinter as tk
from tkinter import filedialog, ttk
from tkinter.scrolledtext import ScrolledText
from urllib.parse import urlsplit
from uuid import uuid4
import webbrowser
from xml.etree.ElementTree import ParseError
from zipfile import BadZipFile

from customer_guideline import CustomerGuideline, read_reference
from cvat_bbox_agent import AgentError, CVAT, ROOT, SERVER, YOLO_WEIGHTS
from extra_geometry import joint_indices
from job_preview import SAM3_MIXED, POSE_MODEL
from job_workflow import batch_range, preview_job, save_review
from local_annotation_review import PALETTE, SAM3_MODEL, sam3_ready
from yolo_bbox import COCO_LABELS


MODES = {"SAM 3 và Pose": SAM3_MIXED, "Job BBox (YOLO)": "bbox"}
SHAPE_OPTIONS = {"Rectangle": ("rectangle",), "Polygon": ("polygon",),
                 "Rectangle và Polygon": ("rectangle", "polygon"), "Polyline": ("polyline",),
                 "Mask": ("mask",), "Skeleton": ("skeleton",)}
PROMPT_ALIASES = {"pole": "utility pole", "traffic sign": "traffic sign", "road": "road surface",
                  "lane/crosswalk": "white pedestrian crossing stripe", "lane/single yellow": "yellow lane marking"}


def parse_job_url(value):
    parts = urlsplit(value.strip())
    route = re.fullmatch(r"/tasks/([1-9]\d*)/jobs/([1-9]\d*)/?", parts.path)
    if (parts.scheme != "https" or not re.fullmatch(r"[A-Za-z0-9.:-]+", parts.hostname or "")
            or parts.username is not None or not route or parts.query or parts.fragment):
        raise ValueError("Nhập link HTTPS dạng https://may-chu/tasks/123/jobs/456.")
    _ = parts.port
    task_id, job_id = map(int, route.groups())
    return f"https://{parts.netloc.lower()}", task_id, job_id


def normalized(name):
    return " ".join(name.replace("_", " ").replace("-", " ").lower().split())


def yolo_classes():
    package = find_spec("ultralytics")
    if not YOLO_WEIGHTS.is_file() or package is None:
        return ()
    import yaml

    # This tool ships a COCO YOLO checkpoint. Read the class list bundled with Ultralytics.
    path = Path(package.origin).parent / "cfg" / "datasets" / "coco.yaml"
    names = yaml.safe_load(path.read_text(encoding="utf-8"))["names"]
    return tuple(names.values())


def selection(mode, rows, sources, reference=None):
    maps = {shape: {} for shape in ("rectangle", "polygon", "polyline", "mask", "skeleton")}
    for name, source, output in rows:
        shapes = SHAPE_OPTIONS[output]
        if reference and reference.shapes and shapes != (reference.shapes.get(name),):
            raise ValueError(f"Kiểu shape của nhãn {name} không khớp guideline.")
        if shapes == ("skeleton",):
            if maps["skeleton"]:
                raise ValueError("Chọn một nhãn Skeleton người mỗi lượt để tránh nhân đôi pose.")
            maps["skeleton"][name] = name
            continue
        source = source.strip()
        if not source:
            raise ValueError(f"Chọn lớp YOLO hoặc nhập mô tả cho nhãn {name}.")
        if mode == "bbox" and source not in sources:
            raise ValueError(f"Lớp YOLO {source} không được hỗ trợ.")
        if mode != "bbox":
            source = " ".join(source.lower().split())
        for shape in shapes:
            mapping = maps[shape]
            if source in mapping:
                raise ValueError(f"Hai nhãn {mapping[source]} và {name} cùng dùng {source} cho kiểu {shape}. Chỉ tick một nhãn.")
            mapping[source] = name
    if not any(maps.values()):
        raise ValueError("Chọn ít nhất một nhãn CVAT để xem trước.")
    thresholds = reference.settings if reference and reference.settings else None
    return CustomerGuideline(reference.name if reference else "Nhãn đã chọn",
                             bbox=maps["rectangle"], segmentation=maps["polygon"],
                             bbox_min_score=thresholds.bbox_min_score if thresholds else 0.30,
                             bbox_min_area_px=thresholds.bbox_min_area_px if thresholds else 150,
                             mask_min_score=thresholds.mask_min_score if thresholds else 0.50,
                             extra={kind: maps[kind] for kind in ("polyline", "mask", "skeleton") if maps[kind]})


class JobRunnerUI:
    def __init__(self, root):
        self.root = root
        self.events = Queue()
        self.busy = False
        self.cvat = None
        self.review = None
        self.reference = None
        self.labels = []
        self.sources = ()
        self.frame_count = 0
        self.rows = []
        self.url, self.token, self.org = tk.StringVar(), tk.StringVar(), tk.StringVar()
        self.mode = tk.StringVar(value=next(iter(MODES)))
        self.first, self.last = tk.StringVar(value="1"), tk.StringVar(value="5")
        root.title("CVAT Job Runner")
        root.geometry("950x730")
        root.minsize(740, 610)
        root.protocol("WM_DELETE_WINDOW", self.close)

        panel = ttk.Frame(root, padding=16)
        panel.pack(expand=True, fill="both")
        panel.columnconfigure(1, weight=1)
        panel.rowconfigure(6, weight=3)
        panel.rowconfigure(9, weight=1)
        ttk.Label(panel, text="CVAT Job Runner", font=("TkDefaultFont", 19, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))
        self.connection_widgets = []
        for row, caption, var, mask in ((1, "Link job", self.url, ""), (2, "Access token", self.token, "●"),
                                        (3, "Organization", self.org, "")):
            ttk.Label(panel, text=caption).grid(row=row, column=0, sticky="w", padx=(0, 12), pady=3)
            entry = ttk.Entry(panel, textvariable=var, show=mask)
            entry.grid(row=row, column=1, sticky="ew", pady=3)
            self.connection_widgets.append(entry)
            var.trace_add("write", self.connection_changed)

        choices = ttk.Frame(panel)
        choices.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(6, 4))
        self.mode_box = ttk.Combobox(choices, textvariable=self.mode, values=list(MODES), state="readonly", width=32)
        self.mode_box.pack(side="left")
        self.mode_box.bind("<<ComboboxSelected>>", self.mode_changed)
        self.load_button = ttk.Button(choices, text="Tải nhãn từ job", command=self.load_labels)
        self.load_button.pack(side="left", padx=8)
        self.guideline_button = ttk.Button(choices, text="Chọn guideline", command=self.choose_reference)
        self.guideline_button.pack(side="left")
        self.clear_guideline = ttk.Button(choices, text="Bỏ guideline", command=self.clear_reference)
        self.clear_guideline.pack(side="left", padx=6)

        heading = ttk.Frame(panel)
        heading.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(6, 3))
        ttk.Label(heading, text="Nhãn trong job", font=("TkDefaultFont", 10, "bold")).pack(side="left")
        self.select_labels_button = ttk.Button(heading, text="Chọn tất cả", command=self.select_all_labels)
        self.select_labels_button.pack(side="left", padx=(8, 0))
        self.clear_labels_button = ttk.Button(heading, text="Bỏ chọn tất cả", command=self.clear_label_selection)
        self.clear_labels_button.pack(side="left", padx=8)
        self.label_hint = ttk.Label(heading)
        self.label_hint.pack(side="right")
        label_area = ttk.Frame(panel)
        label_area.grid(row=6, column=0, columnspan=2, sticky="nsew")
        self.canvas = tk.Canvas(label_area, height=140, highlightthickness=1, highlightbackground="#cfd5dd")
        self.canvas.pack(side="left", expand=True, fill="both")
        scroll = ttk.Scrollbar(label_area, command=self.canvas.yview)
        scroll.pack(side="right", fill="y")
        self.canvas.configure(yscrollcommand=scroll.set)
        self.label_table = ttk.Frame(self.canvas, padding=6)
        self.table_window = self.canvas.create_window(0, 0, window=self.label_table, anchor="nw")
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self.table_window, width=e.width))
        self.label_table.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        image_range = ttk.Frame(panel)
        image_range.grid(row=7, column=0, columnspan=2, sticky="ew", pady=6)
        self.range_widgets = []
        for caption, var in (("Từ ảnh", self.first), ("Đến ảnh", self.last)):
            ttk.Label(image_range, text=caption).pack(side="left", padx=(0, 5))
            entry = ttk.Entry(image_range, textvariable=var, width=6)
            entry.pack(side="left", padx=(0, 12))
            self.range_widgets.append(entry)
            var.trace_add("write", self.invalidate_review)
        self.range_hint = ttk.Label(image_range)
        self.range_hint.pack(side="left")
        self.status = ttk.Label(panel, text="Nhập thông tin job, sau đó tải nhãn.", wraplength=660)
        self.status.grid(row=8, column=0, columnspan=2, sticky="ew", pady=(0, 5))
        self.log = ScrolledText(panel, height=5, wrap="word", state="disabled")
        self.log.grid(row=9, column=0, columnspan=2, sticky="nsew")
        buttons = ttk.Frame(panel)
        buttons.grid(row=10, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        self.preview_button = ttk.Button(buttons, text="Chạy xem trước", command=self.start_preview)
        self.preview_button.pack(side="left")
        self.report_button = ttk.Button(buttons, text="Mở bản review", command=self.open_report)
        self.report_button.pack(side="left", padx=8)
        self.save_button = ttk.Button(buttons, text="Lưu lên CVAT", command=self.start_save)
        self.save_button.pack(side="left")
        self.job_button = ttk.Button(buttons, text="Mở job CVAT", command=self.open_job)
        self.job_button.pack(side="right")
        self.render_labels()
        root.after(100, self.poll)

    def refresh_buttons(self):
        ready = not self.busy
        for widget in [*self.connection_widgets, *self.range_widgets, self.load_button]:
            widget.configure(state="normal" if ready else "disabled")
        self.mode_box.configure(state="readonly" if ready else "disabled")
        self.guideline_button.configure(state="normal" if ready else "disabled")
        self.clear_guideline.configure(state="normal" if ready and self.reference else "disabled")
        selection_state = "normal" if ready and any(row[1].get() for row in self.rows) else "disabled"
        self.preview_button.configure(state=selection_state)
        self.clear_labels_button.configure(state=selection_state)
        can_select = ready and any(not chosen.get() and source.get().strip()
                                   for _, chosen, source, *_ in self.rows)
        self.select_labels_button.configure(state="normal" if can_select else "disabled")
        self.report_button.configure(state="normal" if ready and self.review else "disabled")
        self.save_button.configure(state="normal" if ready and self.review and self.review.shapes else "disabled")
        self.job_button.configure(state="normal" if ready and self.cvat else "disabled")
        for _, _, _, _, check, entry, entry_state, shape_box in self.rows:
            check.configure(state="normal" if ready else "disabled")
            entry.configure(state=entry_state if ready else "disabled")
            if shape_box is not None:
                shape_box.configure(state="readonly" if ready else "disabled")

    def invalidate_review(self, *_):
        if self.review:
            self.review = None
            self.status.configure(text="Lựa chọn đã thay đổi. Chạy xem trước để cập nhật kết quả.")
        self.refresh_buttons()

    def select_all_labels(self):
        yolo = MODES[self.mode.get()] == "bbox"

        def keys(source, output):
            if output == "Skeleton":
                return {("skeleton", "person")}
            prompt = source.strip() if yolo else " ".join(source.lower().split())
            return {(shape, prompt) for shape in SHAPE_OPTIONS[output]} if prompt else set()

        used = set()
        for _, chosen, source, output, *_ in self.rows:
            if chosen.get():
                used.update(keys(source.get(), output.get()))
        missing = duplicates = 0
        for _, chosen, source, output, *_ in self.rows:
            if chosen.get():
                continue
            options = keys(source.get(), output.get())
            if not options:
                missing += 1
            elif options & used:
                duplicates += 1
            else:
                chosen.set(True)
                used.update(options)
        self.invalidate_review()
        selected = sum(chosen.get() for _, chosen, *_ in self.rows)
        message = f"Đã chọn {selected} nhãn."
        if missing:
            message += f" {missing} nhãn chưa chọn vì thiếu lớp YOLO hoặc mô tả SAM 3."
        if duplicates:
            message += f" {duplicates} nhãn chưa chọn vì trùng lớp YOLO, mô tả SAM 3 hoặc chế độ Pose."
        self.status.configure(text=message)

    def clear_label_selection(self):
        for _, chosen, *_ in self.rows:
            chosen.set(False)
        self.invalidate_review()
        self.status.configure(text="Đã bỏ chọn tất cả nhãn. Tick nhãn bạn muốn chạy.")

    def connection_changed(self, *_):
        self.cvat = None
        self.labels = []
        self.frame_count = 0
        self.review = None
        self.status.configure(text="Thông tin kết nối đã thay đổi. Tải lại nhãn từ job.")
        self.render_labels()

    def mode_changed(self, _event):
        self.invalidate_review()
        self.render_labels()

    def choose_reference(self):
        path = filedialog.askopenfilename(parent=self.root, initialdir=ROOT / "data",
                                          filetypes=[("Guideline Word hoặc Markdown", "*.docx *.md")])
        if not path:
            return
        try:
            reference = read_reference(path, PALETTE)
        except (OSError, ValueError, BadZipFile, ParseError, KeyError) as exc:
            self.status.configure(text=f"Không đọc được guideline: {exc}")
            return
        self.reference = reference
        self.invalidate_review()
        self.render_labels()
        note = "Đã áp dụng danh sách nhãn và kiểu shape." if reference.shapes else "Nhãn vẫn do bạn chọn trong bảng."
        self.status.configure(text=f"{reference.name} · {note}")
        self.append(self.status.cget("text"))
        self.log_reference_mismatches()

    def log_reference_mismatches(self):
        if not self.reference or not self.reference.shapes or not self.labels:
            return
        labels = {label["name"]: label for label in self.labels if not (label.get("parent") or label.get("parent_id"))}
        for name, shape in self.reference.shapes.items():
            if name not in labels:
                self.append(f"Guideline có nhãn {name} nhưng job chưa có nhãn này.")
            elif labels[name].get("type", "any") not in (shape, "any"):
                self.append(f"Nhãn {name} cần kiểu {shape} theo guideline. Kiểm tra kiểu nhãn trên CVAT.")

    def clear_reference(self):
        self.reference = None
        self.invalidate_review()
        self.render_labels()
        self.status.configure(text="Đã bỏ guideline. Chọn nhãn trong bảng để chạy xem trước.")

    def render_labels(self):
        for child in self.label_table.winfo_children():
            child.destroy()
        self.rows.clear()
        mode = MODES[self.mode.get()]
        yolo = mode == "bbox"
        self.label_hint.configure(text=f"Guideline: {self.reference.name}" if self.reference
                                  else "Tick nhãn cần chạy. Các nhãn khác được bỏ qua.")
        self.range_hint.configure(text=f"{self.frame_count} ảnh trong job · tối đa 10 mỗi lượt" if self.frame_count
                                  else "Tối đa 10 ảnh mỗi lượt")
        if not self.labels:
            ttk.Label(self.label_table, text="Tải nhãn từ job để bắt đầu.").grid(sticky="w")
            self.refresh_buttons()
            return
        self.label_table.columnconfigure(1, weight=1)
        self.label_table.columnconfigure(3, weight=1)
        for column, text in enumerate(("Chọn", "Tên nhãn CVAT", "Kiểu xuất", "Lớp YOLO" if yolo else "Mô tả cho SAM 3")):
            ttk.Label(self.label_table, text=text).grid(row=0, column=column, sticky="w", padx=6)
        yolo_aliases = {normalized(name): name for name in self.sources}
        yolo_aliases.update({normalized(label): name for name, label in COCO_LABELS.items() if name in self.sources})
        scope = self.reference.shapes if self.reference else {}
        settings = self.reference.settings if self.reference else None
        bbox_hints = {target: source for source, target in settings.bbox.items()} if settings else {}
        polygon_hints = {target: source for source, target in settings.segmentation.items()} if settings else {}
        for row, label in enumerate(self.labels, 1):
            name, kind = label["name"], label.get("type", "any")
            expected_shape = scope.get(name)
            ttk.Label(self.label_table, text=name).grid(row=row, column=1, sticky="w", padx=6)
            options = [option for option, shapes in SHAPE_OPTIONS.items()
                       if all((kind == shape or (kind == "any" and shape != "skeleton")) and (not expected_shape or expected_shape == shape)
                              and (not yolo or shape == "rectangle") for shape in shapes)]
            reason = ""
            if scope and not expected_shape:
                reason = "Ngoài danh sách nhãn trong guideline"
            elif not options or label.get("parent") or label.get("parent_id"):
                reason = "Shape không phù hợp với chế độ đã chọn"
            elif kind == "skeleton":
                try:
                    joint_indices(label)
                except ValueError as exc:
                    reason = str(exc)
            if reason:
                ttk.Label(self.label_table, text=expected_shape or kind).grid(row=row, column=2, sticky="w", padx=6)
                ttk.Label(self.label_table, text=reason).grid(row=row, column=3, sticky="w", padx=6)
                continue
            chosen = tk.BooleanVar(value=False)
            default = "Polygon" if name in polygon_hints and not yolo and "Polygon" in options else options[0]
            output = tk.StringVar(value=default)
            shape_box = None
            if len(options) > 1:
                shape_box = ttk.Combobox(self.label_table, textvariable=output, values=options,
                                         width=21, state="readonly")
                shape_box.grid(row=row, column=2, sticky="w", padx=6)
                output.trace_add("write", self.invalidate_review)
            else:
                ttk.Label(self.label_table, text=default).grid(row=row, column=2, sticky="w", padx=6)
            suggestions = polygon_hints if default == "Polygon" else bbox_hints
            hint = normalized(suggestions.get(name, name))
            if name in ("area/direct", "area/alternative") and name not in suggestions:
                hint = ""
            initial = yolo_aliases.get(hint, "") if yolo else PROMPT_ALIASES.get(hint, hint)
            if self.reference and self.reference.model_inputs is not None:
                inputs = self.reference.model_inputs[name]
                initial = inputs.get("yolo_class" if yolo else "sam3_prompt", "")
                if not yolo and expected_shape == "polyline" and not initial:
                    initial = PROMPT_ALIASES.get(name, "")
                if yolo and initial and initial not in self.sources:
                    ttk.Label(self.label_table, text=f"YOLO không có lớp {initial}. Chọn lại lớp.").grid(
                        row=row, column=4, sticky="w", padx=6)
                    initial = ""
            source = tk.StringVar(value=initial)
            if kind == "skeleton":
                source.set("YOLO Pose · COCO người")
            check = ttk.Checkbutton(self.label_table, variable=chosen, command=self.invalidate_review)
            check.grid(row=row, column=0)
            state = "readonly" if yolo else "disabled" if kind == "skeleton" else "normal"
            if yolo:
                entry = ttk.Combobox(self.label_table, values=self.sources, textvariable=source, state=state, width=23)
            else:
                entry = ttk.Entry(self.label_table, textvariable=source, width=25)
            entry.grid(row=row, column=3, sticky="ew", padx=6, pady=2)
            source.trace_add("write", self.invalidate_review)
            self.rows.append((name, chosen, source, output, check, entry, state, shape_box))
        self.canvas.yview_moveto(0)
        self.refresh_buttons()

    def load_labels(self):
        try:
            server, task, job = parse_job_url(self.url.get())
            token = self.token.get().strip()
            if token and (not token.isascii() or any(c.isspace() or not c.isprintable() for c in token)):
                raise ValueError("Access token phải là một dòng, không có dấu cách.")
            if not token and server != SERVER:
                raise ValueError("Nhập token của máy chủ CVAT trong link job.")
            organization = self.org.get().strip()
        except ValueError as exc:
            self.status.configure(text=str(exc))
            return
        self.connection_changed()

        def work(emit):
            cvat = CVAT(server=server, task_id=task, job_id=job, organization=organization, token=token or None)
            frames = cvat.frames(cvat.job())
            labels = sorted(cvat.list("/api/labels", job_id=job), key=lambda label: label["name"].casefold())
            if not labels:
                raise AgentError("Job chưa có nhãn CVAT.")
            if not frames:
                raise AgentError("Job không có ảnh khả dụng.")
            sources = yolo_classes()
            emit("loaded", (cvat, labels, len(frames), sources))
            return f"Đã tải {len(labels)} nhãn. Chọn nhãn cần gán và chạy xem trước."

        self.start_worker("Đang tải nhãn từ CVAT…", work)

    def start_preview(self):
        cvat = self.cvat
        if not cvat:
            return
        try:
            first, last = batch_range(int(self.first.get()), int(self.last.get()))
            if last > self.frame_count:
                raise ValueError(f"Job có {self.frame_count} ảnh. Chọn nhóm ảnh trong khoảng này.")
            mode = MODES[self.mode.get()]
            if mode == "bbox" and not self.sources:
                raise ValueError(f"Cần Ultralytics và model YOLO tại {YOLO_WEIGHTS}.")
            selected = [(name, source.get(), output.get())
                        for name, chosen, source, output, *_ in self.rows if chosen.get()]
            reference = self.reference
            selected_labels = selection(mode, selected, self.sources, reference)
            if mode != "bbox" and any((selected_labels.bbox, selected_labels.segmentation,
                                       selected_labels.extra.get("polyline"), selected_labels.extra.get("mask"))) and not sam3_ready():
                raise ValueError(f"Thiếu checkpoint SAM 3 tại {SAM3_MODEL}.")
            if selected_labels.extra.get("skeleton") and not POSE_MODEL.is_file():
                raise ValueError(f"Thiếu model pose tại {POSE_MODEL}.")
        except ValueError as exc:
            self.status.configure(text=str(exc))
            self.append(str(exc))
            return
        self.invalidate_review()
        suffix = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:6]
        output = ROOT / "runs" / f"task-{cvat.task_id}-job-{cvat.job_id}-{suffix}"

        def work(emit):
            prompts = selected_labels.bbox | selected_labels.segmentation if mode == SAM3_MIXED else {}
            prompts |= selected_labels.extra.get("polyline", {}) | selected_labels.extra.get("mask", {})
            review = preview_job(cvat, mode, output, emit, selected_labels, prompts, first, last, reference=reference)
            emit("preview", review)
            if not review.shapes:
                return f"Đã xem trước {len(review.frames)} ảnh · Không có shape mới để lưu."
            return f"Đã xem trước {len(review.frames)} ảnh · {len(review.shapes)} shape sẵn sàng lưu."

        self.start_worker("Đang kiểm tra nhóm ảnh…", work)

    def start_save(self):
        cvat, review = self.cvat, self.review
        if not cvat or not review or not review.shapes:
            return

        def work(emit):
            count = save_review(cvat, review, emit)
            emit("saved", None)
            return f"CVAT đã xác nhận thêm {count} shape vào job {cvat.job_id}."

        self.start_worker(f"Đang lưu bản đã xem trước vào job {cvat.job_id}…", work)

    def start_worker(self, status, work):
        if self.busy:
            return
        self.busy = True
        self.refresh_buttons()
        self.status.configure(text=status)
        self.append(status)
        secret = self.token.get().strip()

        def worker():
            def emit(event, value):
                self.events.put((event, value))
            try:
                emit("done", work(emit))
            except Exception as exc:
                message = str(exc) or type(exc).__name__
                emit("error", message.replace(secret, "[token]") if secret else message)

        Thread(target=worker, daemon=True).start()

    def append(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def poll(self):
        try:
            while True:
                event, value = self.events.get_nowait()
                if event == "loaded":
                    self.cvat, self.labels, self.frame_count, self.sources = value
                    self.first.set("1")
                    self.last.set(str(min(5, self.frame_count)))
                    self.render_labels()
                    self.log_reference_mismatches()
                elif event == "preview":
                    self.review = value
                    self.append(f"File xem trước: {value.output / 'index.html'}")
                    self.open_report()
                elif event == "saved":
                    self.review.shapes = []
                elif event == "phase":
                    self.status.configure(text=value[0])
                    self.append(value[0])
                elif event == "log":
                    self.append(value)
                elif event in {"done", "error"}:
                    self.busy = False
                    self.status.configure(text=value if event == "done" else "Lỗi · Xem chi tiết trong nhật ký.")
                    self.append(value)
                    self.refresh_buttons()
        except Empty:
            pass
        self.root.after(100, self.poll)

    def open_report(self):
        if self.review:
            report = self.review.output / "index.html"
            try:
                os.startfile(report)
            except OSError:
                self.append(f"Không tự mở được bản review. Mở file này trong trình duyệt: {report}")

    def open_job(self):
        if self.cvat:
            webbrowser.open(f"{self.cvat.server}/tasks/{self.cvat.task_id}/jobs/{self.cvat.job_id}")

    def close(self):
        if self.busy:
            self.append("Job đang chạy. Giữ cửa sổ mở đến khi nhận kết quả.")
            return
        self.root.destroy()


if __name__ == "__main__":
    window = tk.Tk()
    JobRunnerUI(window)
    window.mainloop()
