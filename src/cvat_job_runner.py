"""Local Windows UI for running YOLO and SegFormer on a CVAT job."""

from collections import Counter
from datetime import datetime
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

from cvat_bbox_agent import AgentError, CVAT, JOB_ID, ORGANIZATION, ROOT, SERVER, TASK_ID, TOKEN_FILE, check_assignee
from customer_guideline import load_guideline
from local_annotation_review import PALETTE, generate_review
from publish_cvat_review import label_map, prepare, publish


DEFAULT_JOB = f"{SERVER}/tasks/{TASK_ID}/jobs/{JOB_ID}"
MODES = {"BBox và segmentation": "both", "Chỉ Bounding Box": "bbox", "Chỉ segmentation": "segmentation"}
NOTES = {"skip_existing_segmentation": "Frame đã có segmentation khác",
         "skip_cross_label_bbox": "BBox trùng vị trí, khác nhãn",
         "skip_missing_truncated": "BBox cần thuộc tính truncated"}


def parse_job_url(value):
    parts = urlsplit(value.strip())
    route = re.fullmatch(r"/tasks/([1-9]\d*)/jobs/([1-9]\d*)/?", parts.path)
    if (parts.scheme != "https" or not re.fullmatch(r"[A-Za-z0-9.:-]+", parts.hostname or "")
            or parts.username is not None or not route or parts.query or parts.fragment):
        raise ValueError("Nhập link HTTPS dạng https://may-chu/tasks/75/jobs/222.")
    # Accessing .port validates any explicit port.
    _ = parts.port
    task_id, job_id = map(int, route.groups())
    return f"https://{parts.netloc.lower()}", task_id, job_id


def job_images(cvat, frames):
    for number, info in frames.items():
        data, mime = cvat.api(f"/api/jobs/{cvat.job_id}/data",
                             params={"type": "frame", "number": number, "quality": "original"}, image=True)
        if mime not in {"image/jpeg", "image/png", "image/webp"}:
            raise AgentError(f"Frame {number} trả định dạng {mime}; cần job ảnh 2D.")
        yield number, info["name"], mime, data


def run_job(cvat, mode, write, output, emit, guideline=None):
    job = cvat.job()
    if write:
        check_assignee(cvat, job)
    frames = cvat.frames(job)
    if not frames:
        raise AgentError("Job không có frame ảnh.")
    label_map(cvat, mode, guideline)
    if guideline:
        emit("log", f"Guideline {guideline.source} · {len(guideline.bbox)} ánh xạ BBox · "
                    f"{len(guideline.segmentation)} ánh xạ segmentation.")
    identity = {"server": cvat.server, "task": cvat.task_id, "job": cvat.job_id,
                "organization": cvat.organization}
    profile = (cvat.server, cvat.task_id, cvat.job_id) == (SERVER, TASK_ID, JOB_ID)
    emit("phase", ("Đang gán nhãn trên máy", len(frames)))

    def on_frame(message):
        emit("log", message)
        emit("step", None)

    generate_review(job_images(cvat, frames), output, job_222=profile, progress=on_frame,
                    destination=identity, guideline=guideline)
    emit("report", output / "index.html")
    if not write:
        return f"Đã xuất bản xem trước cho {len(frames)} frame."
    emit("phase", ("Đang đối chiếu và lưu lên CVAT", 0))
    fresh_job = cvat.job()
    check_assignee(cvat, fresh_job)
    endpoint = f"/api/jobs/{cvat.job_id}/annotations/"
    current = cvat.api(endpoint)
    add, update, notes = prepare(output, cvat.frames(fresh_job), label_map(cvat, mode, guideline),
                                 current, mode, identity, guideline)
    for (number, reason), count in sorted(Counter(notes).items()):
        emit("log", f"Frame {number} · {NOTES[reason]} ({count}).")
    totals = publish(cvat, endpoint, current, add, update, progress=lambda text: emit("log", text))
    counts = Counter(s["type"] for s in add)
    emit("log", f"Đọc lại CVAT: {totals.get('rectangle', 0)} BBox và {totals.get('mask', 0)} mask trong job.")
    return (f"Đã xử lý {len(frames)} frame · thêm {counts['rectangle']} BBox và {counts['mask']} mask · "
            f"sửa {len(update)} hộp · {len(notes)} mục cần rà.")


class JobRunnerUI:
    def __init__(self, root):
        self.root = root
        self.events = Queue()
        self.busy = False
        self.report = None
        self.job_url = None
        root.title("CVAT Job Runner")
        root.geometry("790x625")
        root.minsize(680, 510)
        root.configure(background="#edf1f6")
        root.protocol("WM_DELETE_WINDOW", self.close)

        style = ttk.Style(root)
        style.configure("Title.TLabel", font=("Inter", 22, "bold"))
        style.configure("Primary.TButton", padding=(20, 8))
        style.configure("Status.TLabel", padding=(0, 8), foreground="#234369")

        panel = ttk.Frame(root, padding=22)
        panel.pack(expand=True, fill="both", padx=18, pady=18)
        panel.columnconfigure(1, weight=1)
        panel.rowconfigure(10, weight=1)
        ttk.Label(panel, text="Chạy job CVAT", style="Title.TLabel").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 16))

        self.url = tk.StringVar(value=DEFAULT_JOB)
        self.token = tk.StringVar()
        self.org = tk.StringVar(value=ORGANIZATION)
        self.mode = tk.StringVar(value=next(iter(MODES)))
        self.write = tk.BooleanVar(value=True)
        self.guideline = tk.StringVar()
        self.inputs = []
        for row, label, var, secret in ((1, "Link job", self.url, False), (2, "Access token CVAT", self.token, True),
                                         (3, "Organization", self.org, False)):
            ttk.Label(panel, text=label).grid(row=row, column=0, sticky="w", padx=(0, 14), pady=4)
            entry = ttk.Entry(panel, textvariable=var, show="●" if secret else "")
            entry.grid(row=row, column=1, sticky="ew", pady=4)
            self.inputs.append((entry, "normal"))
        ttk.Label(panel, text="Guideline khách hàng").grid(row=4, column=0, sticky="w", pady=4)
        file_row = ttk.Frame(panel)
        file_row.grid(row=4, column=1, sticky="ew", pady=4)
        guideline_entry = ttk.Entry(file_row, textvariable=self.guideline)
        guideline_entry.pack(side="left", fill="x", expand=True)
        browse = ttk.Button(file_row, text="Chọn file .md", command=self.choose_guideline)
        browse.pack(side="right", padx=(8, 0))
        self.inputs.extend(((guideline_entry, "normal"), (browse, "normal")))
        ttk.Label(panel, text="Phạm vi lưu").grid(row=5, column=0, sticky="w", pady=4)
        combo = ttk.Combobox(panel, values=list(MODES), textvariable=self.mode, state="readonly")
        combo.grid(row=5, column=1, sticky="ew", pady=4)
        self.inputs.append((combo, "readonly"))
        checkbox = ttk.Checkbutton(panel, text="Lưu annotation lên CVAT", variable=self.write)
        checkbox.grid(row=6, column=1, sticky="w", pady=4)
        self.inputs.append((checkbox, "normal"))
        if TOKEN_FILE.is_file():
            ttk.Label(panel, text="Có thể để trống token khi dùng CSV đã có cho CVAT Edufun.").grid(
                row=7, column=1, sticky="w", pady=(0, 5))

        self.status = ttk.Label(panel, text="Sẵn sàng", style="Status.TLabel", wraplength=640)
        self.status.grid(row=8, column=0, columnspan=2, sticky="ew")
        self.bar = ttk.Progressbar(panel, mode="determinate", maximum=1)
        self.bar.grid(row=9, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        self.log = ScrolledText(panel, height=9, wrap="word", state="disabled", font="TkDefaultFont",
                                background="#f6f8fb", foreground="#19304a", relief="flat", padx=8, pady=8)
        self.log.grid(row=10, column=0, columnspan=2, sticky="nsew")
        controls = ttk.Frame(panel)
        controls.grid(row=11, column=0, columnspan=2, sticky="ew", pady=(14, 0))
        self.run_button = ttk.Button(controls, text="Chạy", style="Primary.TButton", command=self.start)
        self.run_button.pack(side="left")
        self.report_button = ttk.Button(controls, text="Mở bản review", state="disabled", command=self.open_report)
        self.report_button.pack(side="left", padx=8)
        self.cvat_button = ttk.Button(controls, text="Mở job CVAT", state="disabled", command=self.open_job)
        self.cvat_button.pack(side="right")
        root.after(100, self.poll)

    def choose_guideline(self):
        path = filedialog.askopenfilename(parent=self.root, filetypes=[("Guideline Markdown", "*.md")])
        if path:
            self.guideline.set(path)

    def start(self):
        if self.busy:
            return
        try:
            server, task_id, job_id = parse_job_url(self.url.get())
            token = self.token.get().strip()
            if token and (not token.isascii() or any(c.isspace() or not c.isprintable() for c in token)):
                raise ValueError("Access token phải là một dòng, không có dấu cách.")
            if not token and server != SERVER:
                raise ValueError("Nhập token cho máy chủ CVAT này.")
            if not token and not TOKEN_FILE.is_file():
                raise ValueError("Nhập token CVAT hoặc đặt tệp config/access_tokens.csv.")
            mode = MODES[self.mode.get()]
            guideline_path = self.guideline.get().strip()
            guideline = load_guideline(guideline_path, PALETTE) if guideline_path else None
        except (ValueError, OSError) as exc:
            self.status.configure(text=str(exc))
            return

        self.busy = True
        self.report = None
        self.job_url = f"{server}/tasks/{task_id}/jobs/{job_id}"
        self.report_button.configure(state="disabled")
        self.cvat_button.configure(state="normal")
        self.run_button.configure(state="disabled")
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        organization = self.org.get().strip()
        write = self.write.get()
        self.token.set("")
        for control, _ in self.inputs:
            control.configure(state="disabled")
        self.phase("Đang kiểm tra CVAT", 0)
        suffix = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:6]
        output = ROOT / "runs" / f"task-{task_id}-job-{job_id}-{suffix}"

        def work():
            def emit(event, value):
                self.events.put((event, value))
            try:
                cvat = CVAT(server=server, task_id=task_id, job_id=job_id, organization=organization, token=token or None)
                emit("done", run_job(cvat, mode, write, output, emit, guideline))
            except Exception as exc:
                text = str(exc) or type(exc).__name__
                emit("error", text.replace(token, "[token]") if token else text)

        Thread(target=work, daemon=True).start()

    def phase(self, text, total):
        self.status.configure(text=text)
        self.bar.stop()
        self.bar.configure(mode="determinate" if total else "indeterminate", maximum=total or 100, value=0)
        if not total:
            self.bar.start(13)

    def append(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def poll(self):
        try:
            while True:
                event, value = self.events.get_nowait()
                if event == "phase":
                    self.phase(*value)
                elif event == "log":
                    self.append(value)
                elif event == "step":
                    self.bar.configure(value=self.bar["value"] + 1)
                elif event == "report":
                    self.report = value
                    self.report_button.configure(state="normal")
                elif event in {"done", "error"}:
                    self.bar.stop()
                    self.bar.configure(mode="determinate", maximum=1, value=int(event == "done"))
                    self.status.configure(text="Hoàn thành · " + value if event == "done" else "Lỗi · Xem chi tiết bên dưới")
                    self.append(value)
                    self.busy = False
                    self.run_button.configure(state="normal")
                    for control, state in self.inputs:
                        control.configure(state=state)
        except Empty:
            pass
        self.root.after(100, self.poll)

    def open_report(self):
        if self.report:
            webbrowser.open(self.report.as_uri())

    def open_job(self):
        if self.job_url:
            webbrowser.open(self.job_url)

    def close(self):
        if self.busy:
            self.append("Job đang chạy. Giữ cửa sổ mở đến khi nhận kết quả.")
            return
        self.root.destroy()


if __name__ == "__main__":
    window = tk.Tk()
    JobRunnerUI(window)
    window.mainloop()
