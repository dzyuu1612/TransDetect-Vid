# Revisions needed in `TransDetect-Vid_BaoCao_v14.docx`

Checklist of edits to bring the report back in sync with the current code in
this repo (`main`, commit after "Fix dashboard parameter wiring..."). Grouped
by priority. Each item gives the location, what's wrong, and suggested text.

---

## A. Required — the code changed since v14 was written

### A1. Re-measure Table 4.3 (FPS)

**Why:** the Classical branch in the Dashboard now also runs Lucas–Kanade
optical flow every frame (`goodFeaturesToTrack` + `calcOpticalFlowPyrLK`),
which it did not do when the 6.4–14.6 FPS numbers were recorded. Those numbers
no longer reflect the current code.

**Action:** re-run both `sg(2)` and `sg(3)` on the Dashboard, Classical method,
and record new FPS ranges. The YOLO11 numbers (21.4–25.4) are unaffected and
can stay as-is.

### A2. Add a figure for the motion-vector output — §3.4.1

**Why:** §3.4.1 (Listing 3.3) currently has no figure, unlike §3.2.1 (Figure
3.1) and §3.3.1 (Figure 3.2). The Dashboard now draws Lucas–Kanade arrows, so
there is something to show.

**Where:** insert right after Listing 3.3 discussion, before §3.5.

**Suggested caption (Figure 3.3):**
> Hình 3.3: Trường vector chuyển động Lucas–Kanade giữa hai khung hình liên
> tiếp của sg(2). Mũi tên đỏ nối vị trí điểm đặc trưng ở frame t đến frame
> t+1; chấm xanh là vị trí hiện tại. Điểm được khởi tạo bằng Shi–Tomasi và chỉ
> giữ lại khi status = 1; module ước lượng chuyển động của điểm, chưa gán ID
> cho từng phương tiện.

Renumber: this becomes **Figure 3.3**; all Chapter 4 figure numbers stay the
same (they are still 4.1–4.4).

### A3. Update §3.7 (Dashboard description)

**Why:** §3.7 describes counters and parameters but never states that the
Classical branch on the Dashboard runs Lucas–Kanade. §3.1 already promises
this for the pipeline; the Dashboard just didn't implement it before.

**Action:** add one sentence, e.g.:
> Nhánh truyền thống trên Dashboard cũng chạy Lucas-Kanade Optical Flow trên
> từng cặp khung hình liên tiếp, vẽ vector chuyển động (mũi tên đỏ) đè lên các
> vùng ứng viên, giống hệt luồng CLI ở Mục 3.4.1.

### A4. Update the verification note in Appendix A.4

**Current text (end of A.4):**
> Lưu ý kiểm chứng: notebook công khai hiện còn lưu output từ đường dẫn
> Windows và các lần chạy lỗi do URL/path mẫu, đồng thời data/sample.mp4
> không có trong repo. Trước khi nộp, nhóm cần xóa output cũ, chạy lại toàn bộ
> notebook trên Colab bằng URL và video upload ở trên, rồi lưu output thành
> công; notebook hiện tại chưa đủ làm bằng chứng một phiên Colab tái lập hoàn
> chỉnh.

**Why it's stale:** the notebook has already been cleaned — stale outputs
(Windows paths, failed traceback from `process_yolo`/`YOLOConfig`, which no
longer exist) were removed, the clone URL now points at the real repo, and
all 6 cells match the flow documented in A.4.

**Suggested replacement:**
> Lưu ý kiểm chứng: notebook đã được dọn output cũ và các cell hiện khớp với
> luồng mô tả ở trên (clone repo thật, cài thư viện, tải model, upload video,
> chạy CLI cho cả hai nhánh, khởi chạy Streamlit). Nhóm vẫn cần tự chạy lại
> toàn bộ trên Colab một lần và lưu output thành công để làm bằng chứng một
> phiên tái lập hoàn chỉnh. Lưu ý Cell 6 chỉ khởi chạy Streamlit trong nền
> (`> streamlit.log 2>&1 &`); Colab không tự lộ cổng 8501 ra ngoài, nên truy
> cập từ trình duyệt cần thêm một dịch vụ tunnel (ngrok/localtunnel), không có
> sẵn trong notebook hiện tại.

---

## B. Required — pre-existing mismatches (not caused by recent edits)

### B1. Appendix A.1 folder tree is missing real files

**Current tree** lists only: `requirements.txt`, `README.md`, `main.py`,
`app_streamlit.py`, `train_yolo.py`, `.streamlit/`, `test_videos/`,
`datasets/`, `outputs/`, `notebooks/`, `src/transdetect/`.

**Missing from the actual repo:** `app.py` (a second, older Streamlit UI),
`yolo_detector.py` (a second YOLO module at root, tuple-based API, no vehicle
filter), `run_demo.py`, `legacy/` (frozen teaching drafts of preprocessing /
classical_detector / optical_flow), `pyproject.toml`, `pyrightconfig.json`,
`HUONG_DAN_CAI_DAT.md`, `UI_REQUIREMENTS.md`, `CODE_WALKTHROUGH.md`.

**Action:** add these to the tree, with one line noting that `legacy/`,
`app.py`, `yolo_detector.py` (root), and `run_demo.py` are pre-refactor
prototypes retained for reference only — they are not imported by `main.py`
or `app_streamlit.py` and are out of scope for the Listings in Chapter 3.

### B2. "No images/labels committed" claim in A.1 is only half true

**Current text:** `datasets/README.md — Nguồn Roboflow + giấy phép; KHÔNG
commit ảnh/nhãn`.

**Actual state:** images: correctly excluded (0 image files tracked in git).
Labels: **not** excluded — thousands of Roboflow `.txt` label files are
committed under `datasets/*/`, including a traffic-sign dataset that the
report itself (§1.3, `datasets/README.md`) declares out of scope.

**Two options — pick one:**
- Fix the repo: `git rm -r --cached` the dataset label folders, add
  `datasets/**/labels/` to `.gitignore`, commit. Keep the sentence in the
  report as-is.
- Fix the sentence: change to "chỉ commit `data.yaml` và file mô tả nguồn;
  ảnh và nhãn huấn luyện không nằm trong phạm vi commit chuẩn của repo" — but
  this is less accurate since labels currently *are* committed.

Recommended: fix the repo (first option), since the report's stated intent
("KHÔNG commit ảnh/nhãn") is the correct policy — the repo just hasn't
followed it consistently yet.

### B3. Figure 4.3 caption is stale

**Current caption:**
> Hình 4.3: Nhánh truyền thống trên video sg(3) (1280p), FPS ≈ 14.6. Tổng
> candidate boxes hiển thị (30); các nhãn/confidence trong bảng là của
> Dashboard bản trước khi sửa, đọc là candidate count.

**Why:** the second sentence refers to a since-fixed Dashboard bug and no
longer describes current behavior. In the current code, the Classical
branch's results table always shows all-zero counts for Car/Motorcycle/
Bus/Truck (the branch cannot classify vehicles), and the candidate count only
appears in the "Vùng ứng viên" card and in the exported CSV/JSON.

**Suggested replacement for the second sentence:**
> Bảng kết quả trên Dashboard hiển thị 0 cho cả bốn cột Car/Motorcycle/
> Bus/Truck vì nhánh truyền thống không phân loại được loại xe; số lượng
> thực (30 vùng ứng viên) chỉ xuất hiện ở ô "Vùng ứng viên" và trong file
> CSV/JSON xuất ra (class_name="candidate").

(Re-check this caption against the actual re-captured Figure 4.3 once A1's
FPS re-measurement and new screenshot are done — the box count may differ if
a different frame is captured.)

### B4. Table 3.1 (pipeline steps) is missing filters already described in §3.3

**Current row:** `Contour | Mask/edge map | Bounding boxes | Lọc theo diện
tích tối thiểu`.

**Missing:** the maximum-area filter (`MAX_CONTOUR_AREA = 150000`) and the
two-sided aspect-ratio filter (`0.25 ≤ w/h ≤ 4.0`), both already explained in
prose in §3.3 but omitted from the summary table.

**Suggested replacement:**
> Contour | Mask/edge map | Bounding boxes | Lọc theo diện tích tối thiểu VÀ
> tối đa, cùng tỉ lệ khung hai phía (0,25 ≤ w/h ≤ 4,0)

---

## C. Optional — cleanup, not blocking submission

### C1. `train_yolo.py` has no CLI and hard-codes machine-specific paths

`datasets/README.md` documents `python train_yolo.py --data ... --model-size n
--epochs 50`, but the actual `train_yolo.py` has no `argparse` — the
`__main__` block hard-codes four absolute Windows paths
(`D:\1ComputerVisionProject1\...`), one of which is the traffic-sign dataset
the report declares out of scope. This isn't cited in the report's Listings,
so it doesn't block the docx, but a reviewer who opens the repo will notice
the mismatch with `datasets/README.md`.

If you want this fixed in code (not just noted), ask and it can be converted
to accept `--data`, `--model-size`, `--epochs` like the README already
promises.

### C2. `README.md` still describes the pre-refactor layout

It documents `classical_detector.py`, `preprocessing.py`, `optical_flow.py`
as living at the repo root — they were moved into `src/transdetect/` during
the refactor. Doesn't affect the docx directly, but worth a pass before
final submission since it's the first file anyone opens.

---

## Suggested order of work

1. **A1** (re-run Dashboard, get real FPS numbers) — needs your machine.
2. **A2** (capture the motion-vector screenshot) — same session as A1.
3. Apply **A3, A4, B1, B2 (sentence fix), B3, B4** as text edits in the docx.
4. Decide on **B2**: fix the repo (remove committed labels) vs. fix the
   sentence. Recommended: fix the repo.
5. **C1/C2** — optional, do if time allows.
