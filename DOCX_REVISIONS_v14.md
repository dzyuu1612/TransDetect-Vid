# Revisions to `TransDetect-Vid_BaoCao_v14.docx` — status

Tracks what's been applied directly to the live docx
(`C:\Users\dzyuu\Downloads\CVFINAL\TransDetect-Vid_BaoCao_v14.docx`). This is
the **current, accurate snapshot** — earlier revisions of this file described
intermediate states (e.g. Figure 4.1 at frame 122 with FPS 8,9–9,2) that have
since been superseded by further rounds of fixes; only what's below still
reflects the live document.

Backups on disk, in chronological order:
`...BEFORE_AUTO_EDIT.docx` → `...BEFORE_SG3_EDIT.docx` →
`...BEFORE_ARROW_NOTE.docx` → `...BEFORE_IMG_SWAP.docx` →
`...BEFORE_RECT_FIX.docx` → `...BEFORE_TOC_UPDATE.docx` →
`...BEFORE_FINAL_FIX.docx` (most recent, right before today's final round).

---

## Done

### Repo-level (kept for context, not part of today's docx-only rounds)

- **`legacy/` deleted** — nothing imported it, and it implemented the
  algorithms *differently* from `src/transdetect/` (`MORPH_CLOSE` vs
  `dilate`, `RETR_EXTERNAL` vs `RETR_LIST`, `(x,y,w,h)` vs `xyxy`, no
  max-area/aspect filter, Sobel default 50 vs 40). Recoverable via
  `git log -- legacy/`.
- **`datasets/` kept**, docs-only (~23 KB) — carries the CC BY 4.0
  attribution for the three Roboflow datasets referenced by §3.5.3.
- **~4,540 committed dataset label files removed** from git, plus the
  out-of-scope traffic-sign dataset. "KHÔNG commit ảnh/nhãn" in Appendix
  A.1 is now literally true.

### YOLO11 architecture claims — §2.6.1, §2.6.4, §3.5.2

Verified by direct measurement before touching the text: fed a 1280×720
frame to `model.predict()` with no `rect` argument (matching what
`yolo_detector.py` actually does — it never passes `rect` or `imgsz`), and
hooked the model's forward call. Actual tensor: **[1, 3, 384, 640]**, not
640×640. Corresponding grid count: 80×48 + 40×24 + 20×12 = **5040**
positions, not 8400.

- 640×640 / 8400 is now framed explicitly as the **illustrative** case
  (kept, per your instruction, since it's still the clean number for
  explaining P3/P4/P5).
  1280×720 / 640×384 / 5040 is presented as what the project's own test
  video actually produces, measured directly — not asserted.
- Table 2.3's Detect row already hedged correctly ("số vị trí thay đổi nếu
  tensor đầu vào là hình chữ nhật") — left as-is, no change needed there.

### Lucas-Kanade over-claims — §2.5, §3.4, §3.4.1, §3.7 (×2), §1.3

- §2.5 and §3.4: removed the implication that optical flow output is used to
  decide whether a candidate box is "real" motion vs static noise. The code
  never associates points to boxes — reworded to state that plainly.
- §3.4.1: removed the "80/100 points, mean 0.67 px" figure — no committed
  script/log backs it as a reproducible number (the measurement was a
  one-off diagnostic in a chat session, not checked into the repo).
  Replaced with a generic, honest statement: the 1px display threshold is a
  prototype visualization parameter, not a validated/optimized one.
- §3.7 (both the Dashboard description and the "Theo dõi tiến trình
  Real-time" bullet): clarified that points/arrows and candidate boxes are
  only drawn on the same frame, not assigned to each other, and that all
  Dashboard counters are per-frame, not unique-vehicle totals.
- §1.3: added an explicit paragraph — no multi-object tracking, no ID
  persistence, counts are per-frame detections/candidates only.

### Section 1.1 / 1.2 — scope wording

- §1.1: "phát hiện và đếm phương tiện" → "phát hiện, phân loại và thống kê
  số detection phương tiện theo từng frame... chưa duy trì ID nên không
  đếm tổng số phương tiện duy nhất."
- §1.2: "so sánh theo FPS và độ chính xác tương đối" → "so sánh theo FPS
  quan sát và đánh giá định tính; chưa có ground truth nên chưa tính
  Precision/Recall/mAP."

### Figure 4.1 (Classical) and Figure 4.2 (YOLO11) — both re-captured live

Both driven through the actual running Dashboard (Playwright automation),
on the same real uploaded video (`duong_pho_sg(3).mp4`, 1280×720, 59.94 FPS
source), **CPU only — no GPU available on this machine**:

- **Figure 4.1** (Classical + Lucas-Kanade): frame 74, headless-browser
  session, FPS current 14.4 / average 14.8, 20 candidate boxes. Caption
  states explicitly this is a headless capture and that a normal
  interactive-browser session on the same machine/video reads lower
  (~8.9–9.4 FPS) — the two conditions are not conflated.
- **Figure 4.2** (YOLO11n): frame 83, same headless conditions, FPS current
  15.3 / average 14.9, 25 total detections (Car 6, Motorcycle 18, Bus 0,
  Truck 1), all confidences ≥ 0.31 (above the 0.25 threshold — addresses the
  earlier concern that the old Figure 4.2 might have shown sub-threshold
  boxes). Caption no longer claims `imgsz=640` is a project default — the
  code only passes `conf`, `iou`, `max_det`, `device`, `verbose` to
  `model.predict()`; `imgsz=640` is Ultralytics' own default, stated as such.

### Table 4.3 — rebuilt as a 5-column table, no more merged "ranges"

Old table conflated **Current FPS** and **Average FPS** (two different
metrics from the *same* single run) into a fake "8,9–14,8" min–max range, as
if they were repeated-trial statistics. Rebuilt with separate columns:

| Phương pháp | Chế độ đo | FPS tức thời | Average FPS | Ghi chú |
|---|---|---|---|---|
| Threshold + Sobel + contour + Lucas-Kanade | CPU laptop, headless, frame 74 | 14,4 | 14,8 | Số quan sát từ Hình 4.1 |
| YOLO11n | CPU laptop, headless, frame 83 | 15,3 | 14,9 | Số quan sát từ Hình 4.2 |

Caption renamed to "FPS quan sát trên Dashboard trong các phiên chạy được
ghi lại; đây không phải benchmark thuật toán." A note is inserted right
below the table restating that these are end-to-end demo-session numbers
(video read + inference + draw + UI update), not pure inference benchmarks.
The old GPU number (25.4 FPS) is **no longer in the table** — it's
mentioned in the preceding paragraph as a prior team measurement whose GPU
(RTX 3060 vs Colab T4) was never confirmed, so it isn't used for comparison.

### Appendix A.4 — rewritten twice this session, now matches the real notebook code

First pass matched the notebook's 6 markdown sections (clone → deps →
model → video → run → compare). Second pass went further: the two
`!python main.py --input INPUT_VIDEO ...` placeholder commands were replaced
with the **actual** `subprocess.run([...])` Python calls the notebook uses
(including `--conf 0.25 --iou 0.45 --max-det 300`, which the placeholder
commands omitted). Also added a sentence citing the exact commit being
cross-referenced (`6920c6ac879c8e1c93a662a8dca2dda965c45563`) at the start
of §4.1.

**Self-caught bug:** this rewrite deleted the manual page-break that used to
sit before "TÀI LIỆU THAM KHẢO" (it lived on a paragraph that got replaced).
Restored via `page_break_before = True` on that heading directly.

### Style bug fixed: Listing captions were polluting the table-of-tables

All 4 `Listing 3.x` captions used the same Word style as table captions
(`Caption Bang`), so "Danh mục bảng biểu" would have listed Listings as if
they were tables once refreshed. Created a dedicated `Caption Listing`
style and reassigned all 4 — confirmed Danh mục bảng biểu now shows exactly
the 10 real tables, nothing else.

### Mục lục / Danh mục hình ảnh / Danh mục bảng biểu — refreshed via Word COM automation

python-docx can't compute real pagination (needs Word's layout engine), so
this repo's copy of Microsoft Word was driven via COM automation
(PowerShell `New-Object -ComObject Word.Application`) to open the file,
call `Fields.Update()` + `TablesOfContents.Item(n).Update()` for all 3 TOC
fields, and save — equivalent to `Ctrl+A → F9 → Update entire table`, done
without you touching Word.

**Root cause found and fixed along the way:** the two list-fields' raw field
codes read `\t "CaptionHinh,1"` / `\t "CaptionBang,1"` (no space), but the
actual paragraph styles are named `"Caption Hinh"` / `"Caption Bang"` (with
a space) — Word matches by literal display name, so the mismatch meant
these two lists always resolved to "No table of contents entries found.",
independent of whether they'd ever been refreshed. Fixed the field code text
directly, then re-ran the update. Confirmed final state: Mục lục has all
chapter/section entries with real page numbers; Danh mục hình ảnh lists all
5 figures; Danh mục bảng biểu lists all 10 tables (and no Listings).

---

## Explicitly dropped for this round (your call)

### Layout pass (blank pages, orphan lines, page breaks around chapter starts)

Word COM automation on this machine hung or ran extremely slowly for
anything beyond a plain field update — page-by-page scanning
(`Repaginate()` + `GoTo` loop) and PDF export (`ExportAsFixedFormat`,
`SaveAs2`) each either timed out or took several minutes with no output,
across three separate attempts. A static XML scan (checking every
page-break point, including table content, for suspiciously short gaps)
found **no structural double-break issues** — so if blank pages remain,
they're a product of Word's actual page layout (font metrics, image
sizing), not something visible in the raw document structure.

You said to drop this and move on. If you want it revisited later: open the
file yourself in Word, note the actual current page numbers of any blank or
orphan-line pages you see, and that gives something concrete to fix instead
of me guessing blindly.

---

## Still open

- **Figure 3.3 (Lucas-Kanade motion vectors, §3.4.1)** — you have a real
  screenshot with visible arrows, but it only exists as a pasted chat image
  with no file path on disk, so it can't be embedded. Save it to a file
  (e.g. `d:\CV\sg3_dashboard.png`) and give the path.
- **`train_yolo.py`** still has no CLI (`argparse`) despite
  `datasets/README.md` documenting `--data`/`--model-size`/`--epochs`, and
  hard-codes machine-specific paths including the out-of-scope traffic-sign
  dataset. Not part of any docx-only round so far — still just noted.
- **`README.md`** still describes the pre-refactor file layout.

Neither of the last two blocks the docx; fix if there's time.
