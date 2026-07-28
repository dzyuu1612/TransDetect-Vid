# Revisions to `TransDetect-Vid_BaoCao_v14.docx` — status

Tracks what's been applied directly to the live docx
(`C:\Users\dzyuu\Downloads\CVFINAL\TransDetect-Vid_BaoCao_v14.docx`) vs what's
still outstanding. Backups kept alongside it:
`TransDetect-Vid_BaoCao_v14.BEFORE_AUTO_EDIT.docx` (before the first pass),
`TransDetect-Vid_BaoCao_v14.BEFORE_SG3_EDIT.docx` (before the sg(3)-only
rewrite of Chapter 4).

---

## Done

- **A3.** §3.7 now states the Dashboard's Classical branch also runs
  Lucas-Kanade and draws the motion arrows, matching what the code does.
- **A4.** Appendix A.4's verification note rewritten — no longer references
  the stale Colab traceback; notes the notebook is cleaned but still needs
  one real Colab run for evidence, and flags that Cell 6 needs a tunnel
  service to be reachable from a browser.
- **B1.** Appendix A.1's folder tree now lists the previously-missing files
  (`legacy/`, `app.py`, `yolo_detector.py` at root, `run_demo.py`,
  `pyproject.toml`, `pyrightconfig.json`, `HUONG_DAN_CAI_DAT.md`,
  `UI_REQUIREMENTS.md`, `CODE_WALKTHROUGH.md`), with a note that the
  pre-refactor files aren't imported by `main.py`/`app_streamlit.py`.
- **B2.** Fixed at the repo level, not the sentence: removed ~4,540 committed
  dataset label files and the out-of-scope traffic-sign dataset from git
  (commit `0770bd7`). "KHÔNG commit ảnh/nhãn" in A.1 is now literally true.
- **B4.** Table 3.1's Contour row now lists the max-area filter and the
  two-sided aspect-ratio filter, matching the prose in §3.3.
- **Yellow-highlighted "needs update" block** (§4.3 through end of Chapter 5,
  47 runs) — cleared once the content it was flagging got real numbers.

### Chapter 4 restructured to a single demo video (sg(3) only)

This was the resolution for A1 + B3 together, once real Dashboard numbers
for `sg(3)` became available (screenshot: Classical Pipeline, FPS current
9.2 / average 8.9, resolution 1280×720, 21 candidate boxes at frame 122,
source frame rate 59.94 FPS, duration 00:00:13):

- Removed old Figure 4.1 and 4.2 (both `sg(2)`, no re-measurement available
  for that video) — image + caption paragraphs deleted.
- Renumbered old Figure 4.3 → **Figure 4.1** (Classical, `sg(3)`) and old
  Figure 4.4 → **Figure 4.2** (YOLO11, `sg(3)`).
- Figure 4.1's caption rewritten with the real measured numbers above,
  replacing the `[CẦN ĐO LẠI...]` placeholder and the stale "Dashboard bản
  trước khi sửa" sentence.
- Table 4.1 (`Bảng 4.1`): removed the `sg(2)` row; `sg(3)` row's frame rate
  updated from the earlier estimate ("≈30 FPS, metadata UI") to the real
  value read off the Dashboard ("59,94 FPS").
- Table 4.3 (`Bảng 4.3`): Classical FPS range `6,4–14,6` → `8,9–9,2` (now a
  single video, and Classical is slower than before because Lucas-Kanade
  now runs every frame on the Dashboard too — see A3). YOLO11 range
  `21,4–25,4` → `25,4` (only `sg(3)`'s value remains relevant).
- §4.1 and the paragraph before Table 4.3 reworded from "hai video" (two
  videos) to "một video" (one video) accordingly.
- Chapter 3's Figure 3.1 (§3.2.1, illustrates preprocessing on an `sg(2)`
  frame) was **not** touched — that's a different, unrelated figure (a
  Chapter 3 pedagogy illustration, not a Chapter 4 experimental result), and
  the specific source video doesn't matter for its purpose.

**If this reading of "giữ 1 cái sg3 thôi" was wrong** (e.g. you actually
wanted `sg(2)` re-measured and kept, not dropped), the pre-edit state is in
`TransDetect-Vid_BaoCao_v14.BEFORE_SG3_EDIT.docx` — say so and it gets
restored from there.

### Shi-Tomasi mentions — kept, simplified where it seemed to matter

The three Shi-Tomasi mentions in the docx (§2.5.2 theory, §3.4.1 ×2) were
**not removed** — `cv2.goodFeaturesToTrack` (Shi-Tomasi corner detection) is
literally what the code calls to pick which points Lucas-Kanade tracks, so
removing it would make the report describe an incomplete algorithm. Instead,
`CODE_WALKTHROUGH.md`'s explanation (which used denser linear-algebra
language, "AᵀA gần suy biến") was rewritten in plain terms — what a "corner"
is, why a flat region or a single straight edge isn't trackable, why Lucas-
Kanade needs corners specifically. If the docx's own three mentions still
feel too dense, say which one and it can get the same simplification.

---

### Motion-arrow drawing rule documented in §3.4.1

Added a sentence after the Listing 3.3 discussion explaining the display
convention, because a reader looking at the figure will see ~20 arrows over
~100 dots and reasonably wonder why: every tracked point draws a blue dot,
but only points displaced ≥1 px draw a red arrow. Measured on the real
sg(3) video, ~80/100 points sit on static background with mean displacement
0.67 px (sub-pixel noise), so arrowing them would render noise as motion.
Arrow length is fixed for visibility after the UI downscale; direction is
always the true measured (u,v).

---

## Outstanding

### Caption numbers in Figure 4.1 may not match your final screenshot

Figure 4.1's caption currently reads "FPS ≈ 8,9–9,2" and "21 candidate
boxes", taken from the earlier screenshot (frame 122). Your most recent
screenshot shows FPS 9.0/9.4 and 25 candidate boxes at frame 18 — different
frame, so different counts, which is expected. **Whichever screenshot you
finally insert, check these two numbers in the caption match it**, or tell
me the numbers and I'll update the caption.

### Insert Figure 3.3 — Lucas-Kanade motion vectors (§3.4.1)

You already have a real screenshot (Dashboard, Classical Pipeline, `sg(3)`,
visible red arrows + blue dots on the moving vehicles). It hasn't been
inserted yet because it only exists as a pasted image in chat — there's no
file path on disk to embed. Save it to a file (e.g.
`d:\CV\sg3_dashboard.png` or anywhere in `Downloads`) and give the path;
insertion point is right after §3.4.1's Listing 3.3 discussion, before §3.5,
as:

> Hình 3.3: Trường vector chuyển động Lucas–Kanade trên video sg(3). Mũi tên
> đỏ nối vị trí điểm đặc trưng ở frame trước đến frame hiện tại (đã phóng
> đại độ dài mũi tên ×5 chỉ để dễ nhìn, xem `display_scale` ở
> `visualization.py`); chấm xanh là vị trí hiện tại. Điểm được khởi tạo
> bằng Shi–Tomasi và chỉ giữ lại khi status = 1; module ước lượng chuyển
> động của điểm, chưa gán ID cho từng phương tiện.

### Docx currently open in Word

The file was locked (`~$...docx` present) during this session; edits only
went through after it was closed. If you reopen it in Word, close it again
before asking for further automated edits, or the next save will fail the
same way (harmlessly — nothing gets corrupted, the save just doesn't happen).

### C1/C2 — still optional, unchanged from before

- `train_yolo.py` has no CLI (`argparse`) despite `datasets/README.md`
  documenting `--data`/`--model-size`/`--epochs` flags, and hard-codes
  machine-specific paths including the out-of-scope traffic-sign dataset.
- `README.md` still describes the pre-refactor file layout (root-level
  `classical_detector.py` etc.).

Neither blocks the docx; fix if there's time.
