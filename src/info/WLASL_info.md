# WLASL dataset reference

What the WLASL data looks like, and what `src/preprocess.py` did to it. Notebooks should link here
instead of repeating any of this.

## From the WLASL GitHub page

Data Description
-----------------
* `gloss`: *str*, data file is structured/categorised based on sign gloss, or namely, labels.
* `bbox`: *[int]*, bounding box detected using YOLOv3 of (xmin, ymin, xmax, ymax) convention. Following OpenCV convention, (0, 0) is the up-left corner.
* `fps`: *int*, frame rate (=25) used to decode the video as in the paper.
* `frame_start`: *int*, the starting frame of the gloss in the video (decoding
with FPS=25), *indexed from 1*.
* `frame_end`: *int*, the ending frame of the gloss in the video (decoding with FPS=25). -1 indicates the gloss ends at the last frame of the video.
* `instance_id`: *int*, id of the instance in the same class/gloss.
* `signer_id`: *int*, id of the signer.
* `source`: *str*, a string identifier for the source site.
* `split`: *str*, indicates sample belongs to which subset.
* `url`: *str*, used for video downloading.
* `variation_id`: *int*, id for dialect (indexed from 0).
* `video_id`: *str*, a unique video identifier.

## Our copy of the data

Checked against `data/WLASL/splits/asl2000.json` and `data/WLASL/WLASL2000/` on 2026-09-26.

* 21095 instances, 2000 glosses, 119 signers.
* The videos come pre-cut from the original YouTube videos, so each `video_id` is one instance.
  No `video_id` appears twice.
* Clips are 256x256 at 25 fps (all of a random sample of 400).
* `frame_end = -1` never occurs, despite the description above.
* `frame_start` is 1 for 20791 instances. Most of the rest are offsets into the original,
  uncut video (see [frame-range resets](#frame-range-resets)).

### Gotchas

* **There is a gloss called `empty`.** Don't use `"empty"` as a placeholder or sentinel gloss
  name. `stats.reverse_preproc_format` did, and silently dropped 6 instances of it.
* **The label files on disk are off by one frame; a rerun is due.** WLASL's `frame_start` is
  1-based, but the current labels store it as-is, and `utils.cv_load` returns
  `frames[frame_start:frame_end]` (0-based). So each of the 20799 asl2000 instances whose range
  wasn't reset skips its first annotated frame, while the 296 reset instances (reset to 0-based
  `0` to frame count) keep every frame. Lengths (`frame_end - frame_start`) are one short too, so
  the current `cutoff_9` splits remove clips of 10 or fewer real frames. `preprocess.py` now
  converts starts to 0-based (cache version 2), but the labels have deliberately not been
  regenerated, so every experiment in progress uses the same inputs. Regenerate them before the
  final round of results (tracked in `src/TODO.md`); see
  [what the rerun will change](#what-the-rerun-will-change).
* Every `frame_end` in the kept data is within its clip: 20725 instances end exactly at the last
  frame, and 74 are annotated shorter than the clip (by up to 91 frames).

## Naming conventions

### Split vs Set

For the naming of different functions, 'set' and 'split' can somtimes be used interchangibly to mean different things, which can be confusing. So for all code written by me,
* **SPLIT**: A split of WLASL, one of asl100, asl300, asl1000 and asl2000
* **SET**: A subset of a given wlasl split, one of train, val test

## Preprocessing

The numbers below describe the current label files (1-based starts, see [Gotchas](#gotchas)), and
come from each split's
`data/WLASL/preprocessed/labels/<split>/preprocess_log.json`, which lists every instance that was
reset or removed and why. Those logs are the source of truth: if you rerun `preprocess.py`,
update this section from them.

Settings: `--strictness reset reset` (bad frame ranges and bboxes are reset rather than removed),
YOLOv8n bboxes, and `--length_cutoff` 0 or 9.

| Split | Train | Val | Test | Total | Frame-range resets | Removed by `cutoff_9` |
|---|---|---|---|---|---|---|
| asl100 | 1442 | 338 | 258 | 2038 | 52 | 0 |
| asl300 | 3549 | 901 | 668 | 5118 | 124 | 0 |
| asl1000 | 8978 | 2320 | 1876 | 13174 | 251 | 2 |
| asl2000 | 14296 | 3920 | 2879 | 21095 | 296 | 3 |

Counts are before the cutoff. Nothing else was removed: every video could be opened, and YOLO
found a person in every video, so no bbox fell back to the whole frame.

`asl100_bottom` and `asl100_worst` are not made by `preprocess.py`, so they have no log.

### Frame-range resets

A frame range is reset to the whole clip (`0` to its frame count) when its start frame is past
the end of the clip, or its end frame is impossible. Of the 296 resets in asl2000:

* 295 are `northtexas` clips where the annotation gives frame numbers in the original, uncut
  video (e.g. `70266` "book": 3732-3852), but covers exactly the length of the pre-cut clip. For
  these the reset loses nothing.
* 1, `70053` "out", is annotated as 8591-8653 (63 frames), but its clip has 247 frames. Its reset
  uses the whole clip, so this one sample may include frames outside the sign.

### Removed by `cutoff_9`

Samples with 9 or fewer frames by the old, off-by-one length calculation (see
[Gotchas](#gotchas)):

| Video id | Gloss | Set | Frames (annotated) | In splits |
|---|---|---|---|---|
| 18223 | earring | train | 1-10 | asl1000, asl2000 |
| 59958 | turkey | val | 1-10 | asl1000, asl2000 |
| 15144 | deduct | val | 1-9 | asl2000 |

### What the rerun will change

Previewed on 2026-09-26 by running the new frame-range code on the raw splits against the real
videos (no labels written):

* The same 296 instances are reset, to the same ranges.
* Every other `frame_start` goes down by one (1 to 0 for 20791 asl2000 instances), so each clip
  gains its first annotated frame.
* `cutoff_9` keeps `18223` and `59958` (10 frames each), and removes only `15144` "deduct"
  (9 frames), so asl1000_cutoff_9 loses nothing and asl2000_cutoff_9 loses 1.
* Bboxes are recomputed by YOLO over the corrected ranges, because the version-2 cache starts
  empty, so they may shift slightly.
