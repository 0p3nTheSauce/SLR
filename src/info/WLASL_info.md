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

**Source.** The YouTube download scripts no longer work reliably, so, following the WLASL
instructions, the data was requested from the WLASL authors. They supplied `WLASL2000.zip` (the
videos), `splits.zip` (the annotation JSONs) and `pose_per_individual_videos.zip`.

Checked against `data/WLASL/splits/asl2000.json` and `data/WLASL/WLASL2000/` on 2026-09-26.

* 21095 instances, 2000 glosses, 119 signers. The paper reports 21,083 videos, and per subset
  2,038 / 5,117 / 13,168 / 21,083; our annotation file has 2,038 / 5,118 / 13,174 / 21,095. The
  annotations have evidently changed since publication, for reasons we don't know.
* The supplied videos were already preprocessed by the authors: each is cut from its original
  YouTube video and resized to 256x256, so each `video_id` is one instance, and all 21095 are
  present and readable. No `video_id` appears twice. Evidence that the cutting and resizing came
  after annotation: every raw `bbox` extends past 256 px (up to 492), so bboxes are in the
  original videos' coordinates, and some `northtexas` frame ranges are offsets into the original
  video (see [frame-range resets](#frame-range-resets)).
* Clips are 256x256 at 25 fps (all of a random sample of 400), and the shortest has 9 frames.
* The original WLASL loader (`code/I3D/datasets/nslt_dataset.py`) skips videos whose file has
  fewer than 9 frames, so it removes none of these clips. Our `cutoff_9` splits are stricter.
* `frame_end = -1` never occurs, despite the description above.
* `frame_start` is 1 for 20791 instances. Most of the rest are offsets into the original,
  uncut video (see [frame-range resets](#frame-range-resets)).

### Are the clips pre-cut?

Almost entirely, yes: for 21014 of the 21095 instances the annotated frame range changes nothing.
Computed on 2026-10-05 by
[`frame_ranges.ipynb`](../results/dataset_analysis/frame_ranges.ipynb), which compares each
raw annotation with the frame count OpenCV decodes from its video:

| Annotated range vs clip | Instances |
|---|---|
| Whole clip (start 1, end at the last frame) | 20718 |
| Outside the clip, so preprocessing resets it to the whole clip | 296 |
| Trims 1-2 frames off the start (`handspeak` x2, `aslpro`, `signschool`, `spreadthesign`, `asllex`) | 6 |
| Trims real footage: 74 `lillybauer` clips, plus `69512` "today" (`aslbrick`, 12 frames) | 75 |

* The 75 real trims cut a lot of footage: `lillybauer` trims 24-108 frames (median 64), and the
  median clip is only 52% annotated. 73 end early, `68722` "water" starts 73 frames late, and
  `68770` "now" does both (frames 49-116 of 175).
* The cut frames are often not rest: `lillybauer` clips seem to show the sign twice, and the
  annotation picks one repetition. Checked by eye (2026-10-05) on 5 of the 74: in `68770` "now",
  `68560` "throw", `68372` "when" and `68722` "water", the cut frames show the same sign again
  (in "now", the annotated repetition has a facial expression and the cut one doesn't);
  `69022` "morning" was unclear from thumbnails. Not checked across all 74.
* So the frame ranges matter for only 75 instances, but for those they change what the model
  sees a lot (often one repetition of the sign instead of two). Keep them.
* `CAP_PROP_FRAME_COUNT` (what `preprocess.fix_bad_frame_range` reads, rather than decoding)
  equals the decoded frame count for all 21095 videos, so the frame counts preprocessing
  checks against are exact.

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

### One sign, several glosses (e.g. `before` / `former` / `past`)

WLASL is labelled by English gloss, and the gloss-to-sign mapping isn't one-to-one:

* **One gloss, several signs.** `variation_id` numbers the distinct sign forms filed under a
  gloss. `before` has two: variation 0 is a backward movement over the shoulder, variation 1 is
  one flat hand moving back from the other.
* **One sign, several glosses.** Lifeprint glosses the over-the-shoulder sign as
  "PAST | PAST-[before] | PAST-[former]" ([before][lp-before], [past][lp-past]), and the MSU ASL
  Browser describes [FORMER][msu-former] with the same movement. So `before` (variation 0),
  `former` and `past` can hold the same sign.
* **A known WLASL problem.** Neidle and colleagues at Boston University show that WLASL
  sometimes files one sign under several glosses and several signs under one gloss
  ([Neidle & Ballard 2022][asllrp21]; [Neidle et al. 2022][lrec22]), and have published
  [revised gloss labels][asllrp-glosses] for about 19,700 WLASL videos. The WLASL paper itself
  admits gloss ambiguity, and the NLA-SLR paper calls such pairs "visually indistinguishable
  signs".

What this looks like in our data (asl2000_cutoff_9 labels and MViTv2_S exp000's stashed test
predictions, `results/satnac_2026/all_misspredictions.ipynb`; checked 2026-10-05):

| Gloss | Train / val / test instances | Variations |
|---|---|---|
| `before` | 18 / 4 / 4 | 0 and 1 (10 / 8 in train) |
| `former` | 5 / 1 / 1 | 0 only |
| `past` | 10 / 3 / 2 | 0 only |

All 4 `before` test instances are mispredicted, split by variation: both variation-0 instances
are predicted as `former` (0.37) and `past` (0.74), the over-the-shoulder glosses above; the
variation-1 ones as `beside` and `next`. The `former` test instance is predicted correctly. Such
errors say more about the labels than the model, so treat confusions between these glosses
with care when reading per-class results.

Still open: whether this is an annotation clash or signers using the forms interchangeably, and
whether the Boston University labels merge or split these classes (tracked in `src/TODO.md`).

Sources (links added 2026-10-05):

* Lifeprint: [before][lp-before], [past][lp-past]
* MSU ASL Browser: [former][msu-former]
* Neidle & Ballard 2022, [ASLLRP Report 21][asllrp21]
* [Revised WLASL gloss labels (ASLLRP)][asllrp-glosses]
* Neidle et al. 2022, [LREC sign language workshop][lrec22]

[lp-before]: https://www.lifeprint.com/asl101/pages-signs/b/before.htm
[lp-past]: https://www.lifeprint.com/asl101/pages-signs/p/past.htm
[msu-former]: https://commtechlab.msu.edu/sites/aslweb/F/W1337.htm
[asllrp21]: https://www.bu.edu/asllrp/rpt21/asllrp21.pdf
[asllrp-glosses]: https://www.bu.edu/asllrp/wlasl-alt-glosses.pdf
[lrec22]: https://aclanthology.org/2022.signlang-1.26.pdf

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

Before the rerun (2026-10-05), the old `asl100`/.../`asl2000` labels were renamed to
`asl100_1_indexed`/.../`asl2000_1_indexed` (and `src/runs/asl100` to `src/runs/asl100_1_indexed`,
with its Que runs updated), so runs trained on the 1-based labels can still be tested on them. The
rerun only rewrites the plain splits: the `*_cutoff_9` labels stay 1-based for now, so the paused
asl100_cutoff_9 sweep can finish on the labels it started with.
