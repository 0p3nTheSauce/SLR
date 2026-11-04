# Near-duplicate search with per-frame perceptual hashes

Findings from the near-duplicate search in [`duplicates.py`](duplicates.py) (`find_near_duplicates`;
then `find_dups.py`, merged into it on 2026-11-04) on WLASL-2000, run on 2026-11-02. The question: are
there copies of the same clip that are *not* byte-identical (re-encoded, trimmed or cropped), beyond
the exact copies described in [`WLASL_info.md`](../../info/WLASL_info.md#duplicate-clips)?

**Short answer: none found yet.** The pipeline recovers every exact copy, but every new pair that was
checked turned out to be two different signs by the same signer in the same recording session. The
per-frame pHash can't tell those apart, so the verification step needs to work at pixel level
before this search can say anything about trimmed copies.

## Method

1. **Hash.** Each frame is greyscaled, area-resized to 256 then 64 px, and turned into a 64-bit
   pHash: the 8×8 low-frequency DCT corner, thresholded at its median. All 21,095 videos took 36 min
   on the local machine (stashed as `stashed_results/asl2000_frame_phash_results.json`, which is
   gitignored; rebuild it with `stash_phashes()`).
2. **Candidates (blocking).** Each video becomes its set of distinct frame hashes. Index hash →
   videos, drop hashes shared by more than 50 videos (stop-list), count the hashes each pair shares,
   and keep pairs where `shared / min(set sizes) >= 0.5`.
3. **Verify.** Slide one clip's hash sequence along the other's at every offset that overlaps at
   least half the shorter clip, and keep the pair if the best mean per-frame Hamming distance is
   ≤ 2 bits.
4. **Group.** Union-find over the verified pairs.

Robustness on the 20 test clips (earlier in the exercise): per-frame hashes survive JPEG
re-encoding and trims (the best offset recovers the trim length). Whole-clip ("global") and 3D-DCT
hashes did worse and were dropped.

## Results

| Stage | Count |
|---|---|
| Videos | 21,095 |
| Hash buckets kept / dropped by the stop-list | 43,053 / 5 (largest: the black frame, in 1,960 videos) |
| Pairs sharing at least one hash | 118,062 |
| Candidates (≥ 50% shared) | 2,773 |
| Verified (≤ 2 bits) | 2,260 pairs in 945 groups, including groups of 141, 54, 22 and 11 |
| Recall on the old `exact` pairs | **1,301 / 1,301** |
| Recall on the old `near` pairs | 45 / 670 |
| Verified pairs that aren't exact copies | 959 (940 also leaving out the old `near` pairs) |

The candidate and verification steps take about 6 s once the hashes are stashed.

### The old `near` groups are false positives

`near` was the second kind in `find_duplicates.py`'s report (`stashed_results/duplicates_asl2000.json`;
both removed on 2026-11-04, last in commit `9291980`), from 16×16 thumbnails of same-length clips. Of its 670
pairs:

- 100% have the same signer, and only 2% the same gloss (e.g. *milk*/*think*, *peaceful*/*rubber*);
- 82% share no frame hash at all;
- their aligned raw-pixel mean |difference| is 4–7, well above a re-encode (see below).

So they are different signs from one session, which the thumbnails averaged together. This does
not affect the thesis: [`duplicate_accuracy.ipynb`](../error_analysis/duplicate_accuracy.ipynb) and
`signer_shortcut.py` use exact (MD5) copies only.

### The new pairs are false positives too

The 141- and 54-video groups are chains: signer 11 on `signingsavvy`, across about 120 glosses,
linked one pair at a time. Requiring the overlap to cover 90% of the shorter clip (rather than 50%)
gives 1,944 verified pairs, a largest group of 53 and 625 new pairs, with exact recall still
1,301/1,301. Of those 625:

- 609 are cross-gloss and 623 have the same signer (e.g. *pickle*/*sour*, *k*/*w*, *gun*/*talk*);
- 39 have a per-frame distance of **0**, i.e. identical hash sequences;
- the aligned raw-pixel mean |difference| is 5–14, including the distance-0 pairs.

For calibration, the same measure is about 0 for exact copies and about 2 after a JPEG quality-50
round trip.

## Why: the hash identifies the session, not the sign

An 8×8 low-frequency DCT of a 64 px frame is dominated by the background, lighting, clothing and
body position. The hands are a small part of the frame and barely move those coefficients (e.g.
the 28 frames of video 05741 collapse to 2 distinct hashes). Two signs recorded in the same session
can therefore have identical hash sequences. Exact copies are recovered because they are trivially
identical, but nothing in the hash separates a trimmed copy from a different sign.

## Next steps

- **Pixel-level verifier.** Blocking is still useful (2,773 candidates out of about 222 million
  pairs), so decode only the candidates and compare aligned frames. Whole-frame mean |diff| is not
  enough: the *m*/*n* fingerspelling pair differs by only 1.4, because the difference is confined
  to the hand. Measure where frames differ instead, e.g. the fraction of pixels with
  |diff| > 25, maximised over frames: re-encoding noise is small and spread out, while a different
  sign gives large, concentrated changes around the hands.
- **Candidate recall is unmeasured.** There are no confirmed trimmed copies to test the
  hash-sharing step against. Synthetic trims and crops of real clips (as in `find_dups.compare_hashes`, commit `9291980`) are the
  fallback.
- **Spatial crops.** Not tested yet; a crop changes the DCT basis, so it may need a hash on a
  centre region or a finer grid.
