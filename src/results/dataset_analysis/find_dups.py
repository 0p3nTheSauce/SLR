# For the objective of comapring near duplicates with different
# lenghts, and assuming thumbs now contains a run of consistently
# spaced frames, we could compare a runs of frames to a run of
# frames from videos with different lengths.

# We might also expect a case where a video is stored under more
# than a pair (three similar videos).

# We might also expect that near duplicates have the same signer ID

import itertools
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from scipy.fft import dctn

from src.run_types import RAW_DIR
from src.utils import cv_load
from src.results import get_stash_path, load_json, stash_json


def _build_fpath(vid: str) -> Path:
    return (RAW_DIR / vid).with_suffix(".mp4")


def _area_matrix(n_in: int, n_out: int) -> np.ndarray:
    """(n_out, n_in) matrix; row i holds the fractional coverage of each
    input pixel by output bin i, normalised to sum to 1."""
    edges = np.linspace(0, n_in, n_out + 1)
    lo, hi = edges[:-1, None], edges[1:, None]
    px = np.arange(n_in)[None, :]
    overlap = np.clip(np.minimum(hi, px + 1) - np.maximum(lo, px), 0, None)
    return (overlap / (n_in / n_out)).astype(np.float32)


def _resize_area(grey: np.ndarray, h_out: int, w_out: int | None = None) -> np.ndarray:
    """Batched area resize of TxHxW -> Tx h_out x w_out (float32)."""
    w_out = w_out or h_out
    _, h, w = grey.shape
    r_h = _area_matrix(h, h_out)  # (h_out, H)
    r_w = _area_matrix(w, w_out)  # (w_out, W)
    return r_h @ grey @ r_w.T  # broadcasts over T


def _prep_vid(frames: np.ndarray, h_out: int = 64, canon: int = 256) -> np.ndarray:
    """Greyscale (average), area-resize to canon x canon, then block-average to h_out."""
    grey = frames.mean(axis=-1, dtype=np.float32)
    grey = _resize_area(grey, canon)  # guarantees a consistent shape
    t = grey.shape[0]
    q = canon // h_out  # requires canon % h_out == 0
    small = grey.reshape(t, h_out, q, h_out, q).mean(axis=(2, 4))
    return np.rint(small).astype(np.uint8)


def _pHash(frames: np.ndarray, hash_h_out: int = 8) -> np.ndarray:
    """Batched pHash applied framewise

    Args:
        frames (np.ndarray): TxHxW
        hash_h_out (int, optional): To exclude high frequences, assumes Tx64x64. Defaults to 8.

    Returns:
        np.ndarray: T pHash integers.
    """
    dcts = cast(np.ndarray, dctn(frames.astype(np.float32), axes=(1, 2)))

    # keep only low frequencies
    dctlowfreq = dcts[:, :hash_h_out, :hash_h_out]

    # average brightness frequencies
    med = np.median(dctlowfreq, axis=(1, 2), keepdims=True)

    # flip bits to 0 or 1 if higher or lower than average
    diff = dctlowfreq > med

    flat_diff = diff.reshape(diff.shape[0], -1)

    packed_bytes = np.packbits(flat_diff, axis=-1)

    hash_ints = packed_bytes.view(np.uint64).squeeze(axis=-1)

    return hash_ints


def _pHash_global(frames: np.ndarray, hash_h_out: int = 8) -> np.ndarray:
    """Collapses the video over the time axis first, then computes a single pHash.

    Args:
        frames (np.ndarray): TxHxW
        hash_h_out (int, optional): To exclude high frequences, assumes Tx64x64. Defaults to 8.

    Returns:
        np.ndarray: Unified pHash of video.
    """
    if len(frames) == 0:
        raise ValueError("Cannot perform hash for len(frames)==0")
    # 1. Collapse the time dimension by taking the mean across frames
    # Shape changes from (T, H, W) -> (H, W)
    temporal_mean = frames.mean(axis=0)

    # 2. Add a fake batch dimension so it plays nice with your existing code structure
    # Shape changes from (H, W) -> (1, H, W)
    video_as_one_frame = temporal_mean[np.newaxis, ...]

    # 3. Compute the standard pHash on the combined frame
    dcts = cast(np.ndarray, dctn(video_as_one_frame.astype(np.float32), axes=(1, 2)))
    dctlowfreq = dcts[:, :hash_h_out, :hash_h_out]

    med = np.median(dctlowfreq, keepdims=True)
    diff = dctlowfreq > med

    flat_diff = diff.reshape(diff.shape[0], -1)
    packed_bytes = np.packbits(flat_diff, axis=-1)

    # Returns a 1-element array containing the single global uint64 hash
    return packed_bytes.view(np.uint64).squeeze(axis=-1)


def _hamming(i1: int, i2: int) -> int:
    """Compute the hamming distance between two integers"""
    return int(int(i1) ^ int(i2)).bit_count()


def _round_trip_compress(frames: np.ndarray) -> np.ndarray:
    # return np.asarray([cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 50]) for f in frames])
    out = np.empty_like(frames)
    for i, f in enumerate(frames):
        ok, buff = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 50])
        if not ok:
            raise RuntimeError(f"JPEG encode failed on frame: {i}")
        out[i] = cv2.imdecode(buff, cv2.IMREAD_COLOR)
    return out


def _compare_pf_hashes(
    original_pfH: np.ndarray,
    variant_pfH: np.ndarray,
    min_overlap: int | None = None,
) -> tuple[float, int]:
    """Mean per-frame Hamming distance minimised over temporal offsets.

    Offset k means variant frame j lines up with original frame j + k, so
    k > 0: the variant starts k frames into the original (start trimmed),
    k < 0: the variant has -k extra frames before the original starts.

    Returns:
        (best_mean_distance, best_k)
    """
    n1, n2 = len(original_pfH), len(variant_pfH)
    if min_overlap is None:
        min_overlap = max(1, min(n1, n2) // 2)

    best_dist, best_k = float("inf"), 0

    # every k that leaves at least `min_overlap` overlapping frames
    for k in range(min_overlap - n2, n1 - min_overlap + 1):
        o0, v0 = max(k, 0), max(-k, 0)
        overlap = min(n1 - o0, n2 - v0)

        o = original_pfH[o0 : o0 + overlap]
        v = variant_pfH[v0 : v0 + overlap]

        mean_dist = float(np.bitwise_count(o ^ v).mean())
        if mean_dist < best_dist:
            best_dist, best_k = mean_dist, k

    return best_dist, best_k


def report(all_trims: dict[str, Callable], original: np.ndarray, augmented: np.ndarray):
    """Compare a series of provided crops/trims on the global and per frame pHashes.
    Video preparation is applied internally (do not call _prep_vid before calling this function).

    Args:
        all_trims (dict[str, Callable]): Dictionary containing lambda augmentations
        original (np.ndarray): Unaltered original (applicable if additional augmentations such as recoding applied)
        augmented (np.ndarray): Pre-augmented (e.g. jpeg round trip)
    """
    o_prep = _prep_vid(original)
    o_gH = _pHash_global(o_prep)
    o_pfH = _pHash(o_prep)
    for k, f in all_trims.items():
        print()
        print(f"{' '.join(k.split('_'))}:")
        auged = _prep_vid(f(augmented))
        gH = _pHash_global(auged)
        pfH = _pHash(auged)
        g_d = _hamming(o_gH[0], gH[0])
        pf_d, pf_k = _compare_pf_hashes(o_pfH, pfH)
        # print(f"Global pHash: {g}")
        print(f"Hamming global: {g_d}")
        print(f"Best mean per-frame hamming: {pf_d} offset: {pf_k}")


def pHash(frames: np.ndarray, hash_h_out: int = 8) -> np.ndarray:
    """Batched pHash applied framewise, including greyscale and resize.

    Args:
        frames (np.ndarray): TxHxW
        hash_h_out (int, optional): To exclude high frequences, assumes Tx64x64. Defaults to 8.

    Returns:
        np.ndarray: T pHash integers.
    """
    return _pHash(_prep_vid(frames), hash_h_out)


def build_frame_dictionary(video_ids: list[str]) -> dict[int, list[tuple[str, int]]]:
    """Build frame dictionary from per frame perceptual hashes

    Args:
        video_ids (list[str]): E.g. ["05741"]

    Returns:
        dict[int, list[tuple[str, int]]]: Frame-pHash indexed buckets containing (video_id, frame_index) elements.
    """
    frame_dict: dict[int, list[tuple[str, int]]] = defaultdict(list)

    for vid in video_ids:
        fname = _build_fpath(vid)
        pf_pHashes = pHash(cv_load(fname, 0, 0, all=True))
        for i, p in enumerate(pf_pHashes):
            frame_dict[p].append((vid, i))

    return frame_dict


RunIndex = dict[int, list[tuple[str, list[int]]]]  # hash -> [(video_id, [indexes])]


def hash_videos(video_ids: list[str]) -> dict[str, np.ndarray]:
    """Decode each video and return its per-frame pHashes"""
    return {vid: pHash(cv_load(_build_fpath(vid), 0, 0, all=True)) for vid in video_ids}


def aggregate_hashes(vids_to_pf_hashes: dict[str, list[int]]) -> RunIndex:
    """Index [(video_id, [frame_indexes])] under frame hash keys

    Args:
        vids_to_pf_hashes (dict[str, np.ndarray]): video_id (key) to frame_hash (value)

    Returns:
        RunIndex: frame_hash (key) to video entries (video_id, [frame_indexes])
    """
    index = defaultdict(list[tuple[str, list[int]]])
    for vid, pfHashes in vids_to_pf_hashes.items():
        uhashes = defaultdict(list[int])  # hash -> [indexes]
        for i, pfH in enumerate(pfHashes):
            uhashes[int(pfH)].append(i)

        for uhash, idexes in uhashes.items():
            index[uhash].append((vid, idexes))
    return index


def stoplist(index: RunIndex, max_videos: int) -> tuple[RunIndex, RunIndex]:  # step 2
    """Split into (kept, dropped) by distinct videos per bucket."""
    kept: RunIndex = defaultdict(list)
    dropped: RunIndex = defaultdict(list)
    for ph, vid_entries in index.items():
        if len(vid_entries) > max_videos or len(vid_entries) == 1:
            dropped[ph].extend(vid_entries)
        else:
            kept[ph].extend(vid_entries)
    return kept, dropped


# def count_shared(kept: RunIndex):
#     for ph, vid_entries in kept.items():
#         for (va, indexesa), (vb, indexesb) in itertools.combinations(vid_entries, 2):


def _summarise(
    fdict: dict[int, list[tuple[str, int]]], known_pairs: list[tuple[str, str]]
):

    black = _pHash(np.zeros([1, 64, 64]))[0]
    white = _pHash(np.ones([1, 64, 64]) * 255)[0]

    bucket_lengths = {}  # size buckets
    video_to_hash = defaultdict(list)  # inverse dict
    indexes_to_hash = defaultdict(list[int])  # inverse dict

    for pH, elem in fdict.items():
        bucket_lengths[pH] = len(elem)
        for vid, idx in elem:
            video_to_hash[vid].append(pH)
            indexes_to_hash[idx].append(pH)

    # lengths_to_hash = {v : k for k, v in bucket_lengths.items()}
    largest_bucket = max(
        [(pH, len_elem) for pH, len_elem in bucket_lengths.items()], key=lambda x: x[1]
    )
    smallest_bucket = min(
        [(pH, len_elem) for pH, len_elem in bucket_lengths.items()], key=lambda x: x[1]
    )

    print(f"Largest bucket ({largest_bucket[0]}): {largest_bucket[1]}")
    for vid, idx in fdict[largest_bucket[0]]:
        print(f"Video: {vid} Idx: {idx}")
    print("-" * 10)
    print(f"Smallest bucket ({smallest_bucket[0]}): {smallest_bucket[1]}")
    for vid, idx in fdict[smallest_bucket[0]]:
        print(f"Video: {vid} Idx: {idx}")
    print("-" * 10)
    print()
    for v1, v2 in known_pairs:
        for v in [v1, v2]:
            vhashes = set(video_to_hash[v])
            print(f"Number of pHash entries for {v}: {len(vhashes)}")
            for ph in vhashes:
                frames_per_vid = Counter(vid for vid, _ in fdict[ph])
                top_vid, n_frames = frames_per_vid.most_common(1)[0]
                print(
                    f"Bucket {ph}: {len(frames_per_vid)} distinct videos; "
                    f"most frames from {top_vid} ({n_frames})"
                )
        print("-" * 10)
    print()
    print(f"Number of black ({black}) frames: {len(fdict[black])}")
    print(f"Number of white ({white}) frames: {len(fdict[white])}")


def initial_tests():
    vid1 = "05741"  # (before, test) signer 11
    vid2 = "41452"  # (past, train) signer 11
    vid3 = "00381"  # (able, train) signer 11: same signer, different action
    vid4 = "41448"  # (past, test) signer 4: different signer, same action (relative to past)

    # wait = 0
    # show_frame = True

    hams = []

    for v in [vid1, vid2, vid3, vid4]:
        fname = _build_fpath(v)
        # print(f"{v} exists: {fname.exists()}")
        frames = cv_load(fname, 0, 0, all=True)
        # print(f"frames shape: {frames.shape}")
        preproc = _prep_vid(frames)
        # print(f"frames shape after greyscale and resize: {preproc.shape}")
        # if show_frame:
        #     cv2.imshow(f"First frame: {fname}", preproc[0])
        #     cv2.waitKey(wait)
        #     cv2.destroyAllWindows()

        # print(f"First frame pHash: {_pHash(preproc)[0]}")
        # print(f"Video pHash: {_pHash_global(preproc)[0]}")

        hams.append(_pHash_global(preproc)[0])

    print(f"(A) Distance identical videos: {_hamming(hams[0], hams[1])}")
    print(f"(B) Distance same signer, different action: {_hamming(hams[0], hams[2])}")
    print(f"(C) Distance different signer, same action: {_hamming(hams[0], hams[3])}")
    print(f"Redundency check, should be same as (C): {_hamming(hams[1], hams[3])}")


def _crop_frac(
    f: np.ndarray,
    start: float = 0.0,
    end: float = 0.0,
    top: float = 0.0,
    bottom: float = 0.0,
    left: float = 0.0,
    right: float = 0.0,
) -> np.ndarray:
    """Trim a TxHxWxC clip by a fraction of each axis' length, per side."""
    t, h, w, _ = f.shape
    t0, t1 = round(start * t), t - round(end * t)
    y0, y1 = round(top * h), h - round(bottom * h)
    x0, x1 = round(left * w), w - round(right * w)
    return f[t0:t1, y0:y1, x0:x1]


def _make_aug(sides: tuple[str, ...], p: float):
    # factory function, so sides and p are bound per augmentation
    return lambda f: _crop_frac(f, **{s: p for s in sides})


def _build_augs(ops: dict[str, tuple[str, ...]], fracs) -> dict:
    return {
        f"{name}_{p:.2f}": _make_aug(sides, p)
        for name, sides in ops.items()
        for p in fracs
    }


# name -> which sides get trimmed by the same fraction p
TEMPORAL_OPS = {
    "trim_start": ("start",),
    "trim_end": ("end",),
    "trim_start_end": ("start", "end"),
}
SPATIAL_OPS = {
    "crop_left": ("left",),
    "crop_right": ("right",),
    "crop_left_right": ("left", "right"),
    "crop_top": ("top",),
    "crop_bottom": ("bottom",),
    "crop_top_bottom": ("top", "bottom"),
}
COMBINED_OPS = {
    "trim_crop_all": ("start", "end", "top", "bottom", "left", "right"),
}

FRACS = (0.05, 0.10, 0.20, 0.25)
IDENTITY = {"identity": lambda f: f}


def compare_hashes():
    v = "05741"  # (before, test) signer 11
    fname = _build_fpath(v)
    frames = cv_load(fname, 0, 0, all=True)

    all_trims = IDENTITY | _build_augs(TEMPORAL_OPS, FRACS)
    all_crops = IDENTITY | _build_augs(SPATIAL_OPS, FRACS)
    all_trims_and_crops = all_trims | all_crops | _build_augs(COMBINED_OPS, FRACS)

    o_preped = _prep_vid(frames)
    o_gH = _pHash_global(o_preped)
    o_pfH = _pHash(o_preped)
    # print(f"Origninal global: {ogH}")
    # print(f"Original frame-wise: {ofH}")

    print("-" * 10)
    report(all_trims_and_crops, frames, frames)
    print("-" * 10)
    print()

    jpeg_recoded = _round_trip_compress(frames)
    recoded_preped = _prep_vid(jpeg_recoded)
    rgH = _pHash_global(recoded_preped)
    # print(f"Recoded pHash: {rgH}")
    print(f"Hamming global recoded: {_hamming(rgH[0], o_gH[0])}")
    pfH = _pHash(recoded_preped)
    pf_d, pf_k = _compare_pf_hashes(o_pfH, pfH)
    print(f"Best mean per-frame hamming recoded: {pf_d} offset: {pf_k}")
    print("-" * 10)

    report(all_trims_and_crops, frames, jpeg_recoded)
    print("-" * 10)
    print()


def small_set_test():
    vid1 = "05741"  # (before, test) signer 11
    vid2 = "41452"  # (past, train) signer 11
    vid3 = "00381"  # (able, train) signer 11: same signer, different action
    vid4 = "41448"  # (past, test) signer 4: different signer, same action (relative to past)
    vid5 = "05743"
    vid6 = "41454"
    num_vids = 26
    vids = {vid1, vid2, vid3, vid4, vid5, vid6}

    for p in RAW_DIR.iterdir():
        if len(vids) == num_vids:
            break
        vids.add(p.stem)

    fdict = build_frame_dictionary(list(vids))

    _summarise(fdict, [(vid1, vid2), (vid5, vid6)])

    # for pH, elem in fdict.items():
    #     print(f"pHash: {pH}")
    #     for vid, idx in elem:
    #         print(f"Video: {vid} Idx: {idx}")
    #     print()


def build_example_vids() -> set[str]:
    vid1 = "05741"  # (before, test) signer 11
    vid2 = "41452"  # (past, train) signer 11
    vid3 = "00381"  # (able, train) signer 11: same signer, different action
    vid4 = "41448"  # (past, test) signer 4: different signer, same action (relative to past)
    vid5 = "05743"
    vid6 = "41454"
    num_vids = 26
    vids = {vid1, vid2, vid3, vid4, vid5, vid6}

    for p in RAW_DIR.iterdir():
        if len(vids) == num_vids:
            break
        vids.add(p.stem)

    return vids


def stash_hashes(stub: str, vids: list[str]) -> Path:
    return stash_json(
        {v: list(map(int, idxes)) for v, idxes in hash_videos(vids).items()},
        get_stash_path(stub),
    )


def test_pairs():
    max_vids = 50
    # stash_p = stash_hashes("test_vids", list(build_example_vids()))
    # print(stash_p) stashed_results/test_vidsresults.json
    vid_hashes: dict[str, list[int]] = load_json(
        Path("./stashed_results/test_vidsresults.json")
    )
    index = aggregate_hashes(vid_hashes)
    print("Index[h0]:")
    for pH, vid_entries in index.items():
        print(f"{pH} : {vid_entries}")
        break
    kept, dropped = stoplist(index, max_vids)
    print("Kept[h0]:")
    for pH, vid_entries in kept.items():
        print(f"{pH} : {vid_entries}")
        break
    print("Dropped[h0]:")
    for pH, vid_entries in dropped.items():
        print(f"{pH} : {vid_entries}")
        break
    


if __name__ == "__main__":
    # initial_tests()
    # compare_hashes()
    # print("\n" * 2, "-" * 10, "\n" * 2)
    # small_set_test()
    test_pairs()
