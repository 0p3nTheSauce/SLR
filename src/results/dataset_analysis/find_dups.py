# For the objective of comapring near duplicates with different
# lenghts, and assuming thumbs now contains a run of consistently
# spaced frames, we could compare a runs of frames to a run of
# frames from videos with different lengths.

# We might also expect a case where a video is stored under more
# than a pair (three similar videos).

# We might also expect that near duplicates have the same signer ID

from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from scipy.fft import dctn

from src.run_types import RAW_DIR
from src.utils import cv_load


def _build_fpath(vid: str) -> Path:
    return (RAW_DIR / vid).with_suffix(".mp4")


def _prep_vid(frames: np.ndarray, h_out: int = 64) -> np.ndarray:
    """Greyscale (average) then resize (inter area)

    Args:
        frames (np.ndarray): Batched frames (TxHxWxC)
        h_out (int, optional): Length of sides of output frames (H and W). Defaults to 64.

    Returns:
        np.ndarray: Greyscaled resized frames.
    """
    grey = frames.mean(axis=-1)
    t, h, w = grey.shape

    q_h = h // h_out
    q_w = w // h_out

    # Every HxW frame is split into h_outxh_out blocks of qxq pixels
    resized = grey.reshape(t, h_out, q_h, h_out, q_w).mean(axis=(2, 4))
    return resized.astype(np.uint8)


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
    og_prepped = _prep_vid(original)
    o_gH = _pHash_global(og_prepped)
    o_pfH = _pHash(og_prepped)
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


def _summarise(fdict: dict[int, list[tuple[str, int]]], known_pairs: list[tuple[str, str]]):
    
    black = _pHash(np.zeros([1, 64,64]))[0]
    white = _pHash(np.ones([1, 64,64])* 255)[0]

    bucket_lengths = {} #size buckets
    video_to_hash = defaultdict(list) # inverse dict
    indexes_to_hash = defaultdict(list[int]) # inverse dict
    
    for pH, elem in fdict.items():
        bucket_lengths[pH] = len(elem)
        for vid, idx in elem:
            video_to_hash[vid].append(pH)
            indexes_to_hash[idx].append(pH)

    # lengths_to_hash = {v : k for k, v in bucket_lengths.items()}
    largest_bucket = max([(pH, len_elem) for pH, len_elem in bucket_lengths.items()], key=lambda x: x[1])
    smallest_bucket = min([(pH, len_elem) for pH, len_elem in bucket_lengths.items()], key=lambda x: x[1])
    
    
    print(f"Largest bucket ({largest_bucket[0]}): {largest_bucket[1]}")   
    for vid, idx in fdict[largest_bucket[0]]:
        print(f"Video: {vid} Idx: {idx}")
    print('-'*10)
    print(f"Smallest bucket ({smallest_bucket[0]}): {smallest_bucket[1]}")
    for vid, idx in fdict[smallest_bucket[0]]:
        print(f"Video: {vid} Idx: {idx}")
    print('-'*10)
    print()
    for v1, v2 in known_pairs:
        for v in [v1, v2]:
            vhashes = set(video_to_hash[v])
            print(f"Number of pHash entries for {v}: {len(vhashes)}")
            for ph in vhashes:
                idx_counts = Counter(fdict[ph])
                most_common = max([(idx, c) for idx, c in idx_counts.items()], key=lambda x:x[1])
                print(f"Most common index for {most_common[0]}: {most_common[1]}")
        print('-'*10)
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

def compare_hashes():
    v = "05741"  # (before, test) signer 11
    fname = _build_fpath(v)
    frames = cv_load(fname, 0, 0, all=True)
    # preproc = _prep_vid(frames)

    T = frames.shape[0]
    frame_deviations = [d for d in (5, 10, 20, round(0.25 * T)) if 2 * d < T]
    starts = {f"trim_start_{d}": lambda f, d=d: f[d:] for d in frame_deviations}
    ends = {f"trim_end_{d}": lambda f, d=d: f[:-d] for d in frame_deviations}
    both = {f"trim_start_end_{d}": lambda f, d=d: f[d:-d] for d in frame_deviations}
    all_trims = starts | ends | both | {"identity": lambda f: f}


    lefts = {f"crop left_{d}": lambda f, d=d: f[:, :, :-d] for d in frame_deviations}
    rights = {f"crop right_{d}": lambda f, d=d: f[:, :, d:] for d in frame_deviations}
    left_and_rights = {f"crop left_and_right_{d}": lambda f, d=d: f[:, :, d:-d] for d in frame_deviations}
    tops = {f"crop top_{d}": lambda f, d=d: f[:, :-d, :] for d in frame_deviations}
    bottoms = {f"crop bottom_{d}": lambda f, d=d: f[:, d:, :] for d in frame_deviations}
    top_and_bottoms = {f"crop top_and_bottom_{d}": lambda f, d=d: f[:, d:-d, :] for d in frame_deviations}
    all_crops = lefts | rights | left_and_rights | tops | bottoms | top_and_bottoms | {"identity": lambda f: f}

    top_bottom_start_end = {f"crop top_and_bottom_start_end_{d}": lambda f, d=d: f[d:-d, d:-d, d:-d] for d in frame_deviations}
    all_trims_and_crops = all_trims | all_crops | top_bottom_start_end

    og_prepped = _prep_vid(frames)
    o_gH = _pHash_global(og_prepped)
    o_pfH = _pHash(og_prepped)
    # print(f"Origninal global: {ogH}")
    # print(f"Original frame-wise: {ofH}")

    print("-" * 10)
    report(all_trims_and_crops, frames, frames)
    print("-" * 10)
    print()
    
    recoded = _round_trip_compress(frames)
    recoded_preped = _prep_vid(recoded)
    rgH = _pHash_global(recoded_preped)
    # print(f"Recoded pHash: {rgH}")
    print(f"Hamming global recoded: {_hamming(rgH[0], o_gH[0])}")
    pfH = _pHash(recoded_preped)
    pf_d, pf_k = _compare_pf_hashes(o_pfH, pfH)
    print(f"Best mean per-frame hamming recoded: {pf_d} offset: {pf_k}")
    print("-" * 10)

    report(all_trims_and_crops, frames, recoded)
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

if __name__ == "__main__":
    # initial_tests()
    compare_hashes()
    print('\n'*2, '-'*10, '\n'*2)
    small_set_test()
    
