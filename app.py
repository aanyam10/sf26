from __future__ import annotations

import gc
import io
import json
import os
import random
import shutil
import time
import zipfile
from datetime import datetime
from typing import List, Tuple

import matplotlib
import numpy as np
import pydicom
import streamlit as st
from scipy import ndimage


# Ensure GIF rendering works in headless environments.
matplotlib.use("Agg")
from matplotlib import animation, pyplot as plt


# Default to Colab-equivalent folder layout in repo.
DEFAULT_HEALTHY_ROOT = "data"
DEFAULT_BEFORE_PATHS = [
    os.path.join(DEFAULT_HEALTHY_ROOT, "trial1"),
    os.path.join(DEFAULT_HEALTHY_ROOT, "trial2"),
    os.path.join(DEFAULT_HEALTHY_ROOT, "trial3"),
]
DEFAULT_AFTER_PATHS = [
    os.path.join(DEFAULT_HEALTHY_ROOT, "trial1"),
    os.path.join(DEFAULT_HEALTHY_ROOT, "trial2"),
    os.path.join(DEFAULT_HEALTHY_ROOT, "trial3"),
]


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        tf = get_tf()
        tf.random.set_seed(seed)
    except Exception:
        # Allow app to load even if TF is unavailable until runtime.
        pass


def get_tf():
    try:
        import tensorflow as tf  # type: ignore
    except Exception as e:
        raise RuntimeError("TensorFlow is unavailable in this environment.") from e
    return tf


def get_sorted_dicom_files(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")
    fs = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    if not fs:
        raise FileNotFoundError(f"No files in {folder}")

    def k(path: str):
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
            if hasattr(ds, "InstanceNumber"):
                return (0, int(ds.InstanceNumber))
            if hasattr(ds, "ImagePositionPatient"):
                return (1, float(ds.ImagePositionPatient[2]))
        except Exception:
            pass
        return (2, os.path.basename(path))

    return sorted(fs, key=k)


def read_hu(path: str) -> np.ndarray:
    ds = pydicom.dcmread(path, force=True)
    arr = ds.pixel_array.astype(np.float32)
    s = float(getattr(ds, "RescaleSlope", 1.0))
    b = float(getattr(ds, "RescaleIntercept", 0.0))
    return arr * s + b


def get_pixel_spacing_mm(path: str) -> Tuple[float, float]:
    ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
    px = getattr(ds, "PixelSpacing", [1.0, 1.0])
    return float(px[0]), float(px[1])


def resize_2d(img: np.ndarray, target_shape: Tuple[int, int], order: int = 1) -> np.ndarray:
    if img.shape == target_shape:
        return img
    z = (target_shape[0] / img.shape[0], target_shape[1] / img.shape[1])
    return ndimage.zoom(img, zoom=z, order=order)


def ct_window_norm(hu: np.ndarray, lo: float = -1000.0, hi: float = 400.0) -> np.ndarray:
    return np.clip((hu - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def keep_largest_components_2d(mask: np.ndarray, keep: int = 2) -> np.ndarray:
    lab, n = ndimage.label(mask)
    if n == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = ndimage.sum(mask, lab, index=np.arange(1, n + 1))
    ids = np.argsort(sizes)[::-1][:keep] + 1
    return np.isin(lab, ids)


def remove_small_blobs(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    lab, n = ndimage.label(mask)
    if n == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = ndimage.sum(mask, lab, index=np.arange(1, n + 1))
    keep_ids = np.where(sizes >= min_pixels)[0] + 1
    return np.isin(lab, keep_ids)


def lung_mask_2d(ct_hu: np.ndarray) -> np.ndarray:
    m = (ct_hu > -1000) & (ct_hu < -250)
    m = ndimage.binary_opening(m, structure=np.ones((3, 3)))
    m = ndimage.binary_closing(m, structure=np.ones((5, 5)))
    m = ndimage.binary_fill_holes(m)
    m = keep_largest_components_2d(m, keep=2)
    return m.astype(bool)


def disk_mask(shape: Tuple[int, int], cy: int, cx: int, r: int) -> np.ndarray:
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= (r**2)


def bresenham_line(y0: int, x0: int, y1: int, x1: int):
    pts = []
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        pts.append((y0, x0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
    return pts


def nearest_active(mask: np.ndarray, yx: Tuple[int, int]):
    c = np.argwhere(mask)
    if c.size == 0:
        return None
    d2 = (c[:, 0] - yx[0]) ** 2 + (c[:, 1] - yx[1]) ** 2
    j = int(np.argmin(d2))
    return int(c[j, 0]), int(c[j, 1])


def breathing_order(n: int):
    if n <= 1:
        return [0]
    return list(range(n)) + list(range(n - 2, -1, -1))


def run_treatment(
    active_global: np.ndarray,
    px_radius: int,
    max_steps: int = 160000,
    make_gif: bool = False,
    gif_path: str | None = None,
    ct_slices_u8: List[np.ndarray] | None = None,
    gray_masks: List[np.ndarray] | None = None,
    fps: int = 12,
    frame_stride: int = 6,
    max_gif_frames: int = 900,
    laser_off_when_idle: bool = False,
    static_overlay: bool = False,
):
    steps, treated = 0, 0
    initial = int(active_global.sum())

    writer = fig = ax = im = dot = None
    captured = 0
    global_capture_step = 0
    breath_seq = [0]

    if (
        make_gif
        and gif_path
        and ct_slices_u8 is not None
        and gray_masks is not None
        and len(ct_slices_u8) > 0
        and len(ct_slices_u8) == len(gray_masks)
    ):
        breath_seq = breathing_order(len(ct_slices_u8))
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.axis("off")

        first_bg = breath_seq[0]
        bg = ct_slices_u8[first_bg].astype(np.float32) / 255.0
        frame = np.repeat(bg[..., None], 3, axis=2)
        visible = active_global if static_overlay else (active_global & gray_masks[first_bg])
        frame[visible, 0] = 1.0
        frame[visible, 1] = 1.0
        frame[visible, 2] = 0.0

        im = ax.imshow(frame, interpolation="nearest")
        dot = plt.Circle((0, 0), radius=px_radius, edgecolor="red", fill=False, linewidth=2)
        dot.set_visible(False)
        ax.add_patch(dot)

        writer = animation.PillowWriter(fps=fps)
        writer.setup(fig, gif_path, dpi=90)
        writer.grab_frame()
        captured = 1

        def draw(background_idx, show_dot=False, laser_pos=(0, 0), laser_color="red"):
            bg0 = ct_slices_u8[background_idx].astype(np.float32) / 255.0
            rgb = np.repeat(bg0[..., None], 3, axis=2)
            visible_overlay = active_global if static_overlay else (active_global & gray_masks[background_idx])
            rgb[visible_overlay, 0] = 1.0
            rgb[visible_overlay, 1] = 1.0
            rgb[visible_overlay, 2] = 0.0
            im.set_data(rgb)

            if show_dot:
                py, px = laser_pos
                dot.center = (px, py)
                dot.set_edgecolor(laser_color)
                dot.set_visible(True)
            else:
                dot.set_visible(False)

    if initial > 0:
        c = np.argwhere(active_global)
        y, x = np.mean(c, axis=0).astype(int)
    else:
        y, x = active_global.shape[0] // 2, active_global.shape[1] // 2

    while active_global.any() and steps < max_steps:
        t = nearest_active(active_global, (y, x))
        if t is None:
            break
        ty, tx = t
        for py, px in bresenham_line(y, x, ty, tx):
            steps += 1
            hit = active_global & disk_mask(active_global.shape, py, px, px_radius)
            laser_on = np.any(hit)
            if np.any(hit):
                n = int(hit.sum())
                active_global[hit] = False
                treated += n

            if writer is not None and (steps % frame_stride == 0) and (captured < max_gif_frames):
                bg_idx = breath_seq[global_capture_step % len(breath_seq)]
                if laser_off_when_idle and not laser_on:
                    draw(bg_idx, show_dot=False)
                else:
                    draw(
                        bg_idx,
                        show_dot=True,
                        laser_pos=(py, px),
                        laser_color=("red" if laser_on else "blue"),
                    )
                writer.grab_frame()
                captured += 1
                global_capture_step += 1

            if not active_global.any() or steps >= max_steps:
                break
        y, x = ty, tx

    if writer is not None:
        if captured < max_gif_frames:
            bg_idx = breath_seq[global_capture_step % len(breath_seq)]
            draw(
                bg_idx,
                show_dot=False if laser_off_when_idle else True,
                laser_pos=(y, x),
                laser_color="red",
            )
            writer.grab_frame()
        writer.finish()
        plt.close(fig)

    remaining = int(active_global.sum())
    removed = (100.0 * treated / initial) if initial > 0 else 0.0
    return initial, treated, remaining, removed, steps


def save_csv(path: str, counts: np.ndarray, col: str):
    import csv

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["slice_index", col])
        for i, c in enumerate(counts.tolist()):
            w.writerow([i, int(c)])


def detect_lesion_before(c_hu, healthy_hus, zthr=2.5, gray_min=0.10, gray_max=0.85, min_px=40):
    c_n = ct_window_norm(c_hu)
    h_n = [ct_window_norm(h) for h in healthy_hus]
    base = np.median(np.stack(h_n, axis=0), axis=0)
    diff = c_n - base
    med = np.median(diff)
    mad = np.median(np.abs(diff - med)) + 1e-6
    z = 0.6745 * (diff - med) / mad
    lesion = (z > zthr) & lung_mask_2d(c_hu) & (c_n >= gray_min) & (c_n <= gray_max)
    lesion = ndimage.binary_opening(lesion, structure=np.ones((3, 3)))
    lesion = ndimage.binary_closing(lesion, structure=np.ones((5, 5)))
    return remove_small_blobs(lesion, min_px).astype(bool)


def run_before(cancer_path, healthy_paths, out_dir, make_gif=True, fps=12, frame_stride=6, max_gif_frames=900):
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    hf = [get_sorted_dicom_files(p) for p in healthy_paths]
    cf = get_sorted_dicom_files(cancer_path)
    n = min([len(cf)] + [len(x) for x in hf])
    hf, cf = [x[:n] for x in hf], cf[:n]
    y_mm, x_mm = get_pixel_spacing_mm(cf[0])
    px_radius = max(1, int(round(2.5 / ((y_mm + x_mm) / 2.0))))

    lesions, grays, ct_slices_u8 = [], [], []
    counts = np.zeros(n, dtype=np.int32)
    ref = None
    for i in range(n):
        c = read_hu(cf[i])
        ref = c.shape if ref is None else ref
        c = resize_2d(c, ref, 1)
        c_n = ct_window_norm(c)
        hs = [resize_2d(read_hu(x[i]), ref, 1) for x in hf]
        l = detect_lesion_before(c, hs)
        g = (c_n >= 0.10) & (c_n <= 0.85)
        l = l & g
        lesions.append(l)
        grays.append(g)
        counts[i] = int(l.sum())
        ct_slices_u8.append((c_n * 255.0).astype(np.uint8))
        if (i + 1) % 20 == 0:
            gc.collect()

    active, gray_union = np.zeros(ref, bool), np.zeros(ref, bool)
    for l, g in zip(lesions, grays):
        active |= l
        gray_union |= g
    active &= gray_union
    active = remove_small_blobs(active, 40)

    gif_path = os.path.join(out_dir, "before_treatment.gif")
    init, treated, rem, pct, steps = run_treatment(
        active,
        px_radius,
        160000,
        make_gif=make_gif,
        gif_path=gif_path,
        ct_slices_u8=ct_slices_u8,
        gray_masks=grays,
        fps=fps,
        frame_stride=frame_stride,
        max_gif_frames=max_gif_frames,
        laser_off_when_idle=False,
        static_overlay=True,
    )

    csv_path = os.path.join(out_dir, "slice_detected_counts.csv")
    save_csv(csv_path, counts, "detected_pixels_gray_only")
    summary = {
        "initial_total_cancer_pixels": init,
        "treated_total_pixels": treated,
        "remaining_total_pixels": rem,
        "percent_removed": round(pct, 2),
        "laser_steps": steps,
        "runtime_seconds": round(time.time() - t0, 2),
        "csv_path": csv_path,
        "gif_path": gif_path,
    }
    return summary


def remove_small_components_3d(mask3d, min_vox=40, min_span=1):
    lab, n = ndimage.label(mask3d)
    if n == 0:
        return np.zeros_like(mask3d, bool)
    keep = []
    for cid in range(1, n + 1):
        pts = np.argwhere(lab == cid)
        if pts.shape[0] < min_vox:
            continue
        zspan = int(pts[:, 0].max() - pts[:, 0].min() + 1)
        if zspan < min_span:
            continue
        keep.append(cid)
    return np.isin(lab, keep) if keep else np.zeros_like(mask3d, bool)


def temporal_majority_3d(mask3d, min_support=2):
    m = mask3d.astype(np.uint8)
    acc = m.copy()
    acc[1:] += m[:-1]
    acc[:-1] += m[1:]
    out = acc >= min_support
    return mask3d if (out.sum() == 0 and mask3d.sum() > 0) else out


def robust_positive_z(arr, valid):
    vals = arr[valid] if np.any(valid) else arr.ravel()
    med = np.median(vals)
    mad = np.median(np.abs(vals - med)) + 1e-6
    z = 0.6745 * (arr - med) / mad
    return np.clip(z, 0.0, None).astype(np.float32)


def build_ae(shape=(160, 160, 1), lr=1e-3):
    tf = get_tf()
    i = tf.keras.layers.Input(shape=shape)
    x = tf.keras.layers.Conv2D(16, 3, strides=2, padding="same", activation="relu")(i)
    x = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2DTranspose(16, 3, strides=2, padding="same", activation="relu")(x)
    o = tf.keras.layers.Conv2DTranspose(1, 3, strides=2, padding="same", activation="sigmoid")(x)
    m = tf.keras.Model(i, o)
    m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return m


def neurosymbolic_cleanup(mask3d, min_px=18, min_vox=40, min_span=1):
    out = np.zeros_like(mask3d, bool)
    for z in range(mask3d.shape[0]):
        m = ndimage.binary_opening(mask3d[z], structure=np.ones((3, 3)))
        m = ndimage.binary_closing(m, structure=np.ones((5, 5)))
        out[z] = remove_small_blobs(m, min_px)
    out = temporal_majority_3d(out, 2)
    out3 = remove_small_components_3d(out, min_vox, min_span)
    return out3 if out3.sum() > 0 else out


def topk_fallback_3d(score3d, valid3d, ratio=0.006, min_per_slice=14):
    out = np.zeros_like(valid3d, bool)
    for z in range(score3d.shape[0]):
        vi = np.flatnonzero(valid3d[z].ravel())
        if vi.size == 0:
            continue
        k = max(min_per_slice, int(vi.size * ratio))
        if k >= vi.size:
            chosen = vi
        else:
            vals = score3d[z].ravel()[vi]
            chosen = vi[np.argpartition(vals, -k)[-k:]]
        out[z].ravel()[chosen] = True
    return out


def max_lung_width_px(mask3d):
    w = []
    for z in range(mask3d.shape[0]):
        _, xs = np.where(mask3d[z])
        if xs.size:
            w.append(int(xs.max() - xs.min() + 1))
    return max(w) if w else 1


def run_after(cancer_path, healthy_paths, out_dir, make_gif=True, fps=12, frame_stride=6, max_gif_frames=900):
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    hf = [get_sorted_dicom_files(p) for p in healthy_paths]
    cf = get_sorted_dicom_files(cancer_path)
    n = min([len(cf)] + [len(x) for x in hf])
    hf, cf = [x[:n] for x in hf], cf[:n]

    gray_masks, lung_masks, valid_masks, diff_scores = [], [], [], []
    ct_slices_u8 = []
    healthy_ae, cancer_ae = [], []
    counts = np.zeros(n, dtype=np.int32)
    ref = None
    ae_hw = (160, 160)

    for i in range(n):
        c = read_hu(cf[i])
        ref = c.shape if ref is None else ref
        c = resize_2d(c, ref, 1)
        c_n = ct_window_norm(c)
        ct_slices_u8.append((c_n * 255.0).astype(np.uint8))
        h_norms = []
        for x in hf:
            h = resize_2d(read_hu(x[i]), ref, 1)
            hn = ct_window_norm(h)
            h_norms.append(hn)
            healthy_ae.append(resize_2d(hn, ae_hw, 1))
        base = np.median(np.stack(h_norms, axis=0), axis=0)
        diff = np.abs(c_n - base)
        lmask = lung_mask_2d(c)
        gmask = (c_n >= 0.10) & (c_n <= 0.85)
        valid = lmask & gmask
        if valid.sum() < 50:
            valid = lmask
        if valid.sum() < 50:
            valid = gmask
        if valid.sum() < 50:
            valid = np.ones_like(gmask, bool)
        gray_masks.append(gmask)
        lung_masks.append(lmask)
        valid_masks.append(valid)
        diff_scores.append(diff)
        cancer_ae.append(resize_2d(c_n, ae_hw, 1))
        if (i + 1) % 20 == 0:
            gc.collect()

    diff_3d = np.stack(diff_scores, axis=0).astype(np.float32)
    valid_3d = np.stack(valid_masks, axis=0).astype(bool)
    ae_err = np.zeros_like(diff_3d, dtype=np.float32)

    if healthy_ae:
        tf = get_tf()
        x_train = np.stack(healthy_ae, axis=0).astype(np.float32)
        if x_train.shape[0] > 1200:
            idx = np.random.choice(x_train.shape[0], 1200, replace=False)
            x_train = x_train[idx]
        x_train = x_train[..., None]
        x_cancer = np.stack(cancer_ae, axis=0).astype(np.float32)[..., None]
        tf.keras.backend.clear_session()
        ae = build_ae((ae_hw[0], ae_hw[1], 1), 1e-3)
        es = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=2, restore_best_weights=True)
        ae.fit(x_train, x_train, epochs=8, batch_size=8, shuffle=True, verbose=0, callbacks=[es])
        pred = ae.predict(x_cancer, batch_size=8, verbose=0)
        err_small = np.abs(x_cancer - pred)[..., 0]
        for z in range(n):
            ae_err[z] = resize_2d(err_small[z], ref, 1)

    diff_z = np.zeros_like(diff_3d, np.float32)
    ae_z = np.zeros_like(ae_err, np.float32)
    for z in range(n):
        diff_z[z] = robust_positive_z(diff_3d[z], valid_3d[z])
        ae_z[z] = robust_positive_z(ae_err[z], valid_3d[z])
    score = 0.85 * diff_z + 0.15 * ae_z

    lesion_3d, used_pct = None, None
    vals = score[valid_3d] if np.any(valid_3d) else score.ravel()
    for p in [99.2, 98.8, 98.3, 97.8, 97.2, 96.5]:
        hi = float(np.percentile(vals, p))
        lo = float(np.percentile(vals, max(90.0, p - 2.5)))
        seed = (score > hi) & valid_3d
        cand = seed.copy()
        for _ in range(2):
            cand |= ndimage.binary_dilation(cand, structure=np.ones((1, 3, 3))) & (score > lo) & valid_3d
        cand = neurosymbolic_cleanup(cand, 18, 40, 1)
        if int(cand.sum()) >= 40:
            lesion_3d, used_pct = cand, p
            break
    if lesion_3d is None:
        lesion_3d = neurosymbolic_cleanup(topk_fallback_3d(score, valid_3d, 0.006, 14), 18, 40, 1)
        used_pct = -1.0

    lesions = [lesion_3d[z].astype(bool) for z in range(n)]
    for z in range(n):
        lesions[z] = ndimage.binary_dilation(lesions[z], structure=np.ones((3, 3))) & valid_masks[z]
        counts[z] = int(lesions[z].sum())

    active, gray_union = np.zeros(ref, bool), np.zeros(ref, bool)
    for l, g in zip(lesions, gray_masks):
        active |= l
        gray_union |= g
    active &= gray_union
    active = remove_small_blobs(active, 18)

    lung3d = np.stack(lung_masks, axis=0).astype(bool)
    px_per_mm = max_lung_width_px(lung3d) / 300.0
    px_radius = max(1, int(round((5.0 / 2.0) * px_per_mm)))

    gif_path = os.path.join(out_dir, "after_treatment.gif")
    init, treated, rem, pct, steps = run_treatment(
        active,
        px_radius,
        160000,
        make_gif=make_gif,
        gif_path=gif_path,
        ct_slices_u8=ct_slices_u8,
        gray_masks=gray_masks,
        fps=fps,
        frame_stride=frame_stride,
        max_gif_frames=max_gif_frames,
        laser_off_when_idle=True,
    )

    csv_path = os.path.join(out_dir, "slice_detected_counts.csv")
    save_csv(csv_path, counts, "detected_pixels_gray_lung_ai")
    summary = {
        "initial_total_cancer_pixels": init,
        "treated_total_pixels": treated,
        "remaining_total_pixels": rem,
        "percent_removed": round(pct, 2),
        "laser_steps": steps,
        "detection_percentile_used": used_pct,
        "beam_radius_px": px_radius,
        "runtime_seconds": round(time.time() - t0, 2),
        "csv_path": csv_path,
        "gif_path": gif_path,
    }
    return summary


def save_uploaded_cancer_folder(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile], dest_root: str) -> str:
    if os.path.isdir(dest_root):
        shutil.rmtree(dest_root)
    os.makedirs(dest_root, exist_ok=True)

    if not uploaded_files:
        raise RuntimeError("No files uploaded.")

    for i, uf in enumerate(uploaded_files):
        name = os.path.basename(uf.name)
        data = uf.getvalue()
        if name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
                zf.extractall(dest_root)
        else:
            with open(os.path.join(dest_root, f"{i:04d}_{name}"), "wb") as f:
                f.write(data)

    best, best_n = None, -1
    for root, _, fs in os.walk(dest_root):
        n = sum(1 for fn in fs if os.path.isfile(os.path.join(root, fn)))
        if n > best_n:
            best, best_n = root, n
    if best is None or best_n <= 0:
        raise RuntimeError("No uploaded files found.")
    return best


def validate_paths(paths: List[str], label: str):
    for p in paths:
        if not os.path.isdir(p):
            raise FileNotFoundError(f"{label} path missing: {p}")
        _ = get_sorted_dicom_files(p)


def read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def main() -> None:
    st.set_page_config(page_title="PulmoSense", layout="wide")
    st.markdown(
        """
        <h1 style="text-align:center; color:#003366; margin-bottom:0.2rem;">PulmoSense</h1>
        <h3 style="text-align:center; color:#003366; margin-top:0;">
            Using CNN and Neurosymbolic AI To Prevent Radiation-Induced Lung Injuries
        </h3>
        """,
        unsafe_allow_html=True,
    )
    with st.sidebar:
        st.subheader("Healthy Folders")
        healthy_root = st.text_input("Healthy root", value=DEFAULT_HEALTHY_ROOT)
        before_paths = [os.path.join(healthy_root, "trial1"), os.path.join(healthy_root, "trial2"), os.path.join(healthy_root, "trial3")]
        after_paths = [os.path.join(healthy_root, "trial1"), os.path.join(healthy_root, "trial2"), os.path.join(healthy_root, "trial3")]
        st.code("\n".join(before_paths), language="text")

        st.subheader("GIF")
        make_gif = st.checkbox("Generate GIFs", value=True)
        fps = st.number_input("FPS", min_value=1, max_value=30, value=12)
        frame_stride = st.number_input("Frame stride", min_value=1, max_value=30, value=6)
        max_gif_frames = st.number_input("Max GIF frames", min_value=50, max_value=2000, value=900)

        output_root = st.text_input("Output root", value="outputs")

    st.markdown("<div style='height:1.25rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size:1.05rem; font-weight:600; margin:0 0 0.35rem 0;'>"
        "Upload cancer DICOM files"
        "</p>",
        unsafe_allow_html=True,
    )
    uploads = st.file_uploader(
        "Upload cancer files",
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    run_btn = st.button("Run Comparison", type="primary", use_container_width=True)

    if not run_btn:
        st.info("Upload cancer files and click Run Comparison.")
        return

    if not uploads:
        st.error("Please upload cancer files first.")
        return

    seed_everything(42)

    try:
        validate_paths(before_paths, "before")
        validate_paths(after_paths, "after")
    except Exception as e:
        st.error(str(e))
        st.info("Place healthy references in `data/trial1`, `data/trial2`, `data/trial3`.")
        return

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_root, run_id)
    before_dir = os.path.join(run_dir, "before")
    after_dir = os.path.join(run_dir, "after")
    upload_dir = os.path.join(run_dir, "uploaded_cancer")
    os.makedirs(run_dir, exist_ok=True)

    with st.status("Running Colab-equivalent pipelines...", expanded=True) as status:
        try:
            st.write("Saving cancer upload...")
            cancer_path = save_uploaded_cancer_folder(uploads, upload_dir)
            st.write(f"Cancer folder: `{cancer_path}`")

            st.write("Running BEFORE...")
            before = run_before(
                cancer_path,
                before_paths,
                before_dir,
                make_gif=make_gif,
                fps=int(fps),
                frame_stride=int(frame_stride),
                max_gif_frames=int(max_gif_frames),
            )

            st.write("Running AFTER...")
            after = run_after(
                cancer_path,
                after_paths,
                after_dir,
                make_gif=make_gif,
                fps=int(fps),
                frame_stride=int(frame_stride),
                max_gif_frames=int(max_gif_frames),
            )

            status.update(label="Completed", state="complete")
        except Exception as e:
            status.update(label="Failed", state="error")
            st.exception(e)
            return

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Before")
        if make_gif and os.path.exists(before["gif_path"]):
            st.image(before["gif_path"])
            st.download_button(
                "Download before_treatment.gif",
                data=read_bytes(before["gif_path"]),
                file_name="before_treatment.gif",
                mime="image/gif",
            )
        else:
            st.info("Before GIF not generated.")

    with col2:
        st.subheader("After")
        if make_gif and os.path.exists(after["gif_path"]):
            st.image(after["gif_path"])
            st.download_button(
                "Download after_treatment.gif",
                data=read_bytes(after["gif_path"]),
                file_name="after_treatment.gif",
                mime="image/gif",
            )
        else:
            st.info("After GIF not generated.")

    payload = {
        "run_id": run_id,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "healthy_root": healthy_root,
        "cancer_path": cancer_path,
        "before": before,
        "after": after,
    }
    payload_path = os.path.join(run_dir, "model_output.json")
    with open(payload_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    st.success(f"Saved outputs to `{run_dir}`")


if __name__ == "__main__":
    main()
