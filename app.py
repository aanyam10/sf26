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
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pydicom
import streamlit as st
from matplotlib import animation, pyplot as plt
from scipy import ndimage


# -----------------------------------------------------------------------------
# Shared utilities
# -----------------------------------------------------------------------------


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        tf = get_tf()
        tf.random.set_seed(seed)
    except Exception:
        # BEFORE flow can run without TensorFlow.
        pass


def get_tf():
    try:
        import tensorflow as tf
    except Exception as e:
        raise RuntimeError(
            "TensorFlow could not be imported. Ensure `tensorflow` is installed in the deployment environment."
        ) from e
    return tf


def get_sorted_dicom_files(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")

    files = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    if not files:
        raise FileNotFoundError(f"No files in {folder}")

    def sort_key(path: str):
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
            if hasattr(ds, "InstanceNumber"):
                return (0, int(ds.InstanceNumber))
            if hasattr(ds, "ImagePositionPatient"):
                return (1, float(ds.ImagePositionPatient[2]))
        except Exception:
            pass
        return (2, os.path.basename(path))

    return sorted(files, key=sort_key)


def read_hu(path: str) -> np.ndarray:
    ds = pydicom.dcmread(path, force=True)
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    return arr * slope + intercept


def get_pixel_spacing_mm(path: str) -> Tuple[float, float]:
    ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
    px = getattr(ds, "PixelSpacing", [1.0, 1.0])
    return float(px[0]), float(px[1])


def resize_2d(img: np.ndarray, target_shape: Tuple[int, int], order: int = 1) -> np.ndarray:
    if img.shape == target_shape:
        return img
    zoom = (target_shape[0] / img.shape[0], target_shape[1] / img.shape[1])
    return ndimage.zoom(img, zoom=zoom, order=order)


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


def disk_mask(shape: Tuple[int, int], cy: int, cx: int, radius: int) -> np.ndarray:
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= (radius ** 2)


def bresenham_line(y0: int, x0: int, y1: int, x1: int) -> List[Tuple[int, int]]:
    points = []
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        points.append((y0, x0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
    return points


def nearest_active(mask: np.ndarray, yx: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    d2 = (coords[:, 0] - yx[0]) ** 2 + (coords[:, 1] - yx[1]) ** 2
    j = int(np.argmin(d2))
    return int(coords[j, 0]), int(coords[j, 1])


def breathing_order(n: int) -> List[int]:
    if n <= 1:
        return [0]
    return list(range(n)) + list(range(n - 2, -1, -1))


def as_py(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def save_uploaded_files(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile], dest_root: str) -> str:
    if os.path.isdir(dest_root):
        shutil.rmtree(dest_root)
    os.makedirs(dest_root, exist_ok=True)

    if not uploaded_files:
        raise RuntimeError("No uploaded files received.")

    for i, uf in enumerate(uploaded_files):
        name = os.path.basename(uf.name)
        raw = uf.getvalue()

        if name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(raw), "r") as zf:
                zf.extractall(dest_root)
        else:
            # Prefix index to avoid accidental filename collisions.
            out_name = f"{i:04d}_{name}"
            out_path = os.path.join(dest_root, out_name)
            with open(out_path, "wb") as f:
                f.write(raw)

    # Pick the deepest folder with the most files.
    best_dir = None
    best_count = -1
    for root, _, files in os.walk(dest_root):
        count = sum(1 for fn in files if os.path.isfile(os.path.join(root, fn)))
        if count > best_count:
            best_count = count
            best_dir = root

    if best_dir is None or best_count <= 0:
        raise RuntimeError("No readable files were found after upload extraction.")

    return best_dir


def save_uploaded_cancer_files(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile], dest_root: str) -> str:
    return save_uploaded_files(uploaded_files, dest_root)


def parse_paths(text: str) -> List[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def evenly_spaced_indices(total: int, limit: int) -> List[int]:
    if limit <= 0 or total <= limit:
        return list(range(total))
    idx = np.linspace(0, total - 1, num=limit)
    idx = np.round(idx).astype(int)
    idx = np.clip(idx, 0, total - 1)
    # Preserve order but remove duplicates from rounding.
    return list(dict.fromkeys(idx.tolist()))


def validate_healthy_paths(paths: List[str], label: str) -> None:
    if not paths:
        raise ValueError(f"{label} requires at least one healthy folder path.")
    for p in paths:
        if not os.path.isdir(p):
            raise FileNotFoundError(f"{label} path missing: {p}")
        _ = get_sorted_dicom_files(p)


def parse_saved_payload(uploaded_json) -> Dict[str, object]:
    raw = uploaded_json.getvalue().decode("utf-8")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("model_output.json must contain a JSON object.")
    if "before" not in payload or "after" not in payload:
        raise ValueError("model_output.json must include both `before` and `after` sections.")
    if not isinstance(payload["before"], dict) or not isinstance(payload["after"], dict):
        raise ValueError("`before` and `after` fields must be JSON objects.")
    return payload


def build_comparison_df(before: Dict[str, object], after: Dict[str, object]) -> pd.DataFrame:
    metrics = [
        "initial_total_cancer_pixels",
        "treated_total_pixels",
        "remaining_total_pixels",
        "percent_removed",
        "laser_steps",
        "stopped_when_all_cancer_gone",
        "runtime_seconds",
    ]
    return pd.DataFrame(
        {
            "metric": metrics,
            "before": [before.get(m, "N/A") for m in metrics],
            "after": [after.get(m, "N/A") for m in metrics],
        }
    ).set_index("metric")


def render_comparison_outputs(before: Dict[str, object], after: Dict[str, object]) -> pd.DataFrame:
    cmp_df = build_comparison_df(before, after)

    st.subheader("Metrics")
    st.dataframe(cmp_df, use_container_width=True)

    model_path = str(after.get("model_path", "") or "")
    if after.get("model_retrained"):
        st.warning(f"AFTER model retrained and saved to `{model_path}`")
    elif model_path:
        st.success(f"AFTER model loaded from cache: `{model_path}`")

    if model_path and os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model_blob = f.read()
        st.download_button(
            "Download AFTER model (.keras)",
            data=model_blob,
            file_name=Path(model_path).name,
            mime="application/octet-stream",
        )

        repo_path = os.path.relpath(model_path, os.getcwd())
        with st.expander("Save model to GitHub"):
            st.code(
                "\n".join(
                    [
                        "git lfs install",
                        "git add .gitattributes",
                        f"git add {repo_path}",
                        'git commit -m "Add cached AFTER model"',
                        "git push",
                    ]
                ),
                language="bash",
            )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Before")
        before_gif = str(before.get("gif_path", "") or "")
        if before_gif and os.path.exists(before_gif):
            st.image(before_gif, use_container_width=True)
        else:
            st.info("Before GIF not found at saved path.")
        st.caption(f"GIF: `{before_gif}`")

    with col2:
        st.markdown("### After")
        after_gif = str(after.get("gif_path", "") or "")
        if after_gif and os.path.exists(after_gif):
            st.image(after_gif, use_container_width=True)
        else:
            st.info("After GIF not found at saved path.")
        st.caption(f"GIF: `{after_gif}`")

    return cmp_df


# -----------------------------------------------------------------------------
# Treatment + GIF renderer
# -----------------------------------------------------------------------------


def run_treatment(
    active_global: np.ndarray,
    px_radius: int,
    max_steps: int = 160000,
    make_gif: bool = False,
    gif_path: Optional[str] = None,
    ct_slices_u8: Optional[List[np.ndarray]] = None,
    gray_masks: Optional[List[np.ndarray]] = None,
    fps: int = 12,
    frame_stride: int = 6,
    max_gif_frames: int = 900,
    laser_off_when_idle: bool = False,
) -> Tuple[int, int, int, float, int]:
    steps = 0
    treated = 0
    initial = int(active_global.sum())

    writer = fig = ax = im = dot = None
    captured = 0
    capture_step = 0
    breath_seq = [0]

    if (
        make_gif
        and gif_path
        and ct_slices_u8 is not None
        and gray_masks is not None
        and len(ct_slices_u8) == len(gray_masks)
        and len(ct_slices_u8) > 0
    ):
        breath_seq = breathing_order(len(ct_slices_u8))
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.axis("off")

        first_bg = breath_seq[0]
        bg = ct_slices_u8[first_bg].astype(np.float32) / 255.0
        frame = np.repeat(bg[..., None], 3, axis=2)
        visible = active_global & gray_masks[first_bg]
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

        def draw_frame(bg_idx: int, show_dot: bool = False, pos: Tuple[int, int] = (0, 0), color: str = "red") -> None:
            bg0 = ct_slices_u8[bg_idx].astype(np.float32) / 255.0
            rgb = np.repeat(bg0[..., None], 3, axis=2)
            visible_overlay = active_global & gray_masks[bg_idx]
            rgb[visible_overlay, 0] = 1.0
            rgb[visible_overlay, 1] = 1.0
            rgb[visible_overlay, 2] = 0.0
            im.set_data(rgb)

            if show_dot:
                py, px = pos
                dot.center = (px, py)
                dot.set_edgecolor(color)
                dot.set_visible(True)
            else:
                dot.set_visible(False)

    if initial > 0:
        coords = np.argwhere(active_global)
        y, x = np.mean(coords, axis=0).astype(int)
    else:
        y, x = active_global.shape[0] // 2, active_global.shape[1] // 2

    while active_global.any() and steps < max_steps:
        target = nearest_active(active_global, (y, x))
        if target is None:
            break
        ty, tx = target

        for py, px in bresenham_line(y, x, ty, tx):
            steps += 1
            hit = active_global & disk_mask(active_global.shape, py, px, px_radius)
            laser_on = np.any(hit)

            if laser_on:
                n = int(hit.sum())
                active_global[hit] = False
                treated += n

            if writer is not None and (steps % frame_stride == 0) and (captured < max_gif_frames):
                bg_idx = breath_seq[capture_step % len(breath_seq)]
                if laser_off_when_idle and not laser_on:
                    draw_frame(bg_idx, show_dot=False)
                else:
                    draw_frame(bg_idx, show_dot=True, pos=(py, px), color=("red" if laser_on else "blue"))
                writer.grab_frame()
                captured += 1
                capture_step += 1

            if not active_global.any() or steps >= max_steps:
                break

        y, x = ty, tx

    if writer is not None:
        if captured < max_gif_frames:
            bg_idx = breath_seq[capture_step % len(breath_seq)]
            draw_frame(
                bg_idx,
                show_dot=False if laser_off_when_idle else True,
                pos=(y, x),
                color="red",
            )
            writer.grab_frame()
        writer.finish()
        plt.close(fig)

    remaining = int(active_global.sum())
    removed_pct = (100.0 * treated / initial) if initial > 0 else 0.0
    return initial, treated, remaining, removed_pct, steps


# -----------------------------------------------------------------------------
# BEFORE pipeline
# -----------------------------------------------------------------------------


def detect_lesion_before(
    c_hu: np.ndarray,
    healthy_hus: List[np.ndarray],
    zthr: float = 2.5,
    gray_min: float = 0.10,
    gray_max: float = 0.85,
    min_px: int = 40,
) -> np.ndarray:
    c_n = ct_window_norm(c_hu)
    h_n = [ct_window_norm(h) for h in healthy_hus]
    baseline = np.median(np.stack(h_n, axis=0), axis=0)
    diff = c_n - baseline
    med = np.median(diff)
    mad = np.median(np.abs(diff - med)) + 1e-6
    z = 0.6745 * (diff - med) / mad

    lesion = (z > zthr) & lung_mask_2d(c_hu) & (c_n >= gray_min) & (c_n <= gray_max)
    lesion = ndimage.binary_opening(lesion, structure=np.ones((3, 3)))
    lesion = ndimage.binary_closing(lesion, structure=np.ones((5, 5)))
    lesion = remove_small_blobs(lesion, min_px)
    return lesion.astype(bool)


def run_before(
    cancer_path: str,
    healthy_paths: List[str],
    out_dir: str,
    make_gif: bool,
    fps: int,
    frame_stride: int,
    max_gif_frames: int,
    max_slices: int,
) -> Dict[str, object]:
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    healthy_files = [get_sorted_dicom_files(p) for p in healthy_paths]
    cancer_files = get_sorted_dicom_files(cancer_path)
    n = min([len(cancer_files)] + [len(hf) for hf in healthy_files])
    if n == 0:
        raise RuntimeError("No slices found.")
    healthy_files = [hf[:n] for hf in healthy_files]
    cancer_files = cancer_files[:n]
    original_n = n

    selected = evenly_spaced_indices(n, max_slices)
    if len(selected) < n:
        healthy_files = [[hf[i] for i in selected] for hf in healthy_files]
        cancer_files = [cancer_files[i] for i in selected]
        n = len(selected)

    y_mm, x_mm = get_pixel_spacing_mm(cancer_files[0])
    px_radius = max(1, int(round(2.5 / ((y_mm + x_mm) / 2.0))))

    lesions, grays, ct_slices_u8 = [], [], []
    counts = np.zeros(n, dtype=np.int32)
    ref_shape = None

    for i in range(n):
        c = read_hu(cancer_files[i])
        if ref_shape is None:
            ref_shape = c.shape
        c = resize_2d(c, ref_shape, order=1)
        c_n = ct_window_norm(c)

        healthy_hus = [resize_2d(read_hu(hf[i]), ref_shape, order=1) for hf in healthy_files]
        lesion = detect_lesion_before(c, healthy_hus)
        gray = (c_n >= 0.10) & (c_n <= 0.85)
        lesion = lesion & gray

        lesions.append(lesion)
        grays.append(gray)
        ct_slices_u8.append((c_n * 255.0).astype(np.uint8))
        counts[i] = int(lesion.sum())

        if (i + 1) % 25 == 0:
            gc.collect()

    active = np.zeros(ref_shape, dtype=bool)
    gray_union = np.zeros(ref_shape, dtype=bool)
    for lesion, gray in zip(lesions, grays):
        active |= lesion
        gray_union |= gray
    active &= gray_union
    active = remove_small_blobs(active, 40)

    gif_path = os.path.join(out_dir, "before_treatment.gif")
    init, treated, rem, pct, steps = run_treatment(
        active,
        px_radius,
        max_steps=160000,
        make_gif=make_gif,
        gif_path=gif_path,
        ct_slices_u8=ct_slices_u8,
        gray_masks=grays,
        fps=fps,
        frame_stride=frame_stride,
        max_gif_frames=max_gif_frames,
        laser_off_when_idle=False,
    )

    csv_path = os.path.join(out_dir, "slice_detected_counts.csv")
    pd.DataFrame({"slice_index": np.arange(n), "detected_pixels_gray_only": counts}).to_csv(csv_path, index=False)

    summary = {
        "original_slice_count": original_n,
        "processed_slice_count": n,
        "initial_total_cancer_pixels": init,
        "treated_total_pixels": treated,
        "remaining_total_pixels": rem,
        "percent_removed": round(float(pct), 2),
        "laser_steps": steps,
        "stopped_when_all_cancer_gone": bool(rem == 0),
        "runtime_seconds": round(float(time.time() - t0), 2),
        "csv_path": csv_path,
        "output_dir": out_dir,
        "gif_path": gif_path,
    }
    return {k: as_py(v) for k, v in summary.items()}


# -----------------------------------------------------------------------------
# AFTER pipeline (with model caching)
# -----------------------------------------------------------------------------


def remove_small_components_3d(mask3d: np.ndarray, min_vox: int = 40, min_span: int = 1) -> np.ndarray:
    lab, n = ndimage.label(mask3d)
    if n == 0:
        return np.zeros_like(mask3d, dtype=bool)
    keep_ids = []
    for cid in range(1, n + 1):
        pts = np.argwhere(lab == cid)
        if pts.shape[0] < min_vox:
            continue
        zspan = int(pts[:, 0].max() - pts[:, 0].min() + 1)
        if zspan < min_span:
            continue
        keep_ids.append(cid)
    if not keep_ids:
        return np.zeros_like(mask3d, dtype=bool)
    return np.isin(lab, keep_ids)


def temporal_majority_3d(mask3d: np.ndarray, min_support: int = 2) -> np.ndarray:
    m = mask3d.astype(np.uint8)
    acc = m.copy()
    acc[1:] += m[:-1]
    acc[:-1] += m[1:]
    out = acc >= min_support
    if out.sum() == 0 and mask3d.sum() > 0:
        return mask3d
    return out


def robust_positive_z(arr: np.ndarray, valid: np.ndarray) -> np.ndarray:
    vals = arr[valid] if np.any(valid) else arr.ravel()
    med = np.median(vals)
    mad = np.median(np.abs(vals - med)) + 1e-6
    z = 0.6745 * (arr - med) / mad
    return np.clip(z, 0.0, None).astype(np.float32)


def build_ae(shape: Tuple[int, int, int] = (160, 160, 1), lr: float = 1e-3):
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


def neurosymbolic_cleanup(mask3d: np.ndarray, min_px: int = 18, min_vox: int = 40, min_span: int = 1) -> np.ndarray:
    out = np.zeros_like(mask3d, dtype=bool)
    for z in range(mask3d.shape[0]):
        m = ndimage.binary_opening(mask3d[z], structure=np.ones((3, 3)))
        m = ndimage.binary_closing(m, structure=np.ones((5, 5)))
        out[z] = remove_small_blobs(m, min_px)
    out = temporal_majority_3d(out, min_support=2)
    out3 = remove_small_components_3d(out, min_vox=min_vox, min_span=min_span)
    return out3 if out3.sum() > 0 else out


def topk_fallback_3d(score3d: np.ndarray, valid3d: np.ndarray, ratio: float = 0.006, min_per_slice: int = 14) -> np.ndarray:
    out = np.zeros_like(valid3d, dtype=bool)
    for z in range(score3d.shape[0]):
        valid_idx = np.flatnonzero(valid3d[z].ravel())
        if valid_idx.size == 0:
            continue
        k = max(min_per_slice, int(valid_idx.size * ratio))
        if k >= valid_idx.size:
            chosen = valid_idx
        else:
            vals = score3d[z].ravel()[valid_idx]
            chosen = valid_idx[np.argpartition(vals, -k)[-k:]]
        out[z].ravel()[chosen] = True
    return out


def max_lung_width_px(mask3d: np.ndarray) -> int:
    widths = []
    for z in range(mask3d.shape[0]):
        _, xs = np.where(mask3d[z])
        if xs.size:
            widths.append(int(xs.max() - xs.min() + 1))
    return max(widths) if widths else 1


@st.cache_resource(show_spinner=False)
def load_model_cached(path: str):
    tf = get_tf()
    return tf.keras.models.load_model(path, compile=False)


def load_or_train_after_model(
    healthy_ae: List[np.ndarray],
    ae_hw: Tuple[int, int],
    model_path: str,
    force_retrain: bool = False,
    max_train_samples: int = 600,
    train_epochs: int = 4,
):
    tf = get_tf()
    if not healthy_ae:
        raise RuntimeError("No healthy slices available to train/load AFTER model.")

    abs_path = os.path.abspath(model_path) if model_path else ""
    if abs_path and os.path.exists(abs_path) and not force_retrain:
        return load_model_cached(abs_path), False

    x_train = np.stack(healthy_ae, axis=0).astype(np.float32)
    if x_train.shape[0] > max_train_samples:
        idx = np.random.choice(x_train.shape[0], max_train_samples, replace=False)
        x_train = x_train[idx]
    x_train = x_train[..., None]

    tf.keras.backend.clear_session()
    ae = build_ae((ae_hw[0], ae_hw[1], 1), 1e-3)
    es = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=2, restore_best_weights=True)
    ae.fit(x_train, x_train, epochs=train_epochs, batch_size=8, shuffle=True, verbose=0, callbacks=[es])

    if abs_path:
        Path(abs_path).parent.mkdir(parents=True, exist_ok=True)
        ae.save(abs_path)
        load_model_cached.clear()
        return load_model_cached(abs_path), True

    return ae, True


def run_after(
    cancer_path: str,
    healthy_paths: List[str],
    out_dir: str,
    model_path: str,
    force_retrain_model: bool,
    make_gif: bool,
    fps: int,
    frame_stride: int,
    max_gif_frames: int,
    max_slices: int,
    ae_train_epochs: int,
    ae_max_train_samples: int,
) -> Dict[str, object]:
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    healthy_files = [get_sorted_dicom_files(p) for p in healthy_paths]
    cancer_files = get_sorted_dicom_files(cancer_path)
    n = min([len(cancer_files)] + [len(hf) for hf in healthy_files])
    if n == 0:
        raise RuntimeError("No slices found.")
    healthy_files = [hf[:n] for hf in healthy_files]
    cancer_files = cancer_files[:n]
    original_n = n

    selected = evenly_spaced_indices(n, max_slices)
    if len(selected) < n:
        healthy_files = [[hf[i] for i in selected] for hf in healthy_files]
        cancer_files = [cancer_files[i] for i in selected]
        n = len(selected)

    gray_masks, lung_masks, valid_masks, diff_scores = [], [], [], []
    ct_slices_u8, healthy_ae, cancer_ae = [], [], []
    counts = np.zeros(n, dtype=np.int32)
    ref_shape = None
    ae_hw = (160, 160)

    for i in range(n):
        c = read_hu(cancer_files[i])
        if ref_shape is None:
            ref_shape = c.shape
        c = resize_2d(c, ref_shape, order=1)
        c_n = ct_window_norm(c)
        ct_slices_u8.append((c_n * 255.0).astype(np.uint8))

        healthy_norms = []
        for hf in healthy_files:
            h = resize_2d(read_hu(hf[i]), ref_shape, order=1)
            h_n = ct_window_norm(h)
            healthy_norms.append(h_n)
            healthy_ae.append(resize_2d(h_n, ae_hw, order=1))

        baseline = np.median(np.stack(healthy_norms, axis=0), axis=0)
        diff = np.abs(c_n - baseline)

        lmask = lung_mask_2d(c)
        gmask = (c_n >= 0.10) & (c_n <= 0.85)
        valid = lmask & gmask
        if valid.sum() < 50:
            valid = lmask
        if valid.sum() < 50:
            valid = gmask
        if valid.sum() < 50:
            valid = np.ones_like(gmask, dtype=bool)

        gray_masks.append(gmask)
        lung_masks.append(lmask)
        valid_masks.append(valid)
        diff_scores.append(diff)
        cancer_ae.append(resize_2d(c_n, ae_hw, order=1))

        if (i + 1) % 25 == 0:
            gc.collect()

    diff_3d = np.stack(diff_scores, axis=0).astype(np.float32)
    valid_3d = np.stack(valid_masks, axis=0).astype(bool)
    ae_err = np.zeros_like(diff_3d, dtype=np.float32)

    model, trained_now = load_or_train_after_model(
        healthy_ae=healthy_ae,
        ae_hw=ae_hw,
        model_path=model_path,
        force_retrain=force_retrain_model,
        max_train_samples=ae_max_train_samples,
        train_epochs=ae_train_epochs,
    )
    x_cancer = np.stack(cancer_ae, axis=0).astype(np.float32)[..., None]
    pred = model.predict(x_cancer, batch_size=8, verbose=0)
    err_small = np.abs(x_cancer - pred)[..., 0]
    for z in range(n):
        ae_err[z] = resize_2d(err_small[z], ref_shape, order=1)

    diff_z = np.zeros_like(diff_3d, dtype=np.float32)
    ae_z = np.zeros_like(ae_err, dtype=np.float32)
    for z in range(n):
        diff_z[z] = robust_positive_z(diff_3d[z], valid_3d[z])
        ae_z[z] = robust_positive_z(ae_err[z], valid_3d[z])
    score = 0.85 * diff_z + 0.15 * ae_z

    lesion_3d = None
    used_pct = None
    vals = score[valid_3d] if np.any(valid_3d) else score.ravel()
    for p in [99.2, 98.8, 98.3, 97.8, 97.2, 96.5]:
        hi = float(np.percentile(vals, p))
        lo = float(np.percentile(vals, max(90.0, p - 2.5)))
        seed = (score > hi) & valid_3d
        cand = seed.copy()
        for _ in range(2):
            cand |= ndimage.binary_dilation(cand, structure=np.ones((1, 3, 3))) & (score > lo) & valid_3d
        cand = neurosymbolic_cleanup(cand, min_px=18, min_vox=40, min_span=1)
        if int(cand.sum()) >= 40:
            lesion_3d = cand
            used_pct = p
            break

    if lesion_3d is None:
        lesion_3d = neurosymbolic_cleanup(
            topk_fallback_3d(score, valid_3d, ratio=0.006, min_per_slice=14),
            min_px=18,
            min_vox=40,
            min_span=1,
        )
        used_pct = -1.0

    lesions = [lesion_3d[z].astype(bool) for z in range(n)]
    for z in range(n):
        lesions[z] = ndimage.binary_dilation(lesions[z], structure=np.ones((3, 3))) & valid_masks[z]
        counts[z] = int(lesions[z].sum())

    active = np.zeros(ref_shape, dtype=bool)
    gray_union = np.zeros(ref_shape, dtype=bool)
    for lesion, gray in zip(lesions, gray_masks):
        active |= lesion
        gray_union |= gray
    active &= gray_union
    active = remove_small_blobs(active, 18)

    lung3d = np.stack(lung_masks, axis=0).astype(bool)
    px_per_mm = max_lung_width_px(lung3d) / 300.0
    px_radius = max(1, int(round((5.0 / 2.0) * px_per_mm)))

    gif_path = os.path.join(out_dir, "after_treatment.gif")
    init, treated, rem, pct, steps = run_treatment(
        active,
        px_radius,
        max_steps=160000,
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
    pd.DataFrame({"slice_index": np.arange(n), "detected_pixels_gray_lung_ai": counts}).to_csv(csv_path, index=False)

    summary = {
        "original_slice_count": original_n,
        "processed_slice_count": n,
        "initial_total_cancer_pixels": init,
        "treated_total_pixels": treated,
        "remaining_total_pixels": rem,
        "percent_removed": round(float(pct), 2),
        "laser_steps": steps,
        "stopped_when_all_cancer_gone": bool(rem == 0),
        "detection_percentile_used": used_pct,
        "beam_radius_px": px_radius,
        "runtime_seconds": round(float(time.time() - t0), 2),
        "csv_path": csv_path,
        "output_dir": out_dir,
        "gif_path": gif_path,
        "model_path": os.path.abspath(model_path) if model_path else "",
        "model_retrained": bool(trained_now),
    }
    return {k: as_py(v) for k, v in summary.items()}


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="Before vs After DICOM Comparator", layout="wide")
    st.title("Before vs After DICOM Comparator")
    st.caption("Upload one cancer case once, run both pipelines, and compare GIFs side by side.")

    with st.sidebar:
        st.subheader("Healthy Data Source")
        healthy_input_mode = st.radio(
            "How should healthy reference data be provided?",
            options=["Upload healthy DICOM files", "Local folder paths"],
            index=0,
        )

        before_paths_raw = ""
        after_paths_raw = ""
        if healthy_input_mode == "Local folder paths":
            st.subheader("Healthy Paths")
            before_paths_raw = st.text_area(
                "BEFORE healthy folders (one per line)",
                value="/path/to/trial1\n/path/to/trial2\n/path/to/trial3",
                height=110,
            )
            after_paths_raw = st.text_area(
                "AFTER healthy folders (one per line)",
                value="/path/to/trial1\n/path/to/trial2\n/path/to/trial3",
                height=110,
            )
        else:
            st.caption("Use the Run tab to upload healthy trial files for trial1/trial2/trial3.")

        st.subheader("Run Settings")
        output_root = st.text_input("Output root", value="outputs")
        model_path = st.text_input("Cached AFTER model path", value="model_cache/after_ae_model.keras")
        force_retrain = st.checkbox("Force retrain AFTER model", value=False)

        st.subheader("GIF")
        make_gif = st.checkbox("Generate GIFs", value=False)
        fps = st.number_input("FPS", min_value=1, max_value=30, value=12)
        frame_stride = st.number_input("Frame stride", min_value=1, max_value=30, value=6)
        max_gif_frames = st.number_input("Max GIF frames", min_value=50, max_value=5000, value=900)

        st.subheader("Performance")
        max_slices = st.number_input(
            "Max slices to process per run",
            min_value=20,
            max_value=1200,
            value=120,
            help="Lower this if the app crashes due to memory limits.",
        )
        ae_train_epochs = st.number_input(
            "AFTER model train epochs (used only when retraining)",
            min_value=1,
            max_value=12,
            value=4,
        )
        ae_max_train_samples = st.number_input(
            "AFTER max training samples",
            min_value=100,
            max_value=2000,
            value=600,
        )

        run_btn = st.button("Run Comparison", type="primary", use_container_width=True)

    run_tab, load_tab = st.tabs(["Run New Comparison", "Load Saved model_output.json"])

    with load_tab:
        st.caption("Use an existing run payload to populate the UI without rerunning the model.")
        saved_payload_file = st.file_uploader(
            "Upload model_output.json",
            type=["json"],
            key="saved_model_output_json",
        )
        render_saved_btn = st.button("Render Saved Output", use_container_width=True, key="render_saved_btn")

        if render_saved_btn:
            if not saved_payload_file:
                st.error("Please upload a model_output.json file first.")
            else:
                try:
                    payload = parse_saved_payload(saved_payload_file)
                    before_saved = payload["before"]
                    after_saved = payload["after"]
                    render_comparison_outputs(before_saved, after_saved)
                except Exception as e:
                    st.exception(e)

    with run_tab:
        uploads = st.file_uploader(
            "Upload cancer DICOM files or ZIP (single case)",
            accept_multiple_files=True,
            key="cancer_uploads",
        )

        before_trial_uploads = []
        after_trial_uploads = []
        reuse_before_for_after = True
        if healthy_input_mode == "Upload healthy DICOM files":
            st.markdown("#### Upload Healthy Reference Trials")
            st.caption("Upload each trial as DICOM files and/or a ZIP archive.")
            before_trial_uploads = [
                st.file_uploader(
                    "BEFORE trial1",
                    accept_multiple_files=True,
                    key="before_trial1_uploads",
                ),
                st.file_uploader(
                    "BEFORE trial2",
                    accept_multiple_files=True,
                    key="before_trial2_uploads",
                ),
                st.file_uploader(
                    "BEFORE trial3",
                    accept_multiple_files=True,
                    key="before_trial3_uploads",
                ),
            ]
            reuse_before_for_after = st.checkbox(
                "Use the same healthy uploads for AFTER",
                value=True,
                key="reuse_before_uploads_for_after",
            )
            if not reuse_before_for_after:
                after_trial_uploads = [
                    st.file_uploader(
                        "AFTER trial1",
                        accept_multiple_files=True,
                        key="after_trial1_uploads",
                    ),
                    st.file_uploader(
                        "AFTER trial2",
                        accept_multiple_files=True,
                        key="after_trial2_uploads",
                    ),
                    st.file_uploader(
                        "AFTER trial3",
                        accept_multiple_files=True,
                        key="after_trial3_uploads",
                    ),
                ]
        else:
            st.caption("Using local healthy paths configured in the sidebar.")

        if not run_btn:
            st.info("Set paths and upload files, then click Run Comparison.")
            return

        if not uploads:
            st.error("Please upload cancer files first.")
            return

        seed_everything(42)

        run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(output_root, run_id)
        before_dir = os.path.join(run_dir, "before")
        after_dir = os.path.join(run_dir, "after")
        upload_dir = os.path.join(run_dir, "uploaded_cancer")
        os.makedirs(run_dir, exist_ok=True)

        before_paths: List[str] = []
        after_paths: List[str] = []

        if healthy_input_mode == "Local folder paths":
            before_paths = parse_paths(before_paths_raw)
            after_paths = parse_paths(after_paths_raw)
            try:
                validate_healthy_paths(before_paths, "BEFORE")
                validate_healthy_paths(after_paths, "AFTER")
            except Exception as e:
                st.error(str(e))
                return
        else:
            if any(len(files) == 0 for files in before_trial_uploads):
                st.error("Please upload healthy files for BEFORE trial1, trial2, and trial3.")
                return
            if not reuse_before_for_after and any(len(files) == 0 for files in after_trial_uploads):
                st.error("Please upload healthy files for AFTER trial1, trial2, and trial3.")
                return

        with st.status("Running pipelines...", expanded=True) as status:
            try:
                st.write("Saving cancer uploads...")
                cancer_path = save_uploaded_files(uploads, upload_dir)
                st.write(f"Cancer input folder: `{cancer_path}`")

                if healthy_input_mode == "Upload healthy DICOM files":
                    healthy_root = os.path.join(run_dir, "uploaded_healthy")
                    before_paths = []
                    for i, files in enumerate(before_trial_uploads, start=1):
                        st.write(f"Saving BEFORE trial{i} uploads...")
                        trial_dir = os.path.join(healthy_root, "before", f"trial{i}")
                        before_paths.append(save_uploaded_files(files, trial_dir))

                    if reuse_before_for_after:
                        after_paths = before_paths.copy()
                    else:
                        after_paths = []
                        for i, files in enumerate(after_trial_uploads, start=1):
                            st.write(f"Saving AFTER trial{i} uploads...")
                            trial_dir = os.path.join(healthy_root, "after", f"trial{i}")
                            after_paths.append(save_uploaded_files(files, trial_dir))

                st.write("Running BEFORE...")
                before = run_before(
                    cancer_path=cancer_path,
                    healthy_paths=before_paths,
                    out_dir=before_dir,
                    make_gif=make_gif,
                    fps=int(fps),
                    frame_stride=int(frame_stride),
                    max_gif_frames=int(max_gif_frames),
                    max_slices=int(max_slices),
                )

                st.write("Running AFTER...")
                after = run_after(
                    cancer_path=cancer_path,
                    healthy_paths=after_paths,
                    out_dir=after_dir,
                    model_path=model_path,
                    force_retrain_model=force_retrain,
                    make_gif=make_gif,
                    fps=int(fps),
                    frame_stride=int(frame_stride),
                    max_gif_frames=int(max_gif_frames),
                    max_slices=int(max_slices),
                    ae_train_epochs=int(ae_train_epochs),
                    ae_max_train_samples=int(ae_max_train_samples),
                )

                status.update(label="Completed", state="complete")
            except Exception as e:
                status.update(label="Failed", state="error")
                st.exception(e)
                return

        cmp_df = render_comparison_outputs(before, after)

        payload = {
            "run_id": run_id,
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "cancer_path": cancer_path,
            "healthy_input_mode": healthy_input_mode,
            "before_healthy_paths": before_paths,
            "after_healthy_paths": after_paths,
            "run_settings": {
                "make_gif": bool(make_gif),
                "fps": int(fps),
                "frame_stride": int(frame_stride),
                "max_gif_frames": int(max_gif_frames),
                "max_slices": int(max_slices),
                "ae_train_epochs": int(ae_train_epochs),
                "ae_max_train_samples": int(ae_max_train_samples),
            },
            "before": before,
            "after": after,
            "comparison_metrics": cmp_df.reset_index().to_dict(orient="records"),
        }
        payload_path = os.path.join(run_dir, "model_output.json")
        with open(payload_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        st.success(f"Saved run outputs to `{run_dir}`")
        st.download_button(
            "Download model_output.json",
            data=json.dumps(payload, indent=2),
            file_name=f"{run_id}_model_output.json",
            mime="application/json",
        )


if __name__ == "__main__":
    main()
