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
from typing import Dict, List, Optional, Tuple

import numpy as np
import pydicom
import streamlit as st
from PIL import Image, ImageDraw
from scipy import ndimage


DEFAULT_MODEL_PATH = "model_cache/after_ae_model.keras"
DEFAULT_OUTPUT_ROOT = "outputs"


def get_tf():
    try:
        import tensorflow as tf
    except Exception as e:
        raise RuntimeError("TensorFlow is required to load and run the model.") from e
    return tf


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        tf = get_tf()
        tf.random.set_seed(seed)
    except Exception:
        # App can still render until model inference is requested.
        pass


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
            out_name = f"{i:04d}_{name}"
            out_path = os.path.join(dest_root, out_name)
            with open(out_path, "wb") as f:
                f.write(raw)

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


def robust_positive_z(arr: np.ndarray, valid: np.ndarray) -> np.ndarray:
    vals = arr[valid] if np.any(valid) else arr.ravel()
    med = np.median(vals)
    mad = np.median(np.abs(vals - med)) + 1e-6
    z = 0.6745 * (arr - med) / mad
    return np.clip(z, 0.0, None).astype(np.float32)


def evenly_spaced_indices(total: int, limit: int) -> List[int]:
    if limit <= 0 or total <= limit:
        return list(range(total))
    idx = np.linspace(0, total - 1, num=limit)
    idx = np.round(idx).astype(int)
    idx = np.clip(idx, 0, total - 1)
    return list(dict.fromkeys(idx.tolist()))


def breathing_order(n: int) -> List[int]:
    if n <= 1:
        return [0]
    return list(range(n)) + list(range(n - 2, -1, -1))


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


@st.cache_resource(show_spinner=False)
def load_model_cached(path: str):
    tf = get_tf()
    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception as load_err:
        try:
            model = build_ae((160, 160, 1), 1e-3)
            model.load_weights(path)
            return model
        except Exception as weights_err:
            raise RuntimeError(
                f"Failed to load model from `{path}`. Original load error: {load_err}"
            ) from weights_err


def prepare_cancer_case(cancer_dir: str, max_slices: int) -> Dict[str, object]:
    dicom_files = get_sorted_dicom_files(cancer_dir)
    if not dicom_files:
        raise RuntimeError("No DICOM files found in upload.")

    selected = evenly_spaced_indices(len(dicom_files), max_slices)
    dicom_files = [dicom_files[i] for i in selected]

    hu_slices: List[np.ndarray] = []
    norm_slices: List[np.ndarray] = []
    u8_slices: List[np.ndarray] = []
    gray_masks: List[np.ndarray] = []
    valid_masks: List[np.ndarray] = []

    ref_shape = None
    for i, f in enumerate(dicom_files):
        hu = read_hu(f)
        if ref_shape is None:
            ref_shape = hu.shape
        hu = resize_2d(hu, ref_shape, order=1)

        norm = ct_window_norm(hu)
        gray = (norm >= 0.10) & (norm <= 0.85)
        lung = lung_mask_2d(hu)
        valid = lung & gray
        if valid.sum() < 50:
            valid = lung
        if valid.sum() < 50:
            valid = gray
        if valid.sum() < 50:
            valid = np.ones_like(gray, dtype=bool)

        hu_slices.append(hu)
        norm_slices.append(norm)
        u8_slices.append((norm * 255.0).astype(np.uint8))
        gray_masks.append(gray)
        valid_masks.append(valid)

        if (i + 1) % 25 == 0:
            gc.collect()

    return {
        "dicom_files": dicom_files,
        "hu_slices": hu_slices,
        "norm_slices": norm_slices,
        "u8_slices": u8_slices,
        "gray_masks": gray_masks,
        "valid_masks": valid_masks,
    }


def detect_lesions_with_model(
    model,
    norm_slices: List[np.ndarray],
    valid_masks: List[np.ndarray],
    min_pixels: int = 18,
) -> List[np.ndarray]:
    if not norm_slices:
        return []

    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    ae_h = int(input_shape[1]) if input_shape[1] else 160
    ae_w = int(input_shape[2]) if input_shape[2] else 160
    ae_hw = (ae_h, ae_w)

    x = np.stack([resize_2d(s, ae_hw, order=1) for s in norm_slices], axis=0).astype(np.float32)[..., None]
    pred = model.predict(x, batch_size=8, verbose=0)
    err_small = np.abs(x - pred)[..., 0]

    lesions: List[np.ndarray] = []
    fallback_scores: List[np.ndarray] = []
    for i in range(len(norm_slices)):
        err = resize_2d(err_small[i], norm_slices[i].shape, order=1)
        z = robust_positive_z(err, valid_masks[i])
        vals = z[valid_masks[i]] if np.any(valid_masks[i]) else z.ravel()

        hi = float(np.percentile(vals, 98.5))
        lo = float(np.percentile(vals, 96.5))

        mask = z > hi
        for _ in range(1):
            mask |= ndimage.binary_dilation(mask, structure=np.ones((3, 3))) & (z > lo)
        mask &= valid_masks[i]
        mask = ndimage.binary_opening(mask, structure=np.ones((3, 3)))
        mask = ndimage.binary_closing(mask, structure=np.ones((5, 5)))
        mask = remove_small_blobs(mask, min_pixels=min_pixels).astype(bool)

        lesions.append(mask)
        fallback_scores.append(z)

    if int(sum(m.sum() for m in lesions)) == 0:
        lesions = []
        for i, score in enumerate(fallback_scores):
            valid = valid_masks[i]
            out = np.zeros_like(valid, dtype=bool)
            valid_idx = np.flatnonzero(valid.ravel())
            if valid_idx.size == 0:
                lesions.append(out)
                continue
            k = max(14, int(valid_idx.size * 0.006))
            vals = score.ravel()[valid_idx]
            if k >= valid_idx.size:
                chosen = valid_idx
            else:
                chosen = valid_idx[np.argpartition(vals, -k)[-k:]]
            out.ravel()[chosen] = True
            out = remove_small_blobs(out, min_pixels=min_pixels).astype(bool)
            lesions.append(out)

    return lesions


def save_gif(frames: List[Image.Image], out_path: str, fps: int) -> None:
    if not frames:
        raise RuntimeError("No frames available to save GIF.")
    duration_ms = max(20, int(round(1000 / max(1, fps))))
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        optimize=False,
        duration=duration_ms,
        loop=0,
    )


def make_before_gif(
    u8_slices: List[np.ndarray],
    lesions: List[np.ndarray],
    out_path: str,
    fps: int,
    max_frames: int,
) -> None:
    order = breathing_order(len(u8_slices))
    if len(order) > max_frames:
        selected = evenly_spaced_indices(len(order), max_frames)
        order = [order[i] for i in selected]

    frames: List[Image.Image] = []
    for idx in order:
        bg = u8_slices[idx]
        rgb = np.repeat(bg[..., None], 3, axis=2)
        m = lesions[idx]
        rgb[m, 0] = 255
        rgb[m, 1] = 255
        rgb[m, 2] = 0
        frames.append(Image.fromarray(rgb))

    save_gif(frames, out_path, fps=fps)


def build_active_map(lesions: List[np.ndarray], gray_masks: List[np.ndarray], min_pixels: int = 18) -> np.ndarray:
    if not lesions:
        raise RuntimeError("Lesion masks are empty.")

    ref_shape = lesions[0].shape
    active = np.zeros(ref_shape, dtype=bool)
    gray_union = np.zeros(ref_shape, dtype=bool)
    for lesion, gray in zip(lesions, gray_masks):
        active |= lesion
        gray_union |= gray
    active &= gray_union
    return remove_small_blobs(active, min_pixels=min_pixels).astype(bool)


def run_treatment_and_make_after_gif(
    active_global: np.ndarray,
    u8_slices: List[np.ndarray],
    gray_masks: List[np.ndarray],
    out_path: str,
    px_radius: int,
    fps: int,
    frame_stride: int,
    max_frames: int,
    max_steps: int = 160000,
) -> Dict[str, int]:
    active = active_global.copy()
    initial = int(active.sum())
    treated = 0
    steps = 0

    breath_seq = breathing_order(len(u8_slices))
    capture_step = 0
    frames: List[Image.Image] = []

    def capture(show_dot: bool = False, pos: Tuple[int, int] = (0, 0), color: Tuple[int, int, int] = (255, 0, 0)):
        nonlocal capture_step
        bg_idx = breath_seq[capture_step % len(breath_seq)]
        bg = u8_slices[bg_idx]
        rgb = np.repeat(bg[..., None], 3, axis=2)
        visible = active & gray_masks[bg_idx]
        rgb[visible, 0] = 255
        rgb[visible, 1] = 255
        rgb[visible, 2] = 0
        img = Image.fromarray(rgb)
        if show_dot:
            py, px = pos
            draw = ImageDraw.Draw(img)
            r = px_radius
            draw.ellipse((px - r, py - r, px + r, py + r), outline=color, width=2)
        frames.append(img)
        capture_step += 1

    capture(show_dot=False)

    if initial > 0:
        coords = np.argwhere(active)
        y, x = np.mean(coords, axis=0).astype(int)
    else:
        y, x = active.shape[0] // 2, active.shape[1] // 2

    while active.any() and steps < max_steps and len(frames) < max_frames:
        target = nearest_active(active, (y, x))
        if target is None:
            break
        ty, tx = target

        for py, px in bresenham_line(y, x, ty, tx):
            steps += 1
            hit = active & disk_mask(active.shape, py, px, px_radius)
            laser_on = bool(np.any(hit))

            if laser_on:
                n = int(hit.sum())
                active[hit] = False
                treated += n

            if (steps % frame_stride == 0) and (len(frames) < max_frames):
                capture(
                    show_dot=True,
                    pos=(py, px),
                    color=((255, 0, 0) if laser_on else (0, 0, 255)),
                )

            if not active.any() or steps >= max_steps or len(frames) >= max_frames:
                break

        y, x = ty, tx

    if len(frames) < max_frames:
        capture(show_dot=False)

    save_gif(frames, out_path, fps=fps)
    remaining = int(active.sum())
    return {
        "initial_pixels": initial,
        "treated_pixels": treated,
        "remaining_pixels": remaining,
        "laser_steps": steps,
    }


def read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def main() -> None:
    st.set_page_config(page_title="Cancer DICOM GIF Generator", layout="wide")
    st.title("Cancer DICOM Before/After GIF Generator")
    st.caption("Uses cached model only: `model_cache/after_ae_model.keras`.")

    with st.sidebar:
        st.subheader("Model")
        model_path = st.text_input("Model path", value=DEFAULT_MODEL_PATH)

        st.subheader("Output")
        output_root = st.text_input("Output root", value=DEFAULT_OUTPUT_ROOT)
        fps = st.number_input("GIF FPS", min_value=1, max_value=30, value=12)
        frame_stride = st.number_input("Treatment frame stride", min_value=1, max_value=30, value=6)
        max_frames = st.number_input("Max GIF frames", min_value=30, max_value=2000, value=320)
        max_slices = st.number_input("Max slices to process", min_value=20, max_value=1200, value=140)

        run_btn = st.button("Generate Before/After GIFs", type="primary", use_container_width=True)

    uploads = st.file_uploader(
        "Upload one cancer case (DICOM files and/or one ZIP)",
        accept_multiple_files=True,
    )

    if not run_btn:
        st.info("Upload one cancer case and click Generate Before/After GIFs.")
        return

    if not uploads:
        st.error("Please upload cancer DICOM files first.")
        return

    model_abs = os.path.abspath(model_path)
    if not os.path.exists(model_abs):
        st.error(f"Model file not found: `{model_abs}`")
        return

    seed_everything(42)
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_root, run_id)
    upload_dir = os.path.join(run_dir, "uploaded_cancer")
    os.makedirs(run_dir, exist_ok=True)
    t0 = time.time()

    before_gif_path = os.path.join(run_dir, "before_treatment.gif")
    after_gif_path = os.path.join(run_dir, "after_treatment.gif")

    with st.status("Running model and rendering GIFs...", expanded=True) as status:
        try:
            st.write("Saving upload...")
            cancer_dir = save_uploaded_files(uploads, upload_dir)
            st.write(f"Cancer folder: `{cancer_dir}`")

            st.write("Preparing slices...")
            case = prepare_cancer_case(cancer_dir, max_slices=int(max_slices))
            dicom_files = case["dicom_files"]

            st.write("Loading model...")
            model = load_model_cached(model_abs)

            st.write("Detecting lesion regions...")
            lesions = detect_lesions_with_model(
                model=model,
                norm_slices=case["norm_slices"],
                valid_masks=case["valid_masks"],
            )

            st.write("Rendering BEFORE GIF...")
            make_before_gif(
                u8_slices=case["u8_slices"],
                lesions=lesions,
                out_path=before_gif_path,
                fps=int(fps),
                max_frames=int(max_frames),
            )

            st.write("Running treatment simulation and rendering AFTER GIF...")
            active_map = build_active_map(lesions, case["gray_masks"], min_pixels=18)
            y_mm, x_mm = get_pixel_spacing_mm(dicom_files[0])
            px_radius = max(1, int(round(2.5 / ((y_mm + x_mm) / 2.0))))
            treatment = run_treatment_and_make_after_gif(
                active_global=active_map,
                u8_slices=case["u8_slices"],
                gray_masks=case["gray_masks"],
                out_path=after_gif_path,
                px_radius=px_radius,
                fps=int(fps),
                frame_stride=int(frame_stride),
                max_frames=int(max_frames),
            )

            payload = {
                "run_id": run_id,
                "timestamp_utc": datetime.utcnow().isoformat() + "Z",
                "model_path": model_abs,
                "cancer_dir": cancer_dir,
                "slices_processed": len(dicom_files),
                "before_gif_path": before_gif_path,
                "after_gif_path": after_gif_path,
                "treatment_summary": treatment,
                "runtime_seconds": round(time.time() - t0, 2),
            }
            payload_path = os.path.join(run_dir, "model_output.json")
            with open(payload_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

            status.update(label="Completed", state="complete")
        except Exception as e:
            status.update(label="Failed", state="error")
            st.exception(e)
            return

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Before GIF")
        st.image(before_gif_path)
        st.download_button(
            "Download before_treatment.gif",
            data=read_bytes(before_gif_path),
            file_name="before_treatment.gif",
            mime="image/gif",
        )

    with col2:
        st.subheader("After GIF")
        st.image(after_gif_path)
        st.download_button(
            "Download after_treatment.gif",
            data=read_bytes(after_gif_path),
            file_name="after_treatment.gif",
            mime="image/gif",
        )

    st.success(f"Done. GIFs saved under `{run_dir}`")


if __name__ == "__main__":
    main()
