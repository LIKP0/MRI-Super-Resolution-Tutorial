import csv
import os
from pathlib import Path
import numpy as np
import importlib
import nibabel as nib
import torch
from scipy.ndimage import zoom
from typing import Dict, List, Tuple, Union


def get_module_from_config(model_name):
    """Resolve and return a class object from a fully qualified class path."""
    module_name, class_name = model_name.rsplit('.', 1)
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)
    return model_class


def instantiate_from_config(model_name, **kwargs):
    """Instantiate an object from a fully qualified class path and keyword arguments."""
    # print(f'Initialize model {model_name}...')
    model_class = get_module_from_config(model_name)
    return model_class(**kwargs)


def collect_oasis_pairs(root: Union[str, Path]) -> List[Tuple[Path, Path]]:
    """Collect matched OASIS image/segmentation file pairs under the root directory."""
    root = Path(root)
    img_paths = sorted(root.rglob("aligned_norm.nii.gz"))
    pairs = []
    for img_path in img_paths:
        seg_path = img_path.with_name("aligned_seg4.nii.gz")
        if seg_path.exists():
            pairs.append((img_path, seg_path))
    if len(pairs) == 0:
        raise RuntimeError(f"No aligned_norm.nii.gz / aligned_seg4.nii.gz pairs found under {Path}")
    return pairs


def describe_nifti(path: Union[str, Path]) -> Dict[str, object]:
    """Return basic metadata summary for a NIfTI file."""
    nii = nib.load(str(path))
    header = nii.header
    return {
        "size": nii.shape,
        "shape": nii.shape,
        "spacing": tuple(float(v) for v in header.get_zooms()[:3]),
        "dtype": str(header.get_data_dtype()),
        "orientation": "".join(nib.aff2axcodes(nii.affine)),
    }


def center_pad_or_crop_2d_plane(volume: np.ndarray, target: int = 224, axis: int = 1, pad_value=0) -> np.ndarray:
    """Center-pad or center-crop all non-slice axes to the target size."""
    result = volume
    plane_axes = [ax for ax in range(result.ndim) if ax != axis]

    for ax in plane_axes:
        size = result.shape[ax]
        if size < target:
            before = (target - size) // 2
            after = target - size - before
            pad_width = [(0, 0)] * result.ndim
            pad_width[ax] = (before, after)
            result = np.pad(result, pad_width, mode="constant", constant_values=pad_value)
        elif size > target:
            start = (size - target) // 2
            slices = [slice(None)] * result.ndim
            slices[ax] = slice(start, start + target)
            result = result[tuple(slices)]

    return result


def make_lr_from_hr(hr: np.ndarray, scale: int = 4, axis: int = 1) -> np.ndarray:
    """Create a low-resolution-like volume by downsampling then upsampling non-slice axes."""
    zoom_down = [1.0] * hr.ndim
    for ax in range(hr.ndim):
        if ax != axis:
            zoom_down[ax] = 1.0 / scale

    low = zoom(hr, zoom_down, order=3)
    zoom_up = [hr.shape[ax] / low.shape[ax] for ax in range(hr.ndim)]
    lr = zoom(low, zoom_up, order=3)
    return np.clip(lr, 0.0, 1.0).astype(np.float32)


def subject_id_from_path(img_path: Union[str, Path]) -> str:
    """Extract subject ID from path parts, preferring segments that start with 'OASIS'."""
    img_path = Path(img_path)
    for part in reversed(img_path.parts):
        if part.startswith("OASIS"):
            return part
    return img_path.parent.name


def preprocess_oasis_pair(
        img_path: Union[str, Path],
        seg_path: Union[str, Path],
        target: int = 224,
        axis: int = 1,
        scale: int = 4,
) -> np.ndarray:
    """Load, normalize, pad/crop, and package LR/HR/mask tensors for one OASIS pair."""
    img = nib.load(str(img_path)).get_fdata(dtype=np.float32)
    seg = nib.load(str(seg_path)).get_fdata(dtype=np.float32)

    vmin = float(np.min(img))
    vmax = float(np.max(img))
    hr = (img - vmin) / max(vmax - vmin, 1e-8)
    mask = seg.astype(np.float32)

    hr = center_pad_or_crop_2d_plane(hr, target=target, axis=axis, pad_value=0).astype(np.float32)
    mask = center_pad_or_crop_2d_plane(mask, target=target, axis=axis, pad_value=0).astype(np.float32)
    lr = make_lr_from_hr(hr, scale=scale, axis=axis)

    return np.stack([lr, hr, mask], axis=0).astype(np.float32)


def show_img_seg_pairs(sample_pairs: List[Tuple[Union[str, Path], Union[str, Path]]], axis: int = 1) -> None:
    """Visualize image views and segmentation map for sampled image/segmentation pairs."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    base_colors = [
        "#000000",
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    fig, axes = plt.subplots(len(sample_pairs), 4, figsize=(14, 3.5 * len(sample_pairs)))
    if len(sample_pairs) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, (img_path, seg_path) in enumerate(sample_pairs):
        img_path = Path(img_path)
        seg_path = Path(seg_path)
        img = nib.load(str(img_path)).get_fdata(dtype=np.float32)
        seg = nib.load(str(seg_path)).get_fdata(dtype=np.float32).astype(np.int16)
        max_label = int(seg.max())
        if max_label >= len(base_colors):
            extra_colors = plt.cm.tab20(np.linspace(0, 1, max_label + 1 - len(base_colors)))
            seg_cmap = ListedColormap(base_colors + [tuple(color) for color in extra_colors])
        else:
            seg_cmap = ListedColormap(base_colors[:max_label + 1])
        slice_indices = [img.shape[ax] // 2 for ax in range(3)]

        for ax in range(3):
            si = slice_indices[ax]
            img_sl = np.take(img, si, axis=ax)
            img_view = np.rot90(img_sl)
            if ax == 0:
                img_view = np.rot90(img_view, k=-1)
            elif ax == 2:
                img_view = np.rot90(img_view, k=-2)
            axes[row, ax].imshow(img_view, cmap="gray")
            axes[row, ax].set_title(f"{img_path.parent.name}\naxis={ax}, slice={si}")
            axes[row, ax].axis("off")

        seg_axis = 1
        seg_si = slice_indices[seg_axis]
        seg_sl = np.take(seg, seg_si, axis=seg_axis)
        axes[row, 3].imshow(np.rot90(seg_sl), cmap=seg_cmap, interpolation="nearest", vmin=0, vmax=max_label)
        axes[row, 3].set_title(f"Segmentation\naxis={seg_axis}, slice={seg_si}")
        axes[row, 3].axis("off")

    plt.tight_layout()
    plt.show()


def plot_batch(batch: Dict[str, object], max_items: int = 4) -> None:
    """Plot LR, HR, residual, and segmentation channels from a training batch."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    base_colors = [
        "#000000",
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    src = batch["src"].detach().cpu().numpy()
    dst = batch["dst"].detach().cpu().numpy()
    mask = batch["mask"].detach().cpu().numpy()
    residual = batch["residual"].detach().cpu().numpy()
    names = batch["name"]
    max_label = int(mask.max())
    if max_label >= len(base_colors):
        extra_colors = plt.cm.tab20(np.linspace(0, 1, max_label + 1 - len(base_colors)))
        mask_cmap = ListedColormap(base_colors + [tuple(color) for color in extra_colors])
    else:
        mask_cmap = ListedColormap(base_colors[:max_label + 1])

    n = min(max_items, src.shape[0])
    residual_absmax = float(np.max(np.abs(residual)))
    if residual_absmax == 0:
        residual_absmax = 1.0

    fig, axes = plt.subplots(4, n, figsize=(3 * n, 12))
    if n == 1:
        axes = np.expand_dims(axes, axis=1)

    for i in range(n):
        axes[0, i].imshow(np.rot90(src[i, 0]), cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"LR\n{names[i]}")
        axes[0, i].axis("off")

        axes[1, i].imshow(np.rot90(dst[i, 0]), cmap="gray", vmin=0, vmax=1)
        axes[1, i].set_title("HR")
        axes[1, i].axis("off")

        axes[2, i].imshow(
            np.rot90(residual[i, 0]),
            cmap="seismic",
            vmin=-residual_absmax,
            vmax=residual_absmax,
        )
        axes[2, i].set_title("Residual")
        axes[2, i].axis("off")

        axes[3, i].imshow(np.rot90(mask[i, 0]), cmap=mask_cmap, interpolation="nearest", vmin=0, vmax=max_label)
        axes[3, i].set_title("Seg labels")
        axes[3, i].axis("off")

    plt.tight_layout()
    plt.show()


def _latest_csvlogger_version_dir(logger_root: Union[str, Path]) -> Path:
    """Return the latest CSVLogger version directory under the logger root."""
    logger_root = Path(logger_root)
    if not logger_root.exists():
        raise FileNotFoundError(f"Logger root does not exist: {logger_root}")

    version_dirs = [p for p in logger_root.glob("version_*") if p.is_dir()]
    if not version_dirs:
        raise FileNotFoundError(f"No version_* directory found under: {logger_root}")

    def _version_key(p: Path) -> int:
        try:
            return int(p.name.split("_")[-1])
        except ValueError:
            return -1

    return sorted(version_dirs, key=_version_key)[-1]


def plot_csvlogger_curves(
        logger_root: Union[str, Path],
        metrics: Union[None, List[str]] = None,
        version: Union[None, int] = None,
) -> None:
    """
    Plot curves from PyTorch Lightning CSVLogger metrics.csv.

    Args:
        logger_root: Directory like logs/<experiment_name>.
        metrics: metric names to plot. If None, auto-detect numeric metrics.
        version: optional CSVLogger version index. If None, use latest version_*.
    """
    import matplotlib.pyplot as plt

    logger_root = Path(logger_root)
    if version is None:
        version_dir = _latest_csvlogger_version_dir(logger_root)
    else:
        version_dir = logger_root / f"version_{version}"
        if not version_dir.exists():
            raise FileNotFoundError(f"CSVLogger version dir not found: {version_dir}")

    csv_path = version_dir / "metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"metrics.csv not found: {csv_path}")

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if len(rows) == 0:
        raise RuntimeError(f"metrics.csv is empty: {csv_path}")

    all_cols = list(rows[0].keys())

    numeric_cols = []
    for k in all_cols:
        if k in {"epoch", "step"}:
            continue
        for r in rows:
            v = r.get(k, "")
            if v is None or v == "":
                continue
            try:
                float(v)
                numeric_cols.append(k)
            except ValueError:
                pass
            break

    numeric_cols = sorted(set(numeric_cols))

    # Build a map from base metric name -> available concrete columns.
    # Example: train/loss -> [train/loss_step, train/loss_epoch]
    metric_variants: Dict[str, List[str]] = {}
    for col in numeric_cols:
        if col.endswith("_step"):
            base = col[:-5]
        elif col.endswith("_epoch"):
            base = col[:-6]
        else:
            base = col
        metric_variants.setdefault(base, []).append(col)

    def _resolve_metric_columns(name: str) -> List[str]:
        # Accept explicit concrete columns directly.
        if name in numeric_cols:
            return [name]

        # Accept base names and map to available variants.
        cols = metric_variants.get(name, [])
        if not cols:
            return []

        # Prefer epoch aggregate over step curve when both exist.
        epoch_col = f"{name}_epoch"
        step_col = f"{name}_step"
        if epoch_col in cols:
            return [epoch_col]
        if step_col in cols:
            return [step_col]
        return [cols[0]]

    if metrics is None:
        # Auto mode: use one representative concrete column per base metric.
        resolved_metrics = []
        for base_name in sorted(metric_variants.keys()):
            resolved_metrics.extend(_resolve_metric_columns(base_name))
        metrics = resolved_metrics
    else:
        missing = []
        resolved_metrics = []
        for m in metrics:
            cols = _resolve_metric_columns(m)
            if not cols:
                missing.append(m)
                continue
            resolved_metrics.extend(cols)
        if missing:
            raise ValueError(
                f"Metrics not found in csv: {missing}. "
                f"Available base metrics: {sorted(metric_variants.keys())}"
            )
        metrics = resolved_metrics

    if len(metrics) == 0:
        raise RuntimeError("No plottable metrics found.")

    n = len(metrics)
    fig, axes = plt.subplots(n, 1, figsize=(8, max(3, 2.5 * n)), squeeze=False)
    axes = axes.flatten()

    for i, m in enumerate(metrics):
        xs = []
        ys = []
        for r in rows:
            y = r.get(m, "")
            if y is None or y == "":
                continue
            x = r.get("step", "")
            if x is None or x == "":
                x = r.get("epoch", "")
            if x is None or x == "":
                continue
            try:
                xs.append(float(x))
                ys.append(float(y))
            except ValueError:
                continue

        axes[i].plot(xs, ys, linewidth=1.5)
        axes[i].set_title(m)
        axes[i].set_xlabel("step")
        axes[i].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def get_affine() -> np.ndarray:
    """Return a default affine used for test-time NIfTI export."""
    return np.array(
        [
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def save_2d_nifti(x: Union[np.ndarray, torch.Tensor], out_path: Union[str, Path],
                  affine: Union[None, np.ndarray] = None) -> None:
    """
    Save one 2D slice to .nii.gz.

    Accepts [H, W], [1, H, W], or [H, W, 1]. Extra singleton dims are squeezed.
    """
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)

    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"save_2d_nifti expects 2D data after squeeze, got shape={arr.shape}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    use_affine = np.eye(4, dtype=np.float32) if affine is None else affine
    nib.save(nib.Nifti1Image(arr.astype(np.float32), affine=use_affine), str(out_path))


def prepare_src_slice_for_save(src_tensor: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Return center slice for 2.5D src input; keep 2D shape for saving."""
    if hasattr(src_tensor, "ndim") and src_tensor.ndim == 3 and src_tensor.shape[0] > 1:
        center_idx = src_tensor.shape[0] // 2
        return src_tensor[center_idx]
    return src_tensor


def stack_to_3d(input_dir: Union[str, Path], output_dir: Union[str, Path], stack_axis: int) -> None:
    """Group indexed 2D NIfTI slices and stack them into 3D NIfTI volumes."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = {}
    for path in sorted(input_dir.glob("*.nii.gz")):
        stem = path.name[:-7] if path.name.endswith(".nii.gz") else None
        if not stem or "_s" not in stem:
            continue
        case_name, rest = stem.split("_s", 1)
        parts = rest.split("_", 1)
        if len(parts) != 2:
            continue
        idx_str, kind = parts
        if not idx_str.isdigit():
            continue
        key = (case_name, kind)
        groups.setdefault(key, []).append((int(idx_str), path))

    for (case_name, kind), slice_items in sorted(groups.items()):
        try:
            slices = []
            for _, path in sorted(slice_items, key=lambda x: x[0]):
                nii = nib.load(str(path))
                arr2d = np.squeeze(np.asarray(nii.dataobj))
                slices.append(arr2d)

            volume3d = np.stack(slices, axis=stack_axis)
            out_path = output_dir / f"{case_name}_{kind}_3d.nii.gz"
            nib.save(nib.Nifti1Image(volume3d, affine=get_affine()), str(out_path))
        except Exception as e:
            print(f"[3D Skip] {case_name}_{kind}: {e}")


def _center_three_views(volume: np.ndarray) -> List[np.ndarray]:
    """Return center sagittal, coronal, and axial views from a 3D volume."""
    x, y, z = volume.shape
    cx, cy, cz = x // 2, y // 2, z // 2
    sag = volume[cx, :, :]
    axi = np.rot90(volume[:, cy, :])
    cor = np.rot90(volume[:, :, cz])
    return [ cor, sag, axi]


def show_gt_pred_three_views(gt: np.ndarray, pred: np.ndarray, title: str) -> None:
    """Show center views for ground truth and prediction volumes."""
    import matplotlib.pyplot as plt

    gt_views = _center_three_views(gt)
    pred_views = _center_three_views(pred)
    names = ["Coronal", "Sagittal", "Axial"]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle(title)

    for j in range(3):
        axes[0, j].imshow(gt_views[j], cmap="gray")
        axes[0, j].set_title(f"GT | {names[j]}")
        axes[0, j].axis("off")

        axes[1, j].imshow(pred_views[j], cmap="gray")
        axes[1, j].set_title(f"Pred | {names[j]}")
        axes[1, j].axis("off")

    plt.tight_layout()
    plt.show()


def show_src_gt_pred_three_views(src: np.ndarray, gt: np.ndarray, pred: np.ndarray, title: str) -> None:
    """Show center views for source, ground truth, and prediction volumes."""
    import matplotlib.pyplot as plt

    volume_rows = [
        ("LR", _center_three_views(src)),
        ("Pred", _center_three_views(pred)),
        ("GT", _center_three_views(gt)),
    ]
    names = ["Coronal", "Sagittal", "Axial"]

    fig, axes = plt.subplots(3, 3, figsize=(12, 11))
    fig.suptitle(title)

    for i, (row_name, views) in enumerate(volume_rows):
        for j, view in enumerate(views):
            axes[i, j].imshow(view, cmap="gray")
            axes[i, j].set_title(f"{row_name} | {names[j]}")
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.show()


def visualize_gt_pred_from_3d_dir(result_3d_dir: Union[str, Path], model_name: str, max_cases: int = 2) -> None:
    """Load and visualize matched source, GT, and prediction 3D volumes."""
    result_3d_dir = Path(result_3d_dir)
    if not result_3d_dir.exists():
        print(f"[SKIP] {model_name}: folder not found -> {result_3d_dir}")
        return

    pred_files = sorted(result_3d_dir.glob("*_dst_pred_3d.nii.gz"))
    if len(pred_files) == 0:
        print(f"[SKIP] {model_name}: no *_dst_pred_3d.nii.gz found in {result_3d_dir}")
        return

    gt_map = {
        p.name.replace("_dst_3d.nii.gz", ""): p
        for p in result_3d_dir.glob("*_dst_3d.nii.gz")
    }
    src_map = {
        p.name.replace("_src_3d.nii.gz", ""): p
        for p in result_3d_dir.glob("*_src_3d.nii.gz")
    }

    print(f"\n[{model_name}] {result_3d_dir}")
    shown = 0
    for pred_path in pred_files:
        case_key = pred_path.name.replace("_dst_pred_3d.nii.gz", "")
        gt_path = gt_map.get(case_key)
        src_path = src_map.get(case_key)
        if gt_path is None:
            print(f"[SKIP] {model_name}: missing GT for case {case_key}")
            continue
        if src_path is None:
            print(f"[SKIP] {model_name}: missing LR/source for case {case_key}")
            continue

        src_vol = np.asarray(nib.load(str(src_path)).dataobj, dtype=np.float32)
        gt_vol = np.asarray(nib.load(str(gt_path)).dataobj, dtype=np.float32)
        pred_vol = np.asarray(nib.load(str(pred_path)).dataobj, dtype=np.float32)
        show_src_gt_pred_three_views(
            src_vol,
            gt_vol,
            pred_vol,
            title=(
                f"{model_name} | {case_key} | "
                f"LR {src_vol.shape} vs GT {gt_vol.shape} vs Pred {pred_vol.shape}"
            ),
        )
        shown += 1
        if shown >= max_cases:
            break

    if shown == 0:
        print(f"[SKIP] {model_name}: no matched LR-GT-Pred case found in {result_3d_dir}")


def to_hms_str(elapsed_seconds):
    """Format elapsed seconds as 'HHhMMmSSs'."""
    total = int(round(elapsed_seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}h{m:02d}m{s:02d}s"
