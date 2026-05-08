import argparse
import time
import yaml
import pytorch_lightning as pl
from tqdm import tqdm
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.regression import MeanAbsoluteError
from utils import *


def parse_args():
    p = argparse.ArgumentParser(description="Test Script Based on Pytorch Lightning")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config")
    p.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint (.ckpt)")
    p.add_argument("--cuda", type=int, default=0, help="CUDA device index, e.g. --cuda 0")
    return p.parse_args()


def _build_metrics_summary(count, psnr, ssim, mae, result_dir):
    metrics = {
        "count": count,
        "psnr": psnr,
        "ssim": ssim,
        "mae": mae,
    }
    summary = (
        "\n===== Global Test Metrics =====\n"
        f"PSNR: {metrics['psnr']:.4f}\n"
        f"SSIM: {metrics['ssim']:.4f}\n"
        f"MAE: {metrics['mae']:.4f}\n"
        "==============================\n"
        f"[INFO] Saving 2D NIfTI slices to: {result_dir}\n"
        f"[INFO] {metrics['count']} samples saved."
    )
    return {"metrics": metrics, "summary": summary}


def test_sr(model, dataloader, device, result_dir, num_inference_steps=None):
    test_psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    test_ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    test_mae = MeanAbsoluteError().to(device)
    count = 0

    for batch in tqdm(dataloader, desc="Testing: "):
        src = batch["src"].to(device)
        dst = batch["dst"].to(device)
        name = batch["name"]

        dst_pred = model.sample(src, num_inference_steps=num_inference_steps)

        test_psnr.update(dst_pred.float(), dst.float())
        test_ssim.update(dst_pred.float(), dst.float())
        test_mae.update(dst_pred.float(), dst.float())

        for b in range(src.shape[0]):
            base = os.path.join(result_dir, name[b])
            src_to_save = prepare_src_slice_for_save(src[b].detach().cpu())
            save_2d_nifti(src_to_save, base + "_src.nii.gz")
            save_2d_nifti(dst[b].detach().cpu(), base + "_dst.nii.gz")
            save_2d_nifti(dst_pred[b].detach().cpu(), base + "_dst_pred.nii.gz")
            count += 1

    return _build_metrics_summary(
        count=count,
        psnr=test_psnr.compute().item(),
        ssim=test_ssim.compute().item(),
        mae=test_mae.compute().item(),
        result_dir=result_dir,
    )


if __name__ == "__main__":
    args = parse_args()
    assert os.path.exists(args.config), "Config file not exist."
    project_root = Path(__file__).resolve().parent
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    ckpt_path = args.ckpt
    if ckpt_path is None:
        ckpt_dir = Path(config["trainer"]["ckpt_dir"])
        if not ckpt_dir.is_absolute():
            ckpt_dir = (project_root / ckpt_dir).resolve()
        ckpt_path = str((ckpt_dir / "last.ckpt").resolve())
        print("\n" + "#" * 18 + " USING DEFAULT CKPT " + "#" * 18)
        print(f"[DEFAULT CKPT] {ckpt_path}")
        print("#" * 56 + "\n")
    else:
        ckpt_path = str(Path(ckpt_path).resolve())
    assert os.path.exists(ckpt_path), f"Checkpoint file not exist: {ckpt_path}"

    pl.seed_everything(config["seed"], workers=True)

    dm = instantiate_from_config(model_name=config["data"]["target"], **config["data"]["params"])
    dm.setup(stage="test")
    model_target = config["model"]["target"]
    model_class = get_module_from_config(model_name=model_target)
    model = model_class.load_from_checkpoint(ckpt_path, **config["model"]["params"])

    num_inference_steps = config["model"]["params"].get("num_inference_steps", None)

    result_dir = os.path.join(config['test_output_dir'])
    os.makedirs(result_dir, exist_ok=True)

    # Control the device here
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    try:
        model = torch.compile(model).to(device)
    except Exception:
        model = model.to(device)
    model.eval()

    scaler_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=scaler_dtype):
        dataloader = dm.test_dataloader()
        t_start = time.perf_counter()
        metrics = test_sr(model, dataloader, device, result_dir, num_inference_steps=num_inference_steps)
        t_elapsed = time.perf_counter() - t_start
        print(metrics["summary"])
        print(f"[INFO] test time: {to_hms_str(t_elapsed)}")
        result_dir_3d = os.path.join(config["test_output_dir"] + "_3d")
        stack_to_3d(result_dir, result_dir_3d, stack_axis=1)
        print(f"[INFO] Saved 3D NIfTI volumes to: {result_dir_3d}")
