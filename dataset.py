import os
import glob
import hashlib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, get_worker_info
import pytorch_lightning as pl
from typing import List, Tuple, Optional
from scipy.ndimage import gaussian_filter, zoom
from utils import instantiate_from_config


class SRSliceDataset(Dataset):
    def __init__(
            self,
            filepaths: List[str],
            slice_axis: int = 1,
            slice_nums: int = 32,
            cache_mode: str = "memmap",
            transform: Optional[object] = None,
    ):
        super().__init__()
        self.filepaths = sorted(filepaths)
        self.slice_axis = slice_axis
        self.slice_nums = slice_nums
        self.transform = transform
        self.cache_mode = cache_mode

        # Build (file_idx, slice_idx) indices centered on each volume.
        self.index: List[Tuple[int, int]] = []
        for fi, fp in enumerate(self.filepaths):
            z = np.load(fp, mmap_mode="r" if self.cache_mode == "memmap" else None)
            src_shape = z[0].shape
            D = src_shape[self.slice_axis]

            assert self.slice_nums <= D, "slice_nums out of range"
            c = D // 2
            start = c - (self.slice_nums // 2)
            sl_indices = range(start, start + self.slice_nums)
            for si in sl_indices:
                self.index.append((fi, si))

    def __len__(self):
        return len(self.index)

    @staticmethod
    def _to_slice(data, axis, si) -> np.ndarray:
        if axis == 0:
            sl = data[si, :, :]
        elif axis == 1:
            sl = data[:, si, :]
        else:
            sl = data[:, :, si]
        return sl

    def __getitem__(self, idx):
        fi, si = self.index[idx]
        fp = self.filepaths[fi]

        if self.cache_mode == "memmap":
            data = np.load(fp, mmap_mode="r")
        else:
            data = np.load(fp)  # .npy, (3, D, W, H)
        src = data[0]  # src
        dst = data[1]  # dst
        mask = data[2]  # mask

        src_sl = self._to_slice(src, self.slice_axis, si).astype(np.float32, copy=True)
        dst_sl = self._to_slice(dst, self.slice_axis, si).astype(np.float32, copy=True)
        mask_sl = self._to_slice(mask, self.slice_axis, si).astype(np.float32, copy=True)
        src_sl = torch.from_numpy(src_sl).unsqueeze(0)
        dst_sl = torch.from_numpy(dst_sl).unsqueeze(0)
        mask_sl = torch.from_numpy(mask_sl).unsqueeze(0)

        # for .npy
        name = os.path.splitext(os.path.basename(fp))[0]

        if self.transform:
            src_sl, dst_sl, mask_sl = self.transform(src_sl, dst_sl, mask_sl)

        residual_sl = dst_sl - src_sl

        return {'src': src_sl, 'dst': dst_sl, 'mask': mask_sl, 'residual': residual_sl, 'name': f'{name}_s{si}'}


class TwoP5DSRSliceDataset(Dataset):
    def __init__(
            self,
            filepaths: List[str],
            slice_axis: int = 1,
            slice_nums: int = 32,
            depth: int = 9,
            cache_mode: str = "memmap",
            transform: Optional[object] = None,
    ):
        super().__init__()
        self.filepaths = sorted(filepaths)
        self.slice_axis = slice_axis
        self.slice_nums = slice_nums
        self.depth = depth
        assert self.depth % 2 == 1, "depth must be an odd number"
        self.transform = transform
        self.cache_mode = cache_mode

        # Use depth consecutive source slices; dst and mask use only the center slice.
        self.index: List[Tuple[int, Tuple[int, int], Tuple[int, int]]] = []
        for fi, fp in enumerate(self.filepaths):
            z = np.load(fp, mmap_mode="r" if self.cache_mode == "memmap" else None)
            src_shape = z[0].shape
            D = src_shape[self.slice_axis]

            assert self.slice_nums <= D, "slice_nums out of range"
            assert self.slice_nums + (self.depth // 2) <= D, "slice_nums + depth//2 out of range"
            c = D // 2
            start = c - (self.slice_nums // 2)
            end = start + self.slice_nums
            sl_indices = range(start, end)
            half = self.depth // 2
            for si in sl_indices:
                sub_start = si - half
                sub_end = sub_start + self.depth
                self.index.append((fi, (sub_start, sub_end), (si, si + 1)))

    def __len__(self):
        return len(self.index)

    @staticmethod
    def _to_slice(data, axis, slice_range: Tuple[int, int]) -> np.ndarray:
        data = np.moveaxis(data, axis, 0)
        sl = data[slice_range[0]:slice_range[1], :, :]
        return sl

    def __getitem__(self, idx):
        fi, src_slice_range, dst_slice_range = self.index[idx]
        fp = self.filepaths[fi]

        if self.cache_mode == "memmap":
            data = np.load(fp, mmap_mode="r")
        else:
            data = np.load(fp)  # .npy, (3, D, W, H)
        src = data[0]
        dst = data[1]
        mask = data[2]

        src_sl = self._to_slice(src, self.slice_axis, src_slice_range)
        dst_sl = self._to_slice(dst, self.slice_axis, dst_slice_range)
        mask_sl = self._to_slice(mask, self.slice_axis, dst_slice_range)

        src_sl = torch.from_numpy(src_sl.astype(np.float32, copy=True))
        dst_sl = torch.from_numpy(dst_sl.astype(np.float32, copy=True))
        mask_sl = torch.from_numpy(mask_sl.astype(np.float32, copy=True))

        name = os.path.splitext(os.path.basename(fp))[0]

        if self.transform:
            src_sl, dst_sl, mask_sl = self.transform(src_sl, dst_sl, mask_sl)

        center_idx = src_sl.shape[0] // 2
        residual_sl = dst_sl - src_sl[center_idx:center_idx + 1]
        mid = dst_slice_range[0]
        return {'src': src_sl, 'dst': dst_sl, 'mask': mask_sl, 'residual': residual_sl, 'name': f'{name}_s{mid}'}


class DegradeSRSliceDataset(TwoP5DSRSliceDataset):
    def __init__(
            self,
            filepaths: List[str],
            slice_axis: int = 1,
            slice_nums: int = 32,
            depth: int = 9,
            cache_mode: str = "memmap",
            split: str = "train",
            base_seed: int = 24,
            bias_strength_range: Tuple[float, float] = (0.05, 0.15),
            blur_sigma_range: Tuple[float, float] = (0.6, 1.4),
            downsample_scale: int = 4,
            noise_sigma_range: Tuple[float, float] = (0.0, 0.02),
            bias_field_sigma_ratio: float = 0.18,
    ):
        super().__init__(
            filepaths=filepaths,
            slice_axis=slice_axis,
            slice_nums=slice_nums,
            depth=depth,
            cache_mode=cache_mode,
            transform=None,
        )

        self.split = split
        self.base_seed = int(base_seed)

        self.bias_strength_range = bias_strength_range
        self.blur_sigma_range = blur_sigma_range
        self.downsample_scale = int(downsample_scale)
        self.noise_sigma_range = noise_sigma_range
        self.bias_field_sigma_ratio = float(bias_field_sigma_ratio)

        self.fixed_degrade_params = {
            "bias_strength": self._range_mean(self.bias_strength_range),
            "blur_sigma": self._range_mean(self.blur_sigma_range),
            "noise_sigma": self._range_mean(self.noise_sigma_range),
        }

        self._worker_rngs = {}

    @staticmethod
    def _range_mean(vrange: Tuple[float, float]) -> float:
        return float(0.5 * (vrange[0] + vrange[1]))

    def _sample_degrade_params(self, rng: np.random.Generator) -> dict:
        return {
            "bias_strength": float(rng.uniform(*self.bias_strength_range)),
            "blur_sigma": float(rng.uniform(*self.blur_sigma_range)),
            "noise_sigma": float(rng.uniform(*self.noise_sigma_range)),
        }

    def _get_train_rng(self) -> np.random.Generator:
        worker = get_worker_info()

        if worker is None:
            worker_id = 0
            seed = self.base_seed
        else:
            worker_id = worker.id
            seed = worker.seed % (2 ** 32)

        if worker_id not in self._worker_rngs:
            self._worker_rngs[worker_id] = np.random.default_rng(seed)

        return self._worker_rngs[worker_id]

    def _get_rng_and_params(self, idx: int):
        """
        train:
        Each worker has a persistent RNG
        The RNG state is continuously advanced
        Each `__getitem__` generates a new random degradation

        val/test:
        Each `idx` creates a temporary RNG using `base_seed + idx`
        The same sample will always get the same degradation
        """
        if self.split in {"val", "test"}:
            rng = np.random.default_rng((self.base_seed + idx) % (2 ** 32))
            params = self.fixed_degrade_params
        else:
            rng = self._get_train_rng()
            params = self._sample_degrade_params(rng)

        return rng, params

    def _make_bias_field(
            self,
            height: int,
            width: int,
            strength: float,
            rng: np.random.Generator,
    ) -> np.ndarray:
        raw = rng.normal(0.0, 1.0, size=(height, width)).astype(np.float32)

        sigma = max(1.0, min(height, width) * self.bias_field_sigma_ratio)
        smooth = gaussian_filter(raw, sigma=sigma, mode="reflect")

        denom = max(float(np.max(np.abs(smooth))), 1e-6)
        smooth = smooth / denom

        bias_field = 1.0 + strength * smooth
        return bias_field.astype(np.float32)

    def _degrade_from_hr(
            self,
            hr_stack: np.ndarray,
            params: dict,
            rng: np.random.Generator,
    ) -> np.ndarray:
        x = hr_stack.astype(np.float32, copy=True)

        _, h, w = x.shape

        # 1. bias field
        bias_field = self._make_bias_field(height=h, width=w, strength=params["bias_strength"], rng=rng, )
        x = x * bias_field[None, :, :]
        x = np.clip(x, 0.0, 1.0)

        # 2. blur
        x = gaussian_filter(x, sigma=(0.0, params["blur_sigma"], params["blur_sigma"]), mode="reflect", )

        # 3. downsample
        scale = float(self.downsample_scale)
        low = zoom(x, zoom=(1.0, 1.0 / scale, 1.0 / scale), order=3, )

        # 4. noise
        noise_sigma = float(params["noise_sigma"])
        if noise_sigma > 0.0:
            noise = rng.normal(0.0, noise_sigma, size=low.shape).astype(np.float32)
            low = low + noise

        # 5. upsample back to HR size
        up = zoom(low, zoom=(1.0, h / low.shape[1], w / low.shape[2]), order=3, )

        return np.clip(up, 0.0, 1.0).astype(np.float32)

    def __getitem__(self, idx):
        fi, src_slice_range, dst_slice_range = self.index[idx]
        fp = self.filepaths[fi]

        if self.cache_mode == "memmap":
            data = np.load(fp, mmap_mode="r")
        else:
            data = np.load(fp)

        dst = data[1]
        mask = data[2]

        hr_stack = self._to_slice(dst, self.slice_axis, src_slice_range)  # 2.5D _to_slice
        dst_sl = self._to_slice(dst, self.slice_axis, dst_slice_range)
        mask_sl = self._to_slice(mask, self.slice_axis, dst_slice_range)

        rng, params = self._get_rng_and_params(idx)
        src_sl = self._degrade_from_hr(hr_stack=hr_stack, params=params, rng=rng, )

        src_sl = torch.from_numpy(src_sl.astype(np.float32, copy=True))
        dst_sl = torch.from_numpy(dst_sl.astype(np.float32, copy=True))
        mask_sl = torch.from_numpy(mask_sl.astype(np.float32, copy=True))

        center_idx = src_sl.shape[0] // 2
        residual_sl = dst_sl - src_sl[center_idx:center_idx + 1]

        name = os.path.splitext(os.path.basename(fp))[0]
        mid = dst_slice_range[0]

        return {
            "src": src_sl,
            "dst": dst_sl,
            "mask": mask_sl,
            "residual": residual_sl,
            "name": f"{name}_s{mid}",
        }


class SRSliceDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_dir: str = None,
            batch_size: int = 8,
            num_workers: int = 4,
            cache_mode: str = "memmap",
            slice_axis: int = 1,
            slice_nums: int = 32,
            train_transforms: Optional[object] = None,
            test_transforms: Optional[object] = None,
            pin_memory: bool = True,
            persistent_workers: bool = True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_mode = cache_mode
        self.slice_axis = slice_axis
        self.slice_nums = slice_nums

        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        if self.train_transforms:
            self.train_transforms = instantiate_from_config(model_name=train_transforms['target'],
                                                            **train_transforms['params'])
        if self.test_transforms:
            self.test_transforms = instantiate_from_config(model_name=test_transforms['target'],
                                                           **test_transforms['params'])

        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        self.train_dir = os.path.join(self.data_dir, "train")
        self.val_dir = os.path.join(self.data_dir, "val")
        self.test_dir = os.path.join(self.data_dir, "test")
        self.train_set = None
        self.val_set = None
        self.test_set = None

    @staticmethod
    def _collect(dirpath: str) -> List[str]:
        data_list = sorted(glob.glob(os.path.join(dirpath, "*.npy")))
        assert len(data_list) > 0, f"Data_list empty collected in {dirpath}."
        return data_list

    def prepare_data(self):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
        )

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            train_files = self._collect(self.train_dir)
            self.train_set = SRSliceDataset(train_files, slice_axis=self.slice_axis, slice_nums=self.slice_nums,
                                            cache_mode=self.cache_mode, transform=self.train_transforms)
            val_files = self._collect(self.val_dir)
            self.val_set = SRSliceDataset(val_files, slice_axis=self.slice_axis, slice_nums=self.slice_nums,
                                          cache_mode=self.cache_mode, transform=self.test_transforms)

        if stage == "test" or stage is None:
            test_files = self._collect(self.test_dir)
            self.test_set = SRSliceDataset(test_files, slice_axis=self.slice_axis, slice_nums=self.slice_nums,
                                           cache_mode=self.cache_mode, transform=self.test_transforms)


class TwoP5DSRSliceDataModule(SRSliceDataModule):
    def __init__(
            self,
            data_dir: str = None,
            batch_size: int = 8,
            num_workers: int = 4,
            cache_mode: str = "memmap",
            slice_axis: int = 1,
            slice_nums: int = 32,
            depth: int = 9,
            train_transforms: Optional[object] = None,
            test_transforms: Optional[object] = None,
            pin_memory: bool = True,
            persistent_workers: bool = True,
    ):
        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            cache_mode=cache_mode,
            slice_axis=slice_axis,
            slice_nums=slice_nums,
            train_transforms=train_transforms,
            test_transforms=test_transforms,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        self.depth = depth

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            train_files = self._collect(self.train_dir)
            self.train_set = TwoP5DSRSliceDataset(
                train_files,
                slice_axis=self.slice_axis,
                slice_nums=self.slice_nums,
                depth=self.depth,
                cache_mode=self.cache_mode,
                transform=self.train_transforms,
            )
            val_files = self._collect(self.val_dir)
            self.val_set = TwoP5DSRSliceDataset(
                val_files,
                slice_axis=self.slice_axis,
                slice_nums=self.slice_nums,
                depth=self.depth,
                cache_mode=self.cache_mode,
                transform=self.test_transforms,
            )

        if stage == "test" or stage is None:
            test_files = self._collect(self.test_dir)
            self.test_set = TwoP5DSRSliceDataset(
                test_files,
                slice_axis=self.slice_axis,
                slice_nums=self.slice_nums,
                depth=self.depth,
                cache_mode=self.cache_mode,
                transform=self.test_transforms,
            )


class DegradeSRSliceDataModule(TwoP5DSRSliceDataModule):
    def __init__(
            self,
            data_dir: str = None,
            batch_size: int = 8,
            num_workers: int = 4,
            cache_mode: str = "memmap",
            slice_axis: int = 1,
            slice_nums: int = 32,
            depth: int = 9,
            pin_memory: bool = True,
            persistent_workers: bool = True,
            base_seed: int = 3407,
            bias_strength_range: Tuple[float, float] = (0.10, 0.20),
            blur_sigma_range: Tuple[float, float] = (0.6, 1.4),
            downsample_scale: int = 4,
            noise_sigma_range: Tuple[float, float] = (0.0, 0.02),
            bias_field_sigma_ratio: float = 0.18,
    ):
        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            cache_mode=cache_mode,
            slice_axis=slice_axis,
            slice_nums=slice_nums,
            depth=depth,
            train_transforms=None,
            test_transforms=None,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        self.base_seed = int(base_seed)
        self.bias_strength_range = bias_strength_range
        self.blur_sigma_range = blur_sigma_range
        self.downsample_scale = int(downsample_scale)
        self.noise_sigma_range = noise_sigma_range
        self.bias_field_sigma_ratio = float(bias_field_sigma_ratio)

    def _build_dataset(self, files: List[str], split: str) -> DegradeSRSliceDataset:
        return DegradeSRSliceDataset(
            files,
            slice_axis=self.slice_axis,
            slice_nums=self.slice_nums,
            depth=self.depth,
            cache_mode=self.cache_mode,
            split=split,
            base_seed=self.base_seed,
            bias_strength_range=self.bias_strength_range,
            blur_sigma_range=self.blur_sigma_range,
            downsample_scale=self.downsample_scale,
            noise_sigma_range=self.noise_sigma_range,
            bias_field_sigma_ratio=self.bias_field_sigma_ratio,
        )

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            train_files = self._collect(self.train_dir)
            self.train_set = self._build_dataset(train_files, split="train")
            val_files = self._collect(self.val_dir)
            self.val_set = self._build_dataset(val_files, split="val")

        if stage == "test" or stage is None:
            test_files = self._collect(self.test_dir)
            self.test_set = self._build_dataset(test_files, split="test")
