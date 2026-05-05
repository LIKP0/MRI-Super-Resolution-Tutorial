import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from typing import List, Tuple, Optional
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
