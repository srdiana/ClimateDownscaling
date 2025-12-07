import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict
from functools import lru_cache
import numpy as np
import xarray as xr
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import netCDF4 as nc

warnings.filterwarnings("ignore")

class ERA5DownscalingDataset(Dataset):
    PHYSICAL_BOUNDS = {
        'T2M': (180, 330), 'MSL': (80000, 110000), 'TP': (0, 0.1),
        '10u': (-100, 100), '10v': (-100, 100), 'SP': (80000, 110000),
        'T850': (180, 320), 'T950': (200, 330), 'T1000': (220, 340),
        'U850': (-100, 100), 'V850': (-100, 100),
    }

    def __init__(
        self,
        data_path: str,
        years_to_process: List[int],
        variables_to_load: List[str] = None,
        scale_factor: int = 4,
        normalize: bool = True,
        stats_file_path: Optional[str] = None,
        downscale_method: str = "avg_pool",
        cache_size: int = 1000,
        time_chunk_size: int = 1000,
        preload_metadata: bool = True,
        precompute_low_res: bool = True,
    ):
        if variables_to_load is None:
            variables_to_load = ["T2M", "MSL", "TP", "10u", "10v"]
            
        self.data_path = Path(data_path)
        self.years_to_process = years_to_process
        self.variables_to_load = variables_to_load
        self.scale_factor = scale_factor
        self.normalize = normalize
        self.downscale_method = downscale_method
        self.cache_size = cache_size
        self.time_chunk_size = time_chunk_size
        self.preload_metadata = preload_metadata
        self.precompute_low_res = precompute_low_res
        
        self.high_res_cache = {}
        self.low_res_cache = {} if precompute_low_res else None
        self.cache_order = []  
        
        self.CLIMATOLOGY = self._load_normalization_parameters(stats_file_path)
        
        print("Initializing optimized ERA5 dataset...")
        self._setup_data_storage()
        
        self._precompute_normalization_tensors()

    def _precompute_normalization_tensors(self):
        """Предвычисляем тензоры для нормализации"""
        if not self.normalize:
            self.norm_mean = None
            self.norm_std = None
            return
            
        means = []
        stds = []
        for var in self.variables_to_load:
            if var in self.CLIMATOLOGY:
                means.append(self.CLIMATOLOGY[var]['mean'])
                stds.append(self.CLIMATOLOGY[var]['std'])
            else:
                means.append(0.0)
                stds.append(1.0)
        
        self.norm_mean = torch.tensor(means, dtype=torch.float32).view(-1, 1, 1)
        self.norm_std = torch.tensor(stds, dtype=torch.float32).view(-1, 1, 1)

    def _load_normalization_parameters(self, stats_file_path):
        if stats_file_path and Path(stats_file_path).exists():
            try:
                return self._load_nc_stats(stats_file_path)
            except Exception as e:
                print(f"Climatology loading error: {e}, using defaults")
        
        return self._get_default_climatology()

    def _load_nc_stats(self, stats_file_path):
        """Оптимизирована работа с netCDF"""
        with nc.Dataset(stats_file_path, 'r') as stat_data:
            params = stat_data.variables['params'][:]
            climate_data = stat_data.variables['climate_statistics'][:]
            
            param_names = [
                p.decode('utf-8').strip() if isinstance(p, bytes) else str(p).strip()
                for p in params
            ]
            
            climatology = {
                param_name: {
                    'mean': float(climate_data[0, i]),
                    'std': float(climate_data[1, i])
                }
                for i, param_name in enumerate(param_names)
            }
        
        print(f"Loaded climatology for {len(climatology)} variables")
        return climatology

    def _get_default_climatology(self):
        """Оптимизирована с использованием dict comprehension"""
        return {
            var: {
                'mean': (self.PHYSICAL_BOUNDS[var][0] + self.PHYSICAL_BOUNDS[var][1]) / 2,
                'std': (self.PHYSICAL_BOUNDS[var][1] - self.PHYSICAL_BOUNDS[var][0]) / 6
            } if var in self.PHYSICAL_BOUNDS else {'mean': 0.0, 'std': 1.0}
            for var in self.variables_to_load
        }

    def _setup_data_storage(self):
        """Оптимизирована загрузка данных"""
        if self.data_path.is_dir():
            zarr_files = list(self.data_path.glob("*.zarr"))
            if not zarr_files:
                raise FileNotFoundError(f"No Zarr files found in: {self.data_path}")
            self.data_file_path = zarr_files[0]
        else:
            self.data_file_path = self.data_path
            
        print(f"Loading data from: {self.data_file_path}")
        
        try:
            self.ds = xr.open_zarr(self.data_file_path, consolidated=True)
        except Exception:
            self.ds = xr.open_zarr(self.data_file_path, consolidated=False)
        
        self.ds = self.ds.chunk({
            "time": self.time_chunk_size, 
            "latitude": -1, 
            "longitude": -1
        })
        
        available_vars = set(self.ds.data_vars.keys())
        self.variables_to_load = [v for v in self.variables_to_load if v in available_vars]
        print(f"Using variables: {self.variables_to_load}")

        self.ds_filtered = self.ds.sel(
            time=self.ds.time.dt.year.isin(self.years_to_process)
        )
        
        self.num_timesteps = len(self.ds_filtered.time)

        sample_var = self.variables_to_load[0]
        self.high_res_shape = self.ds_filtered[sample_var].shape[1:]
        self.low_res_shape = tuple(
            dim // self.scale_factor for dim in self.high_res_shape
        )
        
        print(f"High resolution: {self.high_res_shape}, Low resolution: {self.low_res_shape}")
        print(f"Total timesteps: {self.num_timesteps}")
        print(f"Cache size: {self.cache_size}, Precompute low-res: {self.precompute_low_res}")

    def _load_single_timestep(self, time_idx: int) -> torch.Tensor:
        """Оптимизирована загрузка одного временного шага"""
        
        data_arrays = [
            self.ds_filtered[var].isel(time=time_idx)
            for var in self.variables_to_load
        ]
        
        computed_arrays = xr.Dataset({
            var: arr for var, arr in zip(self.variables_to_load, data_arrays)
        }).compute()
        
        frame_data = np.stack([
            computed_arrays[var].values for var in self.variables_to_load
        ], axis=0)
        
        tensor_data = torch.from_numpy(frame_data).float()
        
        if self.normalize and self.norm_mean is not None:
            tensor_data = (tensor_data - self.norm_mean) / self.norm_std
        
        return tensor_data

    def _update_cache(self, cache: dict, key: int, value: torch.Tensor):
        """Оптимизированное управление кешем с LRU"""
        if key in cache:
            self.cache_order.remove(key)
            self.cache_order.append(key)
            return
        
        if len(cache) >= self.cache_size:
           
            oldest = self.cache_order.pop(0)
            del cache[oldest]
        
        cache[key] = value
        self.cache_order.append(key)

    def _get_cached_high_res(self, time_idx: int) -> torch.Tensor:
        """Оптимизированное получение high-res данных"""
        if time_idx in self.high_res_cache:
            # Обновляем порядок использования
            self.cache_order.remove(time_idx)
            self.cache_order.append(time_idx)
            return self.high_res_cache[time_idx]
        
        data = self._load_single_timestep(time_idx)
        
        self._update_cache(self.high_res_cache, time_idx, data)
        
        return data

    def _get_cached_low_res(self, time_idx: int, high_res_data: torch.Tensor) -> torch.Tensor:
        """Оптимизированное получение low-res данных"""
        if not self.precompute_low_res:
            return self._create_low_resolution(high_res_data)
            
        if time_idx in self.low_res_cache:
            return self.low_res_cache[time_idx]
        
        data = self._create_low_resolution(high_res_data)
        self._update_cache(self.low_res_cache, time_idx, data)
        
        return data

    def _create_low_resolution(self, high_res_data: torch.Tensor) -> torch.Tensor:
        """Оптимизированное создание low-res версии"""
        if self.scale_factor == 1:
            return high_res_data
        
        # Добавляем batch dimension только один раз
        data_4d = high_res_data.unsqueeze(0)
        
        if self.downscale_method == "avg_pool":
            result = F.avg_pool2d(data_4d, kernel_size=self.scale_factor)
        elif self.downscale_method == "bilinear":
            result = F.interpolate(
                data_4d, 
                size=self.low_res_shape, 
                mode='bilinear', 
                align_corners=False
            )
        else:
            raise ValueError(f"Unknown downscale method: {self.downscale_method}")
        
        return result.squeeze(0)

    def __len__(self) -> int:
        return self.num_timesteps

    def __getitem__(self, index: int) -> Dict:
        """Оптимизированное получение элемента"""
        high_res_data = self._get_cached_high_res(index)
        low_res_data = self._get_cached_low_res(index, high_res_data)
        
        return {
            'input': low_res_data,
            'target': high_res_data,
            'variables': self.variables_to_load,
            'scale_factor': self.scale_factor,
        }

    def get_dataset_stats(self) -> Dict:
        """Вычисление статистики датасета (ленивое вычисление)"""
        if not hasattr(self, '_cached_stats'):
            sample = self[0]
            self._cached_stats = {
                "input_stats": {
                    "min": sample['input'].min().item(),
                    "max": sample['input'].max().item(),
                    "mean": sample['input'].mean().item(),
                    "std": sample['input'].std().item(),
                },
                "target_stats": {
                    "min": sample['target'].min().item(),
                    "max": sample['target'].max().item(),
                    "mean": sample['target'].mean().item(),
                    "std": sample['target'].std().item(),
                }
            }
        return self._cached_stats

    def get_dataset_info(self) -> Dict:
        """Получение информации о датасете"""
        return {
            "variables": self.variables_to_load,
            "high_res_shape": self.high_res_shape,
            "low_res_shape": self.low_res_shape,
            "scale_factor": self.scale_factor,
            "time_steps": self.num_timesteps,
            "years": self.years_to_process,
            "normalize": self.normalize,
            "cache_info": {
                "high_res_cache_size": len(self.high_res_cache),
                "low_res_cache_size": len(self.low_res_cache) if self.low_res_cache else 0,
                "cache_capacity": self.cache_size,
            }
        }

    def clear_cache(self):
        """Очистка всех кешей"""
        self.high_res_cache.clear()
        if self.low_res_cache:
            self.low_res_cache.clear()
        self.cache_order.clear()
        print("Cache cleared")

    def close(self):
        """Закрытие датасета"""
        if hasattr(self, 'ds'):
            self.ds.close()
        if hasattr(self, 'ds_filtered'):
            self.ds_filtered.close()
        self.clear_cache()

    def __del__(self):
        self.close()


class OptimizedDataLoader:
    @staticmethod
    def create_dataloaders(
        dataset, 
        batch_size: int = 16, 
        train_ratio: float = 0.7, 
        val_ratio: float = 0.15, 
        test_ratio: float = 0.15,
        num_workers: int = 0
    ):
        """Оптимизированное создание DataLoader'ов"""
        from torch.utils.data import random_split, DataLoader
        
        # Проверка соотношений
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        
        total_size = len(dataset)
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size
        
        # Используем generator для воспроизводимости
        generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size],
            generator=generator
        )
        
        common_params = {
            'batch_size': batch_size,
            'pin_memory': False,
            'num_workers': num_workers,
        }
        
        dataloaders = {
            'train': DataLoader(
                train_dataset, 
                shuffle=True,
                persistent_workers=num_workers > 0,
                prefetch_factor=2 if num_workers > 0 else None,
                **common_params
            ),
            'val': DataLoader(
                val_dataset, 
                shuffle=False,
                **common_params
            ),
            'test': DataLoader(
                test_dataset, 
                shuffle=False,
                **common_params
            )
        }
        
        print(f"Dataset split: Train {train_size}, Val {val_size}, Test {test_size}")
        return dataloaders


if __name__ == "__main__":
    dataset = ERA5DownscalingDataset(
        data_path="../../main_phys_downscaling_code/dataset/train_6h",
        years_to_process=[2018],
        variables_to_load=["T2M", "MSL", "TP", "10u", "10v"],
        scale_factor=4,
        normalize=True,
        cache_size=1000,
        time_chunk_size=1000,
        preload_metadata=True,
        precompute_low_res=True,
    )
    
    print("Dataset info:", dataset.get_dataset_info())
    print("Dataset stats:", dataset.get_dataset_stats())