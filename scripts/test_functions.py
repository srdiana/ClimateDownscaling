from pathlib import Path
import sys
import os
from tqdm import tqdm
import torch
import cartopy.feature as cfeature
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from metrics.metric_functions import (
    rmse, mae, r_squared, bias, psnr,
    power_spectral_density, spectral_rmse,
    correlation_coefficient, spectral_mae,
    spectral_correlation, spectral_bias,
    energy_spectra_metrics, calculate_all_metrics,
    calculate_spatial_metrics_map, save_metrics_to_json
)

import xarray as xr
import pandas as pd
import json
import cartopy.crs as ccrs

sys.path.append(str(Path.cwd().parent))

test_results_dir = Path('test_results')
test_results_dir.mkdir(exist_ok=True, parents=True)

class SpatialDownscalingTester:
    """Класс для пространственного тестирования модели downscaling с полными метриками"""
    
    def __init__(self, model, device, use_topography=False, topography_data=None, 
                 variable_names=None, fs=1.0):
        """
        Args:
            model: модель для тестирования
            device: устройство (cpu/cuda)
            use_topography: использовать ли топографию
            topography_data: данные топографии
            variable_names: список имен переменных/каналов
            fs: частота дискретизации для спектральных метрик
        """
        self.model = model
        self.device = device
        self.use_topography = use_topography
        self.topography_data = topography_data
        self.variable_names = variable_names
        self.fs = fs  
        
        self.results_dir = test_results_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"Инициализирован тестер. Результаты будут сохранены в: {self.results_dir}")
    
    def prepare_batch(self, batch):
        """Подготовка батча для тестирования"""
        inputs = batch['input'].to(self.device)
        targets = batch['target'].to(self.device)
        
        if self.use_topography and self.topography_data is not None:
            batch_size = inputs.shape[0]
            topography = self.topography_data.unsqueeze(0).expand(batch_size, -1, -1, -1)
            
            if inputs.shape[2:] != topography.shape[2:]:
                topography = torch.nn.functional.interpolate(
                    topography,
                    size=inputs.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
            return inputs, targets, topography
        else:
            return inputs, targets, None
    
    def test_on_dataset(self, dataloader, save_zarr=True):
        """Тестирование модели на всем датасете"""
        print(f"Начало тестирования на {len(dataloader.dataset)} сэмплах...")
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_inputs = []
        
        spatial_metrics = {}
        spectral_data = {'targets': [], 'predictions': []}
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Тестирование")):
                if self.use_topography:
                    inputs, targets, topography = self.prepare_batch(batch)
                    outputs = self.model(inputs, topography)
                else:
                    inputs, targets, _ = self.prepare_batch(batch)
                    outputs = self.model(inputs)
                
                inputs_np = inputs.cpu().numpy()
                targets_np = targets.cpu().numpy()
                outputs_np = outputs.cpu().numpy()
                
                all_inputs.append(inputs_np)
                all_targets.append(targets_np)
                all_predictions.append(outputs_np)

                if batch_idx == 0 and len(targets_np) > 0:
                    spectral_data['targets'].append(targets_np[0])
                    spectral_data['predictions'].append(outputs_np[0])
                
                self._accumulate_spatial_metrics(batch_idx, outputs_np, targets_np, spatial_metrics)
        
        all_inputs = np.concatenate(all_inputs, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        all_predictions = np.concatenate(all_predictions, axis=0)
        
        print(f"\nФормы данных:")
        print(f"  Inputs: {all_inputs.shape}")
        print(f"  Targets: {all_targets.shape}")
        print(f"  Predictions: {all_predictions.shape}")
        
        results = self._compute_comprehensive_metrics(all_predictions, all_targets, spatial_metrics, spectral_data)
        
        if save_zarr:
            self._save_spatial_metrics_zarr(results, all_predictions, all_targets)
        self.visualize_all_metrics(results, all_predictions, all_targets)
        self.create_comprehensive_report(results)
        
        return results
    
    def _accumulate_spatial_metrics(self, batch_idx, predictions, targets, spatial_metrics):
        """Накопление пространственных метрик по пикселям"""
        batch_size, n_channels, height, width = predictions.shape
        
        if batch_idx == 0:
            for channel in range(n_channels):
                spatial_metrics[f'channel_{channel}'] = {
                    'mse_sum': np.zeros((height, width)),
                    'mae_sum': np.zeros((height, width)),
                    'bias_sum': np.zeros((height, width)),
                    'corr_sum': np.zeros((height, width)),
                    'count': np.zeros((height, width))
                }
        
        for channel in range(n_channels):
            for sample in range(batch_size):
                mse_map = (predictions[sample, channel] - targets[sample, channel]) ** 2
                mae_map = np.abs(predictions[sample, channel] - targets[sample, channel])
                bias_map = predictions[sample, channel] - targets[sample, channel]
                
                if batch_idx == 0:
                    spatial_metrics[f'channel_{channel}']['corr_sum'] = np.zeros((height, width))
                
                spatial_metrics[f'channel_{channel}']['mse_sum'] += mse_map
                spatial_metrics[f'channel_{channel}']['mae_sum'] += mae_map
                spatial_metrics[f'channel_{channel}']['bias_sum'] += bias_map
                spatial_metrics[f'channel_{channel}']['count'] += 1
    
    def _compute_comprehensive_metrics(self, predictions, targets, spatial_metrics, spectral_data):
        """Вычисление всех метрик: пространственных, спектральных и энергетических"""
        print("\nВычисление всесторонних метрик...")
        
        n_channels = predictions.shape[1]
        
        # 1. Глобальные метрики (по всем пикселям и сэмплам)
        predictions_flat = predictions.flatten()
        targets_flat = targets.flatten()
        
        global_metrics = calculate_all_metrics(targets_flat, predictions_flat, fs=self.fs)
        print("Глобальные метрики вычислены")
        
        channel_metrics = {}
        
        for channel in range(n_channels):
            channel_key = f'channel_{channel}'
            if channel_key in spatial_metrics:
                channel_data = spatial_metrics[channel_key]
                count = channel_data['count']
                
                mse_mean = channel_data['mse_sum'] / count
                mae_mean = channel_data['mae_sum'] / count
                bias_mean = channel_data['bias_sum'] / count
                
                rmse_mean = np.sqrt(mse_mean)
                
                channel_metrics[channel_key] = {
                    'mse_map': mse_mean,
                    'rmse_map': rmse_mean,
                    'mae_map': mae_mean,
                    'bias_map': bias_mean,
                    'count': count
                }
                
                channel_name = self.variable_names[channel] if self.variable_names and channel < len(self.variable_names) else f'Channel {channel}'
                print(f"\n{channel_name}:")
                print(f"  MSE: {np.mean(mse_mean):.4f}")
                print(f"  RMSE: {np.mean(rmse_mean):.4f}")
                print(f"  MAE: {np.mean(mae_mean):.4f}")
                print(f"  Bias: {np.mean(bias_mean):.4f}")
        
        spectral_metrics = {}
        
        if spectral_data['targets'] and spectral_data['predictions']:
            sample_target = np.concatenate(spectral_data['targets'], axis=0)
            sample_pred = np.concatenate(spectral_data['predictions'], axis=0)
            
            for channel in range(min(n_channels, sample_target.shape[0])):
                channel_key = f'channel_{channel}'
                
                try:
                    target_channel = sample_target[channel] if sample_target.ndim > 1 else sample_target
                    pred_channel = sample_pred[channel] if sample_pred.ndim > 1 else sample_pred
                    
                    _, psd_target = power_spectral_density(target_channel, fs=self.fs)
                    _, psd_pred = power_spectral_density(pred_channel, fs=self.fs)
                    
                    min_len = min(len(psd_target), len(psd_pred))
                    
                    spectral_metrics[channel_key] = {
                        'psd_target': psd_target[:min_len],
                        'psd_pred': psd_pred[:min_len],
                        'spectral_rmse': spectral_rmse(target_channel, pred_channel, self.fs),
                        'spectral_mae': spectral_mae(target_channel, pred_channel, self.fs),
                        'spectral_correlation': spectral_correlation(target_channel, pred_channel, self.fs),
                        'spectral_bias': spectral_bias(target_channel, pred_channel, self.fs)
                    }
                    
                except Exception as e:
                    print(f"Ошибка при вычислении спектральных метрик для канала {channel}: {e}")
        
        metrics_maps = calculate_spatial_metrics_map(targets, predictions)
        
        return {
            'global_metrics': global_metrics,
            'channel_metrics': channel_metrics,
            'spectral_metrics': spectral_metrics,
            'metrics_maps': metrics_maps,
            'predictions_shape': predictions.shape,
            'targets_shape': targets.shape
        }
    
    def _save_spatial_metrics_zarr(self, results, predictions, targets):
        """Сохранение пространственных метрик в Zarr файл"""
        print("\nСохранение пространственных метрик в Zarr...")
        
        height, width = predictions.shape[2], predictions.shape[3]
        lats = np.arange(height)
        lons = np.arange(width)
        
        n_channels = predictions.shape[1]
        if self.variable_names is not None and len(self.variable_names) >= n_channels:
            variables = self.variable_names[:n_channels]
        else:
            variables = [f'channel_{i}' for i in range(n_channels)]
        
        print(f"Сохраняем метрики для переменных: {variables}")
        
        data_vars = {}
        for metric_name, metric_map in results.get('metrics_maps', {}).items():
            data_vars[metric_name] = (['lat', 'lon'], metric_map)
        
        for channel_idx in range(n_channels):
            channel_key = f'channel_{channel_idx}'
            if channel_key in results.get('channel_metrics', {}):
                channel_metrics = results['channel_metrics'][channel_key]
                
                for metric_name in ['mse_map', 'rmse_map', 'mae_map', 'bias_map']:
                    if metric_name in channel_metrics:
                        var_name = f"{variables[channel_idx]}_{metric_name.replace('_map', '')}"
                        data_vars[var_name] = (['lat', 'lon'], channel_metrics[metric_name])
        
        if 'channel_0' in results.get('channel_metrics', {}):
            data_vars['count'] = (['lat', 'lon'], results['channel_metrics']['channel_0']['count'])
        
        ds = xr.Dataset(
            data_vars,
            coords={
                'lat': lats,
                'lon': lons
            }
        )
        
        ds.attrs['creation_date'] = datetime.now().isoformat()
        ds.attrs['model'] = self.model.__class__.__name__
        ds.attrs['variables'] = str(variables)
        
        if 'global_metrics' in results:
            for metric_name, metric_value in results['global_metrics'].items():
                if isinstance(metric_value, (int, float, np.number)):
                    ds.attrs[f'global_{metric_name}'] = float(metric_value)
        
        zarr_path = self.results_dir / 'spatial_metrics.zarr'
        ds.to_zarr(zarr_path, mode='w')
        
        print(f"Пространственные метрики сохранены в: {zarr_path}")
        
        np.save(self.results_dir / 'predictions.npy', predictions)
        np.save(self.results_dir / 'targets.npy', targets)
        
        all_metrics = {
            'global_metrics': results.get('global_metrics', {}),
            'channel_statistics': self._extract_channel_statistics(results),
            'spectral_metrics': self._extract_spectral_statistics(results),
            'predictions_shape': predictions.shape,
            'targets_shape': targets.shape
        }
        
        save_metrics_to_json(all_metrics, self.results_dir / 'all_metrics.json')
        
        return zarr_path
    
    def _extract_channel_statistics(self, results):
        """Извлечение статистики по каналам"""
        stats = {}
        channel_metrics = results.get('channel_metrics', {})
        
        for channel_key, metrics in channel_metrics.items():
            channel_stats = {}
            for metric_name, metric_map in metrics.items():
                if 'map' in metric_name and isinstance(metric_map, np.ndarray):
                    channel_stats[f'{metric_name}_mean'] = float(np.nanmean(metric_map))
                    channel_stats[f'{metric_name}_std'] = float(np.nanstd(metric_map))
                    channel_stats[f'{metric_name}_min'] = float(np.nanmin(metric_map))
                    channel_stats[f'{metric_name}_max'] = float(np.nanmax(metric_map))
            stats[channel_key] = channel_stats
        
        return stats
    
    def _extract_spectral_statistics(self, results):
        """Извлечение статистики по спектральным метрикам"""
        stats = {}
        spectral_metrics = results.get('spectral_metrics', {})
        
        for channel_key, metrics in spectral_metrics.items():
            channel_stats = {}
            for metric_name, metric_value in metrics.items():
                if metric_name not in ['psd_target', 'psd_pred'] and isinstance(metric_value, (int, float, np.number)):
                    channel_stats[metric_name] = float(metric_value)
            stats[channel_key] = channel_stats
        
        return stats
    
    def visualize_all_metrics(self, results, predictions, targets):
        """Визуализация всех метрик: пространственных, спектральных и энергетических"""
        print("\nВизуализация всех метрик...")
        
        self.visualize_spatial_metrics()
        self.visualize_spectral_metrics(results)
        # self.visualize_distributions(predictions, targets)
        self.visualize_energy_spectra(results)
        self.visualize_summary_metrics(results)
    
    def visualize_spatial_metrics(self, zarr_path=None):
        """Визуализация пространственных метрик из Zarr файла"""
        print("\nВизуализация пространственных метрик...")
        
        if zarr_path is None:
            zarr_files = list(self.results_dir.glob('*.zarr'))
            if not zarr_files:
                print("Zarr файл не найден")
                return
            zarr_path = zarr_files[0]
        
        ds = xr.open_zarr(zarr_path)
        
        if 'variables' in ds.attrs:
            import ast
            variables = ast.literal_eval(ds.attrs['variables'])
        else:
            var_names = []
            for var in ds.data_vars:
                if var != 'count' and not var.startswith('channel_'):
                    var_base = var.split('_')[0]
                    if var_base not in var_names:
                        var_names.append(var_base)
            variables = var_names if var_names else ['composite']
        
        metrics = ['mse', 'rmse', 'mae', 'bias', 'relative_error']
        
        for var_idx, var_name in enumerate(variables[:3]):  
            n_metrics = min(5, len([m for m in metrics if f"{var_name}_{m}" in ds]))
            if n_metrics == 0:
                continue
                
            n_cols = min(3, n_metrics)
            n_rows = (n_metrics + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
            if n_metrics == 1:
                axes = np.array([axes])
            axes = axes.flatten()
            
            plot_idx = 0
            for metric in metrics:
                if plot_idx >= len(axes):
                    break
                    
                data_var_name = f"{var_name}_{metric}"
                if data_var_name in ds:
                    ax = axes[plot_idx]
                    data = ds[data_var_name]
                    
                    im = ax.imshow(data, cmap='viridis', aspect='auto')
                    ax.set_title(f'{var_name} - {metric.upper()}')
                    ax.set_xlabel('Longitude (pixels)')
                    ax.set_ylabel('Latitude (pixels)')
                    
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    plot_idx += 1
            
            for idx in range(plot_idx, len(axes)):
                axes[idx].set_visible(False)
            
            plt.suptitle(f'Пространственные метрики для {var_name}', fontsize=16)
            plt.tight_layout()
            
            save_path = self.results_dir / f'{var_name}_spatial_metrics.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
    
    def visualize_spectral_metrics(self, results):
        """Визуализация спектральных метрик"""
        print("\nВизуализация спектральных метрик...")
        
        spectral_metrics = results.get('spectral_metrics', {})
        if not spectral_metrics:
            print("Нет данных для спектральной визуализации")
            return
        
        # Для каждого канала с спектральными метриками
        for channel_key, metrics in spectral_metrics.items():
            if 'psd_target' not in metrics or 'psd_pred' not in metrics:
                continue
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()
            
            # 1. PSD сравнение
            ax = axes[0]
            freqs = np.arange(len(metrics['psd_target'])) * self.fs / len(metrics['psd_target'])
            ax.plot(freqs[:len(metrics['psd_target'])], metrics['psd_target'], 'b-', label='Target PSD', alpha=0.7)
            ax.plot(freqs[:len(metrics['psd_pred'])], metrics['psd_pred'], 'r-', label='Prediction PSD', alpha=0.7)
            ax.set_xlabel('Frequency')
            ax.set_ylabel('Power Spectral Density')
            ax.set_title(f'Power Spectral Density - {channel_key}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 2. Спектральные метрики бар-чарт
            ax = axes[1]
            spectral_metric_names = ['spectral_rmse', 'spectral_mae', 'spectral_correlation', 'spectral_bias']
            spectral_metric_values = [metrics.get(name, 0) for name in spectral_metric_names]
            bars = ax.bar(spectral_metric_names, spectral_metric_values, color=['blue', 'red', 'green', 'orange'])
            ax.set_ylabel('Value')
            ax.set_title(f'Спектральные метрики - {channel_key}')
            ax.set_xticklabels(spectral_metric_names, rotation=45, ha='right')
            
            for bar, value in zip(bars, spectral_metric_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.4f}', ha='center', va='bottom')
            
            # 3. Отношение PSD
            ax = axes[2]
            with np.errstate(divide='ignore', invalid='ignore'):
                psd_ratio = metrics['psd_pred'] / metrics['psd_target']
                psd_ratio = np.where(np.isfinite(psd_ratio), psd_ratio, 0)
            
            ax.plot(freqs[:len(psd_ratio)], psd_ratio, 'g-', alpha=0.7)
            ax.axhline(y=1, color='r', linestyle='--', alpha=0.5)
            ax.set_xlabel('Frequency')
            ax.set_ylabel('PSD Ratio (Pred/Target)')
            ax.set_title(f'Отношение PSD - {channel_key}')
            ax.grid(True, alpha=0.3)
            
            # 4. Кумулятивная спектральная энергия
            ax = axes[3]
            cum_energy_target = np.cumsum(metrics['psd_target'])
            cum_energy_pred = np.cumsum(metrics['psd_pred'])
            
            if len(cum_energy_target) > 0 and len(cum_energy_pred) > 0:
                
                cum_energy_target = cum_energy_target / cum_energy_target[-1] if cum_energy_target[-1] > 0 else cum_energy_target
                cum_energy_pred = cum_energy_pred / cum_energy_pred[-1] if cum_energy_pred[-1] > 0 else cum_energy_pred
                
                ax.plot(freqs[:len(cum_energy_target)], cum_energy_target, 'b-', label='Target', alpha=0.7)
                ax.plot(freqs[:len(cum_energy_pred)], cum_energy_pred, 'r-', label='Prediction', alpha=0.7)
                ax.set_xlabel('Frequency')
                ax.set_ylabel('Cumulative Energy')
                ax.set_title(f'Кумулятивная спектральная энергия - {channel_key}')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.suptitle(f'Спектральный анализ - {channel_key}', fontsize=16)
            plt.tight_layout()
            
            save_path = self.results_dir / f'{channel_key}_spectral_analysis.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
    
    def visualize_distributions(self, predictions, targets):
        """Визуализация распределений значений"""
        print("\nВизуализация распределений...")
        
        n_channels = min(predictions.shape[1], 3)  # Ограничим 3 каналами
        
        for channel in range(n_channels):
            sample_idx = 0
            target_channel = targets[sample_idx, channel].flatten()
            pred_channel = predictions[sample_idx, channel].flatten()
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            ax = axes[0]
            ax.hist(target_channel, bins=50, alpha=0.7, label='Target', density=True, color='blue')
            ax.hist(pred_channel, bins=50, alpha=0.7, label='Prediction', density=True, color='red')
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.set_title(f'Распределение значений - Channel {channel}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 2. Q-Q plot
            ax = axes[1]
            if len(target_channel) > 0 and len(pred_channel) > 0:
                target_sorted = np.sort(target_channel)
                pred_sorted = np.sort(pred_channel)
                
                quantiles = np.linspace(0, 1, min(len(target_sorted), len(pred_sorted)))
                target_quantiles = np.quantile(target_channel, quantiles)
                pred_quantiles = np.quantile(pred_channel, quantiles)
                
                ax.scatter(target_quantiles, pred_quantiles, alpha=0.5)
                ax.plot([target_quantiles.min(), target_quantiles.max()],
                       [target_quantiles.min(), target_quantiles.max()], 'r--', alpha=0.7)
                ax.set_xlabel('Target Quantiles')
                ax.set_ylabel('Prediction Quantiles')
                ax.set_title(f'Q-Q Plot - Channel {channel}')
                ax.grid(True, alpha=0.3)
            
            # 3. Box plot
            ax = axes[2]
            box_data = [target_channel, pred_channel]
            bp = ax.boxplot(box_data, labels=['Target', 'Prediction'], patch_artist=True)
            
            colors = ['lightblue', 'lightcoral']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            ax.set_ylabel('Value')
            ax.set_title(f'Box Plot - Channel {channel}')
            ax.grid(True, alpha=0.3)
            
            plt.suptitle(f'Анализ распределений - Channel {channel}', fontsize=16)
            plt.tight_layout()
            
            save_path = self.results_dir / f'channel_{channel}_distributions.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
    
    def visualize_energy_spectra(self, results):
        """Визуализация Energy Spectra"""
        print("\nВизуализация Energy Spectra...")
        
        predictions_path = self.results_dir / 'predictions.npy'
        targets_path = self.results_dir / 'targets.npy'
        
        if predictions_path.exists() and targets_path.exists():
            predictions = np.load(predictions_path)
            targets = np.load(targets_path)
            
            sample_idx = 0
            channel_idx = 0
            
            if predictions.shape[1] > channel_idx and targets.shape[1] > channel_idx:
                target_data = targets[sample_idx, channel_idx].flatten()
                pred_data = predictions[sample_idx, channel_idx].flatten()
                
                try:
                    energy_metrics = energy_spectra_metrics(target_data, pred_data, fs=self.fs)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    band_keys = [k for k in energy_metrics.keys() if k.startswith('band_') and k.endswith('_relative_error')]
                    bands = []
                    rel_errors = []
                    
                    for key in sorted(band_keys):
                        band_idx = key.split('_')[1]
                        freq_range = energy_metrics.get(f'band_{band_idx}_freq_range', 'N/A')
                        rel_error = energy_metrics.get(key, np.nan)
                        
                        if not np.isnan(rel_error):
                            bands.append(freq_range)
                            rel_errors.append(rel_error * 100)  # В процентах
                    
                    if bands:
                        bars = ax.bar(bands, rel_errors, color='skyblue', edgecolor='black')
                        ax.set_xlabel('Frequency Band')
                        ax.set_ylabel('Relative Error (%)')
                        ax.set_title(f'Energy Spectra Analysis - Channel {channel_idx}')
                        ax.grid(True, alpha=0.3, axis='y')
                        
                        for bar, error in zip(bars, rel_errors):
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height,
                                   f'{error:.1f}%', ha='center', va='bottom')
                        
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        
                        save_path = self.results_dir / f'energy_spectra_channel_{channel_idx}.png'
                        plt.savefig(save_path, dpi=300, bbox_inches='tight')
                        plt.show()
                    
                except Exception as e:
                    print(f"Ошибка при визуализации Energy Spectra: {e}")
    
    def visualize_summary_metrics(self, results):
        """Сводная визуализация всех метрик"""
        print("\nСоздание сводной визуализации...")
        
        global_metrics = results.get('global_metrics', {})
        channel_stats = self._extract_channel_statistics(results)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        ax = axes[0]
        main_metrics = ['rmse', 'mae', 'r_squared', 'psnr', 'correlation']
        main_values = [global_metrics.get(m, 0) for m in main_metrics]
        
        bars = ax.bar(main_metrics, main_values, color=['blue', 'red', 'green', 'orange', 'purple'])
        ax.set_ylabel('Value')
        ax.set_title('Основные глобальные метрики')
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, value in zip(bars, main_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.4f}', ha='center', va='bottom')
        
        ax = axes[1]
        if channel_stats:
            n_channels = len(channel_stats)
            channel_indices = list(range(n_channels))
            
            channel_rmse = []
            for i in range(n_channels):
                channel_key = f'channel_{i}'
                if channel_key in channel_stats:
                    rmse_mean = channel_stats[channel_key].get('rmse_map_mean', 0)
                    channel_rmse.append(rmse_mean)
            
            if channel_rmse:
                bars = ax.bar(channel_indices[:len(channel_rmse)], channel_rmse, color='skyblue', edgecolor='black')
                ax.set_xlabel('Channel')
                ax.set_ylabel('Average RMSE')
                ax.set_title('Средний RMSE по каналам')
                ax.grid(True, alpha=0.3, axis='y')
                
                for bar, value in zip(bars, channel_rmse):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.4f}', ha='center', va='bottom')
        
        ax = axes[2]
        ax.axis('tight')
        ax.axis('off')
        
        table_data = []
        for metric, value in global_metrics.items():
            if isinstance(value, (int, float)):
                table_data.append([metric.upper(), f'{value:.6f}'])
        
        if table_data:
            table = ax.table(cellText=table_data[:10],
                            colLabels=['Metric', 'Value'],
                            cellLoc='center',
                            loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)
            ax.set_title('Таблица метрик')
        
        ax = axes[3]
        ax.axis('tight')
        ax.axis('off')
        
        summary_text = f"Всего сэмплов: {results.get('predictions_shape', (0,))[0]}\n"
        summary_text += f"Количество каналов: {results.get('predictions_shape', (0,0))[1]}\n"
        summary_text += f"Разрешение: {results.get('predictions_shape', (0,0,0,0))[2:]}"
        
        if 'global_metrics' in results:
            summary_text += f"\n\nЛучшие метрики:\n"
            summary_text += f"PSNR: {global_metrics.get('psnr', 0):.2f} dB\n"
            summary_text += f"R²: {global_metrics.get('r_squared', 0):.4f}\n"
            summary_text += f"Correlation: {global_metrics.get('correlation', 0):.4f}"
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_title('Сводная статистика')
        
        plt.suptitle('Сводная визуализация всех метрик', fontsize=16, y=0.98)
        plt.tight_layout()
        
        save_path = self.results_dir / 'summary_visualization.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_comprehensive_report(self, results):
        """Создание полного отчета"""
        print("\nСоздание полного отчета...")
        
        metrics_file = self.results_dir / 'all_metrics.json'
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                all_metrics = json.load(f)
        else:
            all_metrics = {}
        
        html_content = self._create_html_report(all_metrics)
        html_path = self.results_dir / 'comprehensive_report.html'
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self._create_text_report(all_metrics)
        
        print(f"Полный отчет создан в: {self.results_dir}")
        print(f"HTML отчет: {html_path}")
        print(f"Текстовый отчет: {self.results_dir / 'full_report.txt'}")
    
    def _create_html_report(self, all_metrics):
        """Создание HTML отчета"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Comprehensive Downscaling Model Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; border-bottom: 2px solid #333; padding-bottom: 10px; }}
                h2 {{ color: #555; margin-top: 30px; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
                h3 {{ color: #777; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
                th {{ background-color: #f2f2f2; font-weight: bold; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .metric-value {{ font-weight: bold; color: #0066cc; }}
                .good {{ color: green; }}
                .bad {{ color: red; }}
                .summary {{ background-color: #f0f8ff; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                .images {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }}
                .image-container {{ border: 1px solid #ddd; padding: 10px; border-radius: 5px; }}
                .image-container img {{ max-width: 100%; height: auto; }}
                .image-title {{ text-align: center; font-weight: bold; margin-top: 10px; }}
            </style>
        </head>
        <body>
            <h1>Comprehensive Downscaling Model Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Model: {self.model.__class__.__name__}</p>
            <p>Device: {self.device}</p>
            <p>Variables: {self.variable_names if self.variable_names else 'Not specified'}</p>
            
            <div class="summary">
                <h2>Executive Summary</h2>
        """
        
        # Добавляем сводку
        if 'global_metrics' in all_metrics:
            gm = all_metrics['global_metrics']
            html += f"""
                <p><strong>Overall Performance:</strong></p>
                <ul>
                    <li>RMSE: <span class="metric-value">{gm.get('rmse', 'N/A'):.4f}</span></li>
                    <li>MAE: <span class="metric-value">{gm.get('mae', 'N/A'):.4f}</span></li>
                    <li>R²: <span class="metric-value {self._get_metric_class(gm.get('r_squared', 0), 'r2')}">{gm.get('r_squared', 'N/A'):.4f}</span></li>
                    <li>PSNR: <span class="metric-value {self._get_metric_class(gm.get('psnr', 0), 'psnr')}">{gm.get('psnr', 'N/A'):.2f} dB</span></li>
                    <li>Correlation: <span class="metric-value {self._get_metric_class(gm.get('correlation', 0), 'corr')}">{gm.get('correlation', 'N/A'):.4f}</span></li>
                </ul>
            """
        
        html += """
            </div>
            
            <h2>Detailed Metrics</h2>
            <h3>Global Metrics</h3>
        """
        
        # Таблица глобальных метрик
        if 'global_metrics' in all_metrics:
            html += "<table>"
            html += "<tr><th>Metric</th><th>Value</th><th>Interpretation</th></tr>"
            for metric, value in all_metrics['global_metrics'].items():
                if isinstance(value, (int, float)):
                    interpretation = self._interpret_metric(metric, value)
                    html += f"<tr><td>{metric.upper()}</td><td class='metric-value'>{value:.6f}</td><td>{interpretation}</td></tr>"
            html += "</table>"
        
        # Канальные метрики
        if 'channel_statistics' in all_metrics:
            html += "<h3>Channel-wise Statistics</h3>"
            for channel, stats in all_metrics['channel_statistics'].items():
                html += f"<h4>{channel}</h4>"
                html += "<table>"
                html += "<tr><th>Metric</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th></tr>"
                for stat_name, stat_value in stats.items():
                    if isinstance(stat_value, (int, float)):
                        html += f"<tr><td>{stat_name}</td><td>{stat_value:.6f}</td>"
                html += "</table>"
        
        # Спектральные метрики
        if 'spectral_metrics' in all_metrics:
            html += "<h3>Spectral Metrics</h3>"
            for channel, metrics in all_metrics['spectral_metrics'].items():
                html += f"<h4>{channel}</h4>"
                html += "<table>"
                html += "<tr><th>Metric</th><th>Value</th></tr>"
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)):
                        html += f"<tr><td>{metric}</td><td>{value:.6f}</td></tr>"
                html += "</table>"
        
        # Визуализации
        html += """
            <h2>Visualizations</h2>
            <div class="images">
        """
        
        # Добавляем изображения
        image_files = list(self.results_dir.glob('*.png'))
        for img_file in image_files[:12]:  # Ограничим 12 изображениями
            img_name = img_file.stem
            html += f"""
                <div class="image-container">
                    <img src="{img_file.name}" alt="{img_name}">
                    <div class="image-title">{img_name}</div>
                </div>
            """
        
        html += """
            </div>
            
            <h2>Files Generated</h2>
            <ul>
        """
        
        # Список файлов
        for file in self.results_dir.iterdir():
            if file.is_file():
                size_mb = file.stat().st_size / (1024 * 1024)
                html += f"<li>{file.name} ({size_mb:.2f} MB)</li>"
        
        html += """
            </ul>
            
            <footer style="margin-top: 50px; padding-top: 20px; border-top: 1px solid #ddd; text-align: center; color: #777;">
                <p>Report generated by Downscaling Model Tester</p>
                <p>For more information, contact the development team</p>
            </footer>
        </body>
        </html>
        """
        
        return html
    
    def _create_text_report(self, all_metrics):
        """Создание текстового отчета"""
        report = f"""
        ========================================
        ПОЛНЫЙ ОТЧЕТ О ТЕСТИРОВАНИИ МОДЕЛИ DOWNSCALING
        ========================================
        Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        Модель: {self.model.__class__.__name__}
        Устройство: {self.device}
        Переменные: {self.variable_names if self.variable_names else 'Не указаны'}
        
        СВОДКА:
        --------
        """
        
        if 'global_metrics' in all_metrics:
            gm = all_metrics['global_metrics']
            report += f"""
        Общее качество модели:
        - RMSE: {gm.get('rmse', 0):.6f}
        - MAE: {gm.get('mae', 0):.6f}
        - R²: {gm.get('r_squared', 0):.6f} ({self._interpret_metric('r_squared', gm.get('r_squared', 0))})
        - PSNR: {gm.get('psnr', 0):.2f} dB ({self._interpret_metric('psnr', gm.get('psnr', 0))})
        - Correlation: {gm.get('correlation', 0):.6f} ({self._interpret_metric('correlation', gm.get('correlation', 0))})
        
            """
        
        report += f"""
        ДЕТАЛЬНЫЕ МЕТРИКИ:
        ------------------
        """
        
        # Глобальные метрики
        if 'global_metrics' in all_metrics:
            report += "\nГлобальные метрики:\n"
            for metric, value in all_metrics['global_metrics'].items():
                if isinstance(value, (int, float)):
                    report += f"  {metric.upper()}: {value:.6f}\n"
        
        # Канальные метрики
        if 'channel_statistics' in all_metrics:
            report += "\nСтатистика по каналам:\n"
            for channel, stats in all_metrics['channel_statistics'].items():
                report += f"\n  {channel}:\n"
                for stat_name, stat_value in stats.items():
                    if isinstance(stat_value, (int, float)):
                        report += f"    {stat_name}: {stat_value:.6f}\n"
        
        report += f"""
        ФАЙЛЫ РЕЗУЛЬТАТОВ:
        ------------------
        Директория: {self.results_dir}
        
        Основные файлы:
        - spatial_metrics.zarr: Пространственные метрики в формате Zarr
        - predictions.npy: Массив предсказаний
        - targets.npy: Массив целей
        - all_metrics.json: Все метрики в JSON формате
        - comprehensive_report.html: HTML отчет
        - full_report.txt: Текстовый отчет
        
        Визуализации:
        - *_spatial_metrics.png: Карты пространственных метрик
        - *_spectral_analysis.png: Спектральный анализ
        - *_distributions.png: Распределения значений
        - energy_spectra_*.png: Energy Spectra анализ
        - summary_visualization.png: Сводная визуализация
        
        ВЫВОДЫ:
        -------
        """
        
        conclusions = self._generate_conclusions(all_metrics)
        report += conclusions
        with open(self.results_dir / 'full_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
    
    def _get_metric_class(self, value, metric_type):
        """Определение класса метрики для стилизации"""
        if metric_type == 'r2':
            return 'good' if value >= 0.7 else 'bad' if value < 0.5 else ''
        elif metric_type == 'psnr':
            return 'good' if value >= 30 else 'bad' if value < 20 else ''
        elif metric_type == 'corr':
            return 'good' if value >= 0.8 else 'bad' if value < 0.6 else ''
        return ''
    
    def _interpret_metric(self, metric, value):
        """Интерпретация значения метрики"""
        if metric == 'r_squared':
            if value >= 0.9:
                return "Отличное соответствие"
            elif value >= 0.7:
                return "Хорошее соответствие"
            elif value >= 0.5:
                return "Умеренное соответствие"
            else:
                return "Плохое соответствие"
        elif metric == 'psnr':
            if value >= 40:
                return "Отличное качество"
            elif value >= 30:
                return "Хорошее качество"
            elif value >= 20:
                return "Приемлемое качество"
            else:
                return "Плохое качество"
        elif metric == 'correlation':
            if value >= 0.9:
                return "Очень сильная корреляция"
            elif value >= 0.7:
                return "Сильная корреляция"
            elif value >= 0.5:
                return "Умеренная корреляция"
            elif value >= 0.3:
                return "Слабая корреляция"
            else:
                return "Очень слабая корреляция"
        elif metric == 'rmse':
            return "Меньше лучше"
        elif metric == 'mae':
            return "Меньше лучше"
        else:
            return ""
    
    def _generate_conclusions(self, all_metrics):
        """Генерация выводов на основе метрик"""
        conclusions = ""
        
        if 'global_metrics' in all_metrics:
            gm = all_metrics['global_metrics']
            
            if gm.get('r_squared', 0) >= 0.8:
                conclusions += "✓ Модель демонстрирует высокую объясняющую способность (R² ≥ 0.8)\n"
            elif gm.get('r_squared', 0) >= 0.6:
                conclusions += "⚠ Модель имеет умеренную объясняющую способность (0.6 ≤ R² < 0.8)\n"
            else:
                conclusions += "✗ Модель имеет низкую объясняющую способность (R² < 0.6)\n"
            
            if gm.get('psnr', 0) >= 35:
                conclusions += "✓ Отличное отношение сигнал/шум (PSNR ≥ 35 dB)\n"
            elif gm.get('psnr', 0) >= 25:
                conclusions += "⚠ Приемлемое отношение сигнал/шум (25 ≤ PSNR < 35 dB)\n"
            else:
                conclusions += "✗ Низкое отношение сигнал/шум (PSNR < 25 dB)\n"
            
            if gm.get('correlation', 0) >= 0.85:
                conclusions += "✓ Очень сильная корреляция с целевыми значениями\n"
            elif gm.get('correlation', 0) >= 0.7:
                conclusions += "⚠ Сильная корреляция с целевыми значениями\n"
            else:
                conclusions += "✗ Слабая корреляция с целевыми значениями\n"
            
            good_metrics = sum([
                1 if gm.get('r_squared', 0) >= 0.7 else 0,
                1 if gm.get('psnr', 0) >= 30 else 0,
                1 if gm.get('correlation', 0) >= 0.8 else 0
            ])
            
            if good_metrics >= 2:
                conclusions += "\n✅ ВЫВОД: Модель показывает хорошие результаты и может быть использована для downscaling.\n"
            elif good_metrics >= 1:
                conclusions += "\n⚠ ВЫВОД: Модель показывает умеренные результаты, требуется дополнительная оптимизация.\n"
            else:
                conclusions += "\n❌ ВЫВОД: Модель показывает плохие результаты, требуется значительная доработка.\n"
        
        return conclusions
