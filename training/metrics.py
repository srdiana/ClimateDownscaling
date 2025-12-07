import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
import torch
import numpy as np
from datetime import datetime

class MetricLogger:
    """Логгер для сохранения метрик в JSON файлы"""
    
    def __init__(self, experiment_name: str, base_dir: str = "../metrics"):
        self.experiment_name = experiment_name
        self.base_dir = Path(base_dir)
        self.train_dir = self.base_dir / "train"
        self.val_dir = self.base_dir / "val"
        
        # Создание директорий
        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.val_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_epoch = 0
        self.metrics_history = {
            'train': [],
            'val': []
        }
    
    def compute_metrics(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
        """Вычисление всех метрик"""
        y_true_np = y_true.detach().cpu().numpy()
        y_pred_np = y_pred.detach().cpu().numpy()
        
    
        mae = np.mean(np.abs(y_true_np - y_pred_np))
        mse = np.mean((y_true_np - y_pred_np) ** 2)
        rmse = np.sqrt(mse)
        data_range = np.max(y_true_np) - np.min(y_true_np)
        if data_range == 0:
            psnr = float('inf')
        else:
            psnr = 20 * np.log10(data_range / np.sqrt(mse))
        
        return {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'psnr': float(psnr) if psnr != float('inf') else 100.0
        }
    
    def log_epoch(self, train_metrics: Dict, val_metrics: Optional[Dict] = None):
        """Логирование метрик эпохи"""
        epoch_data = {
            'epoch': self.current_epoch,
            'timestamp': datetime.now().isoformat(),
            'train_metrics': train_metrics
        }
        
        if val_metrics:
            epoch_data['val_metrics'] = val_metrics
        
        # Сохранение в историю
        self.metrics_history['train'].append(train_metrics)
        if val_metrics:
            self.metrics_history['val'].append(val_metrics)
        
        # Сохранение в отдельный файл для эпохи
        epoch_filename = f"epoch_{self.current_epoch:04d}.json"
        train_file_path = self.train_dir / epoch_filename
        
        with open(train_file_path, 'w') as f:
            json.dump(epoch_data, f, indent=2)
        
        # Обновление общего файла с историей
        history_file = self.base_dir / f"{self.experiment_name}_history.json"
        with open(history_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        self.current_epoch += 1
        print(f"Метрики эпохи {self.current_epoch - 1} сохранены")
    
    def get_best_epoch(self, metric: str = 'mse', mode: str = 'val') -> Dict:
        """Получение лучшей эпохи по метрике"""
        if not self.metrics_history[mode]:
            return {}
        
        if metric == 'psnr':
            best_value = max(epoch[metric] for epoch in self.metrics_history[mode])
        else:
            best_value = min(epoch[metric] for epoch in self.metrics_history[mode])
        
        best_epoch_idx = None
        for i, epoch_metrics in enumerate(self.metrics_history[mode]):
            if epoch_metrics[metric] == best_value:
                best_epoch_idx = i
                break
        
        return {
            'epoch': best_epoch_idx,
            metric: best_value,
            'all_metrics': self.metrics_history[mode][best_epoch_idx]
        }