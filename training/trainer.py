import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from pathlib import Path
from typing import Dict, Optional
import json

from .metrics import MetricLogger

class DownscalingTrainer:
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        experiment_name: str,
        criterion: nn.Module = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        gradient_clip: float = 1.0,
        early_stopping_patience: int = 10,
        use_topography: bool = True,
        topography_data: torch.Tensor = None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.lr_scheduler = lr_scheduler
        self.gradient_clip = gradient_clip
        self.early_stopping_patience = early_stopping_patience
        self.use_topography = use_topography
        
        if topography_data is not None:
            self.topography_data = topography_data.to(device, non_blocking=True)
        else:
            self.topography_data = None
        
        self.criterion = criterion if criterion is not None else nn.MSELoss()
        
        self.metric_logger = MetricLogger(experiment_name)
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        self.train_log_interval = max(1, len(train_loader) // 10) if train_loader else 1
        self.val_log_interval = max(1, len(val_loader) // 4) if val_loader else 1
        
        print(f"Инициализирован тренер для {experiment_name}")
        print(f"Модель: {model.__class__.__name__}")
        print(f"Устройство: {device}")
        print(f"Использование топографии: {use_topography}")
        print(f"Размер train_loader: {len(train_loader) if train_loader else 0}")
        print(f"Размер val_loader: {len(val_loader) if val_loader else 0}")

    def prepare_batch(self, batch: Dict) -> tuple:
        """Подготовка батча с учетом топографии"""
        try:
            inputs = batch['input'].to(self.device, non_blocking=True)
            targets = batch['target'].to(self.device, non_blocking=True)
            

            if inputs.dim() != 4 or targets.dim() != 4:
                raise ValueError(f"Некорректная размерность данных: inputs {inputs.shape}, targets {targets.shape}")
            
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
        except Exception as e:
            print(f"Ошибка при подготовке батча: {e}")
            print(f"Input shape: {batch['input'].shape if 'input' in batch else 'N/A'}")
            print(f"Target shape: {batch['target'].shape if 'target' in batch else 'N/A'}")
            raise

    def _run_epoch(self, loader: DataLoader, is_train: bool) -> Dict[str, float]:
        """Общая логика для train и validation эпох"""
        if loader is None or len(loader) == 0:
            print(f"Предупреждение: {'Train' if is_train else 'Val'} loader пуст")
            return {'loss': 0.0}
            
        self.model.train() if is_train else self.model.eval()
        
        epoch_loss = 0.0
        all_predictions = []
        all_targets = []
        
        total_batches = len(loader)
        log_interval = self.train_log_interval if is_train else self.val_log_interval
        
        pbar = tqdm(loader, desc=f"{'Training' if is_train else 'Validation'} Epoch", 
                   leave=False, disable=total_batches <= 10)
        
        try:
            with torch.set_grad_enabled(is_train):
                for batch_idx, batch in enumerate(pbar):
                    inputs, targets, topography = self.prepare_batch(batch)
                    
                    # if inputs.shape[2:] != targets.shape[2:]:
                    #     print(f"Размеры input {inputs.shape} и target {targets.shape} не совпадают")
                    
                    if self.use_topography and topography is not None:
                        outputs = self.model(inputs, topography)
                    else:
                        outputs = self.model(inputs)
                    
                    if outputs.shape != targets.shape:
                        print(f"ОШИБКА: выход модели {outputs.shape} не соответствует целям {targets.shape}")
                        if outputs.shape[2:] != targets.shape[2:]:
                            outputs = torch.nn.functional.interpolate(
                                outputs, 
                                size=targets.shape[2:], 
                                mode='bilinear', 
                                align_corners=False
                            )
                    
                    loss = self.criterion(outputs, targets)
                    
                    if is_train:
                        self.optimizer.zero_grad(set_to_none=True)  # Оптимизация: set_to_none=True для экономии памяти
                        loss.backward()
                        
                        if self.gradient_clip > 0:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                        
                        self.optimizer.step()
                    
                    epoch_loss += loss.item()
                    
                    if is_train:
                        all_predictions.append(outputs.detach().cpu())
                        all_targets.append(targets.detach().cpu())
                    else:
                        all_predictions.append(outputs.cpu())
                        all_targets.append(targets.cpu())
                    
                    pbar.set_postfix(loss=loss.item())
                    
                    if batch_idx % log_interval == 0:
                        mode = "Train" if is_train else "Val"
                        print(f"  {mode} Batch {batch_idx}/{total_batches}, Loss: {loss.item():.6f}")
                        
        except Exception as e:
            print(f"Ошибка во время {'training' if is_train else 'validation'}: {e}")
            raise
        
        finally:
            pbar.close()
        
        avg_loss = epoch_loss / total_batches
        
        if all_predictions:
            try:
                all_preds = torch.cat(all_predictions)
                all_targets_tensor = torch.cat(all_targets)
                metrics = self.metric_logger.compute_metrics(all_targets_tensor, all_preds)
            except Exception as e:
                print(f"Ошибка при вычислении метрик: {e}")
                metrics = {}
        else:
            metrics = {}
        
        metrics['loss'] = avg_loss
        
        if is_train:
            self.train_losses.append(avg_loss)
        else:
            self.val_losses.append(avg_loss)
        
        return metrics

    def train_epoch(self) -> Dict[str, float]:
        """Обучение на одной эпохе"""
        return self._run_epoch(self.train_loader, is_train=True)

    def validate_epoch(self) -> Dict[str, float]:
        """Валидация на одной эпохе"""
        return self._run_epoch(self.val_loader, is_train=False)

    def train(self, num_epochs: int, save_dir: str = "checkpoints"):
        """Полный цикл обучения"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Начало обучения на {num_epochs} эпох")
        print(f"Размер train_loader: {len(self.train_loader)} батчей")
        print(f"Размер val_loader: {len(self.val_loader)} батчей")
        
        self._validate_data_shapes()
        
        start_time = time.time()
        
        try:
            for epoch in range(num_epochs):
                print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
                epoch_start_time = time.time()
                
                train_metrics = self.train_epoch()
                val_metrics = self.validate_epoch()
                
                if self.lr_scheduler:
                    if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.lr_scheduler.step(val_metrics['loss'])
                    else:
                        self.lr_scheduler.step()
                
                self.metric_logger.log_epoch(train_metrics, val_metrics)
                
                current_val_loss = val_metrics['loss']
                if current_val_loss < self.best_val_loss:
                    self.best_val_loss = current_val_loss
                    self.epochs_without_improvement = 0
                    self.save_checkpoint(save_path / "best_model.pth", epoch)
                    print(f"Новая лучшая модель! Val Loss: {current_val_loss:.6f}")
                else:
                    self.epochs_without_improvement += 1
                
                if epoch % 5 == 0 or epoch == num_epochs - 1:
                    self.save_checkpoint(save_path / f"checkpoint_epoch_{epoch}.pth", epoch)
                
                if self.epochs_without_improvement >= self.early_stopping_patience:
                    print(f"Early stopping на эпохе {epoch + 1}")
                    break
                
                epoch_time = time.time() - epoch_start_time
                print(f"Epoch {epoch + 1} завершена за {epoch_time:.2f} сек")
                print(f"Train Loss: {train_metrics['loss']:.6f}, Val Loss: {val_metrics['loss']:.6f}")
                
                if 'mae' in train_metrics and 'mae' in val_metrics:
                    print(f"Train MAE: {train_metrics['mae']:.4f}, Val MAE: {val_metrics['mae']:.4f}")
        
        except KeyboardInterrupt:
            print("\nОбучение прервано пользователем")
        except Exception as e:
            print(f"\nОшибка во время обучения: {e}")
            raise
        
        finally:
            total_time = time.time() - start_time
            print(f"\nОбучение завершено за {total_time:.2f} секунд")
            
            self.save_checkpoint(save_path / "final_model.pth", epoch if 'epoch' in locals() else 0)
            
            best_epoch_info = self.metric_logger.get_best_epoch('loss', 'val')
            if best_epoch_info:
                print(f"\nЛучшая эпоха: {best_epoch_info.get('epoch', 'N/A')}")
                print(f"Лучшая Val Loss: {best_epoch_info.get('loss', 'N/A'):.6f}")

    def _validate_data_shapes(self):
        """Проверка совместимости размеров данных и модели"""
        print("Проверка совместимости размеров...")
        
        sample_batch = next(iter(self.train_loader))
        inputs, targets, topography = self.prepare_batch(sample_batch)
        
        print(f"Input shape: {inputs.shape}")
        print(f"Target shape: {targets.shape}")
        if topography is not None:
            print(f"Topography shape: {topography.shape}")
        
        self.model.eval()
        with torch.no_grad():
            if self.use_topography and topography is not None:
                outputs = self.model(inputs, topography)
            else:
                outputs = self.model(inputs)
            
            print(f"Model output shape: {outputs.shape}")
            
            if outputs.shape != targets.shape:
                print(f"Размер выхода модели {outputs.shape} не соответствует целям {targets.shape}")
                print("Модель будет автоматически адаптировать размеры во время обучения")
        
        self.model.train()

    def save_checkpoint(self, filepath: Path, epoch: int):
        """Сохранение checkpoint модели"""
        try:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'best_val_loss': self.best_val_loss,
                'metric_logger': self.metric_logger.metrics_history
            }
            
            if self.lr_scheduler:
                checkpoint['lr_scheduler_state_dict'] = self.lr_scheduler.state_dict()
            
            torch.save(checkpoint, filepath)
            print(f"Checkpoint сохранен: {filepath}")
        except Exception as e:
            print(f"Ошибка при сохранении checkpoint: {e}")

    def load_checkpoint(self, filepath: Path):
        """Загрузка checkpoint модели"""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if self.lr_scheduler and 'lr_scheduler_state_dict' in checkpoint:
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            
            self.train_losses = checkpoint['train_losses']
            self.val_losses = checkpoint['val_losses']
            self.best_val_loss = checkpoint['best_val_loss']
            self.metric_logger.metrics_history = checkpoint['metric_logger']
            
            print(f"Checkpoint загружен из {filepath}")
        except Exception as e:
            print(f"Ошибка при загрузке checkpoint: {e}")
            raise