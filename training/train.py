"""
train.py - Multi-Task Training Loop

Implements phased training strategy:
    Phase 1: Classification head warmup (backbone frozen)
    Phase 2: Detection head warmup (backbone frozen)  
    Phase 3: Joint fine-tuning (all parameters, differential LR)

Handles disjoint datasets by alternating batches and masking losses.
"""

import os
import time
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

from model.multitask_net import MultiTaskNet
from model.losses import MultiTaskLoss
from training.config import Config
from training.dataset import create_dataloaders


class MultiTaskTrainer:
    """
    Trainer for the multi-task vehicle recognition model.
    
    Handles:
    - Phased training (warmup → joint fine-tuning)
    - Alternating batches from disjoint datasets
    - Masked loss computation
    - Mixed precision training
    - Gradient clipping
    - Checkpointing
    """
    
    def __init__(self, config=None):
        self.config = config or Config
        self.config.ensure_dirs()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
        
        # Initialize model
        self.model = MultiTaskNet(
            num_vehicle_classes=self.config.NUM_VEHICLE_CLASSES,
            pretrained_backbone=self.config.PRETRAINED_BACKBONE
        ).to(self.device)
        
        # Loss function
        self.criterion = MultiTaskLoss().to(self.device)
        
        # Mixed precision scaler
        self.scaler = GradScaler(enabled=self.config.USE_AMP)
        
        # Training history
        self.history = {
            'train_loss': [], 'val_loss': [],
            'cls_loss': [], 'det_loss': [],
            'cls_acc': [], 'det_iou': []
        }
    
    def _create_optimizer(self, phase):
        """Create optimizer for the given training phase."""
        if phase == 1:
            # Only classification head
            params = self.model.classification_head.parameters()
            optimizer = torch.optim.AdamW(params, lr=self.config.PHASE1_LR,
                                          weight_decay=self.config.WEIGHT_DECAY)
        elif phase == 2:
            # Only detection head
            params = self.model.detection_head.parameters()
            optimizer = torch.optim.AdamW(params, lr=self.config.PHASE2_LR,
                                          weight_decay=self.config.WEIGHT_DECAY)
        elif phase == 3:
            # All parameters with differential LR
            param_groups = self.model.get_param_groups(
                backbone_lr=self.config.PHASE3_BACKBONE_LR,
                head_lr=self.config.PHASE3_HEAD_LR
            )
            # Add loss function parameters (uncertainty weights)
            param_groups.append({
                'params': self.criterion.parameters(),
                'lr': self.config.PHASE3_HEAD_LR
            })
            optimizer = torch.optim.AdamW(param_groups,
                                          weight_decay=self.config.WEIGHT_DECAY)
        
        return optimizer
    
    def _create_scheduler(self, optimizer, num_epochs):
        """Create learning rate scheduler."""
        if self.config.SCHEDULER == 'cosine':
            return CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)
        else:
            return MultiStepLR(optimizer, 
                             milestones=self.config.STEP_LR_MILESTONES,
                             gamma=self.config.STEP_LR_GAMMA)
    
    def train_phase1(self, vehicle_loader):
        """
        Phase 1: Train classification head only.
        Backbone is frozen — only head weights update.
        """
        print("\n" + "=" * 60)
        print("PHASE 1: Classification Head Warmup")
        print("=" * 60)
        
        self.model.backbone.freeze()
        optimizer = self._create_optimizer(phase=1)
        scheduler = self._create_scheduler(optimizer, self.config.PHASE1_EPOCHS)
        
        for epoch in range(self.config.PHASE1_EPOCHS):
            self.model.train()
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, batch in enumerate(vehicle_loader):
                images = batch['image'].to(self.device)
                labels = batch['class_label'].to(self.device)
                
                optimizer.zero_grad()
                
                with autocast(device_type='cuda', enabled=self.config.USE_AMP):
                    output = self.model(images, task='classify')
                    predictions = {'class_logits': output['class_logits']}
                    targets = {'class_labels': labels}
                    loss, loss_dict = self.criterion(predictions, targets, task='classify')
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.GRADIENT_CLIP_NORM
                )
                self.scaler.step(optimizer)
                self.scaler.update()
                
                epoch_loss += loss.item()
                
                # Accuracy
                _, predicted = output['class_logits'].max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                if (batch_idx + 1) % self.config.LOG_INTERVAL == 0:
                    print(f"  Epoch [{epoch+1}/{self.config.PHASE1_EPOCHS}] "
                          f"Batch [{batch_idx+1}/{len(vehicle_loader)}] "
                          f"Loss: {loss.item():.4f} "
                          f"Acc: {100.*correct/total:.1f}%")
            
            scheduler.step()
            avg_loss = epoch_loss / len(vehicle_loader)
            accuracy = 100. * correct / total
            print(f"  Epoch {epoch+1} — Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.1f}%")
            
            self.history['cls_loss'].append(avg_loss)
            self.history['cls_acc'].append(accuracy)
    
    def train_phase2(self, plate_loader):
        """
        Phase 2: Train detection head only.
        Backbone is frozen — only head weights update.
        """
        print("\n" + "=" * 60)
        print("PHASE 2: Detection Head Warmup")
        print("=" * 60)
        
        self.model.backbone.freeze()
        optimizer = self._create_optimizer(phase=2)
        scheduler = self._create_scheduler(optimizer, self.config.PHASE2_EPOCHS)
        
        for epoch in range(self.config.PHASE2_EPOCHS):
            self.model.train()
            epoch_loss = 0.0
            
            for batch_idx, batch in enumerate(plate_loader):
                images = batch['image'].to(self.device)
                bbox_labels = batch['bbox_label'].to(self.device)
                
                optimizer.zero_grad()
                
                with autocast(device_type='cuda', enabled=self.config.USE_AMP):
                    output = self.model(images, task='detect')
                    predictions = {'bbox': output['bbox']}
                    targets = {'bbox_labels': bbox_labels}
                    loss, loss_dict = self.criterion(predictions, targets, task='detect')
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.GRADIENT_CLIP_NORM
                )
                self.scaler.step(optimizer)
                self.scaler.update()
                
                epoch_loss += loss.item()
                
                if (batch_idx + 1) % self.config.LOG_INTERVAL == 0:
                    print(f"  Epoch [{epoch+1}/{self.config.PHASE2_EPOCHS}] "
                          f"Batch [{batch_idx+1}/{len(plate_loader)}] "
                          f"Loss: {loss.item():.4f}")
            
            scheduler.step()
            avg_loss = epoch_loss / len(plate_loader)
            print(f"  Epoch {epoch+1} — Avg Loss: {avg_loss:.4f}")
            
            self.history['det_loss'].append(avg_loss)
    
    def train_phase3(self, vehicle_loader, plate_loader):
        """
        Phase 3: Joint fine-tuning.
        All parameters unfrozen, alternating batches, differential LR.
        """
        print("\n" + "=" * 60)
        print("PHASE 3: Joint Fine-Tuning")
        print("=" * 60)
        
        self.model.backbone.unfreeze()
        optimizer = self._create_optimizer(phase=3)
        scheduler = self._create_scheduler(optimizer, self.config.PHASE3_EPOCHS)
        
        for epoch in range(self.config.PHASE3_EPOCHS):
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            # Create iterators
            vehicle_iter = iter(vehicle_loader)
            plate_iter = iter(plate_loader)
            
            # Alternate between datasets
            max_batches = max(len(vehicle_loader), len(plate_loader))
            
            for batch_idx in range(max_batches):
                # ---- Vehicle classification batch ----
                try:
                    vehicle_batch = next(vehicle_iter)
                except StopIteration:
                    vehicle_iter = iter(vehicle_loader)
                    vehicle_batch = next(vehicle_iter)
                
                images_v = vehicle_batch['image'].to(self.device)
                labels_v = vehicle_batch['class_label'].to(self.device)
                
                optimizer.zero_grad()
                
                with autocast(device_type='cuda', enabled=self.config.USE_AMP):
                    out_v = self.model(images_v, task='classify')
                    loss_v, _ = self.criterion(
                        {'class_logits': out_v['class_logits']},
                        {'class_labels': labels_v},
                        task='classify'
                    )
                
                self.scaler.scale(loss_v).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.GRADIENT_CLIP_NORM
                )
                self.scaler.step(optimizer)
                self.scaler.update()
                
                # ---- Plate detection batch ----
                try:
                    plate_batch = next(plate_iter)
                except StopIteration:
                    plate_iter = iter(plate_loader)
                    plate_batch = next(plate_iter)
                
                images_p = plate_batch['image'].to(self.device)
                bbox_p = plate_batch['bbox_label'].to(self.device)
                
                optimizer.zero_grad()
                
                with autocast(device_type='cuda', enabled=self.config.USE_AMP):
                    out_p = self.model(images_p, task='detect')
                    loss_p, _ = self.criterion(
                        {'bbox': out_p['bbox']},
                        {'bbox_labels': bbox_p},
                        task='detect'
                    )
                
                self.scaler.scale(loss_p).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.GRADIENT_CLIP_NORM
                )
                self.scaler.step(optimizer)
                self.scaler.update()
                
                batch_loss = (loss_v.item() + loss_p.item()) / 2
                epoch_loss += batch_loss
                num_batches += 1
                
                if (batch_idx + 1) % self.config.LOG_INTERVAL == 0:
                    print(f"  Epoch [{epoch+1}/{self.config.PHASE3_EPOCHS}] "
                          f"Batch [{batch_idx+1}/{max_batches}] "
                          f"CLS: {loss_v.item():.4f} DET: {loss_p.item():.4f}")
            
            scheduler.step()
            avg_loss = epoch_loss / max(num_batches, 1)
            print(f"  Epoch {epoch+1} — Avg Combined Loss: {avg_loss:.4f}")
            
            self.history['train_loss'].append(avg_loss)
            
            # Save checkpoint
            if (epoch + 1) % self.config.SAVE_INTERVAL == 0:
                self.save_checkpoint(f"checkpoint_phase3_epoch{epoch+1}.pth")
    
    def validate_classification(self, val_loader):
        """Evaluate classification accuracy on validation set."""
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(self.device)
                labels = batch['class_label'].to(self.device)
                
                with autocast(device_type='cuda', enabled=self.config.USE_AMP):
                    output = self.model(images, task='classify')
                    loss, _ = self.criterion(
                        {'class_logits': output['class_logits']},
                        {'class_labels': labels},
                        task='classify'
                    )
                
                total_loss += loss.item()
                _, predicted = output['class_logits'].max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        accuracy = 100. * correct / max(total, 1)
        avg_loss = total_loss / max(len(val_loader), 1)
        return accuracy, avg_loss
    
    def validate_detection(self, val_loader):
        """Evaluate detection IoU on validation set."""
        self.model.eval()
        total_iou = 0.0
        total_loss = 0.0
        count = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(self.device)
                bbox_labels = batch['bbox_label'].to(self.device)
                
                with autocast(device_type='cuda', enabled=self.config.USE_AMP):
                    output = self.model(images, task='detect')
                    loss, _ = self.criterion(
                        {'bbox': output['bbox']},
                        {'bbox_labels': bbox_labels},
                        task='detect'
                    )
                
                total_loss += loss.item()
                
                # Compute IoU per sample
                pred = output['bbox'].cpu()
                gt = bbox_labels.cpu()
                for j in range(pred.size(0)):
                    iou = self._compute_iou(pred[j], gt[j])
                    total_iou += iou
                    count += 1
        
        avg_iou = total_iou / max(count, 1)
        avg_loss = total_loss / max(len(val_loader), 1)
        return avg_iou, avg_loss
    
    @staticmethod
    def _compute_iou(pred, gt):
        """Compute IoU between two [cx, cy, w, h] boxes."""
        # Convert to xyxy
        px1 = pred[0] - pred[2] / 2
        py1 = pred[1] - pred[3] / 2
        px2 = pred[0] + pred[2] / 2
        py2 = pred[1] + pred[3] / 2
        
        gx1 = gt[0] - gt[2] / 2
        gy1 = gt[1] - gt[3] / 2
        gx2 = gt[0] + gt[2] / 2
        gy2 = gt[1] + gt[3] / 2
        
        ix1 = max(px1.item(), gx1.item())
        iy1 = max(py1.item(), gy1.item())
        ix2 = min(px2.item(), gx2.item())
        iy2 = min(py2.item(), gy2.item())
        
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_p = max(0, (px2 - px1).item()) * max(0, (py2 - py1).item())
        area_g = max(0, (gx2 - gx1).item()) * max(0, (gy2 - gy1).item())
        union = area_p + area_g - inter + 1e-7
        
        return inter / union
    
    def save_checkpoint(self, filename):
        """Save model checkpoint."""
        path = os.path.join(self.config.WEIGHTS_DIR, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'criterion_state_dict': self.criterion.state_dict(),
            'history': self.history,
            'config': {
                'num_classes': self.config.NUM_VEHICLE_CLASSES,
                'input_size': self.config.INPUT_SIZE
            }
        }, path)
        print(f"  Checkpoint saved: {path}")
    
    def train(self):
        """Run full training pipeline (all 3 phases)."""
        self.config.summary()
        
        # Create dataloaders
        print("\nLoading datasets...")
        vehicle_train, vehicle_val, plate_train, plate_val = create_dataloaders(self.config)
        
        start_time = time.time()
        best_cls_acc = 0.0
        best_det_iou = 0.0
        
        # Phase 1: Classification warmup
        if len(vehicle_train.dataset) > 0:
            self.train_phase1(vehicle_train)
            # Validate after Phase 1
            if len(vehicle_val.dataset) > 0:
                acc, val_loss = self.validate_classification(vehicle_val)
                print(f"  Phase 1 Val — Accuracy: {acc:.1f}%, Loss: {val_loss:.4f}")
                best_cls_acc = acc
            self.save_checkpoint("after_phase1.pth")
        else:
            print("WARNING: No vehicle dataset found, skipping Phase 1")
        
        # Phase 2: Detection warmup
        if len(plate_train.dataset) > 0:
            self.train_phase2(plate_train)
            # Validate after Phase 2
            if len(plate_val.dataset) > 0:
                iou, val_loss = self.validate_detection(plate_val)
                print(f"  Phase 2 Val — IoU: {iou:.4f}, Loss: {val_loss:.4f}")
                best_det_iou = iou
            self.save_checkpoint("after_phase2.pth")
        else:
            print("WARNING: No plate dataset found, skipping Phase 2")
        
        # Phase 3: Joint fine-tuning with per-epoch validation
        if len(vehicle_train.dataset) > 0 and len(plate_train.dataset) > 0:
            print("\n" + "=" * 60)
            print("PHASE 3: Joint Fine-Tuning")
            print("=" * 60)
            
            self.model.backbone.unfreeze()
            optimizer = self._create_optimizer(phase=3)
            scheduler = self._create_scheduler(optimizer, self.config.PHASE3_EPOCHS)
            
            for epoch in range(self.config.PHASE3_EPOCHS):
                self.model.train()
                epoch_loss = 0.0
                num_batches = 0
                
                # Create iterators
                vehicle_iter = iter(vehicle_train)
                plate_iter = iter(plate_train)
                max_batches = max(len(vehicle_train), len(plate_train))
                
                for batch_idx in range(max_batches):
                    # ---- Vehicle classification batch ----
                    try:
                        vehicle_batch = next(vehicle_iter)
                    except StopIteration:
                        vehicle_iter = iter(vehicle_train)
                        vehicle_batch = next(vehicle_iter)
                    
                    images_v = vehicle_batch['image'].to(self.device)
                    labels_v = vehicle_batch['class_label'].to(self.device)
                    
                    optimizer.zero_grad()
                    
                    with autocast(device_type='cuda', enabled=self.config.USE_AMP):
                        out_v = self.model(images_v, task='classify')
                        loss_v, _ = self.criterion(
                            {'class_logits': out_v['class_logits']},
                            {'class_labels': labels_v},
                            task='classify'
                        )
                    
                    self.scaler.scale(loss_v).backward()
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.GRADIENT_CLIP_NORM
                    )
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    
                    # ---- Plate detection batch ----
                    try:
                        plate_batch = next(plate_iter)
                    except StopIteration:
                        plate_iter = iter(plate_train)
                        plate_batch = next(plate_iter)
                    
                    images_p = plate_batch['image'].to(self.device)
                    bbox_p = plate_batch['bbox_label'].to(self.device)
                    
                    optimizer.zero_grad()
                    
                    with autocast(device_type='cuda', enabled=self.config.USE_AMP):
                        out_p = self.model(images_p, task='detect')
                        loss_p, _ = self.criterion(
                            {'bbox': out_p['bbox']},
                            {'bbox_labels': bbox_p},
                            task='detect'
                        )
                    
                    self.scaler.scale(loss_p).backward()
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.GRADIENT_CLIP_NORM
                    )
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    
                    batch_loss = (loss_v.item() + loss_p.item()) / 2
                    epoch_loss += batch_loss
                    num_batches += 1
                    
                    if (batch_idx + 1) % self.config.LOG_INTERVAL == 0:
                        print(f"  Epoch [{epoch+1}/{self.config.PHASE3_EPOCHS}] "
                              f"Batch [{batch_idx+1}/{max_batches}] "
                              f"CLS: {loss_v.item():.4f} DET: {loss_p.item():.4f}")
                
                scheduler.step()
                avg_loss = epoch_loss / max(num_batches, 1)
                self.history['train_loss'].append(avg_loss)
                
                # ---- Validation after each epoch ----
                val_msg = f"  Epoch {epoch+1} — Train Loss: {avg_loss:.4f}"
                
                if len(vehicle_val.dataset) > 0:
                    acc, cls_val_loss = self.validate_classification(vehicle_val)
                    val_msg += f"  |  Val Acc: {acc:.1f}%"
                    self.history['cls_acc'].append(acc)
                    if acc > best_cls_acc:
                        best_cls_acc = acc
                        self.save_checkpoint("best_cls.pth")
                
                if len(plate_val.dataset) > 0:
                    iou, det_val_loss = self.validate_detection(plate_val)
                    val_msg += f"  |  Val IoU: {iou:.4f}"
                    self.history['det_iou'].append(iou)
                    if iou > best_det_iou:
                        best_det_iou = iou
                        self.save_checkpoint("best_det.pth")
                
                print(val_msg)
                
                # Save periodic checkpoint
                if (epoch + 1) % self.config.SAVE_INTERVAL == 0:
                    self.save_checkpoint(f"checkpoint_phase3_epoch{epoch+1}.pth")
        else:
            print("WARNING: Need both datasets for Phase 3")
        
        # Final save
        self.save_checkpoint("final_model.pth")
        
        elapsed = time.time() - start_time
        print(f"\nTraining complete! Total time: {elapsed/60:.1f} minutes")
        print(f"Best classification accuracy: {best_cls_acc:.1f}%")
        print(f"Best detection IoU: {best_det_iou:.4f}")
        print(f"Final model saved to: {os.path.join(self.config.WEIGHTS_DIR, 'final_model.pth')}")


if __name__ == "__main__":
    trainer = MultiTaskTrainer()
    trainer.train()
