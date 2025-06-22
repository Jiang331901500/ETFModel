import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from etf_model import ETFModel
from etf_dataset import ETFDataset

# huber损失函数定义
class WeightedHuberLoss(nn.Module):
    """
    时间加权Huber损失函数
    更重视近期预测的准确性
    """
    def __init__(self, delta=0.5, decay=0.5):
        super().__init__()
        self.delta = delta
        self.decay = decay
        self.weights = torch.tensor([decay ** i for i in range(5)], dtype=torch.float32)
    
    def forward(self, pred, target):
        residual = torch.abs(pred - target)
        condition = residual < self.delta
        
        # Huber损失计算
        loss = torch.where(
            condition,
            0.5 * residual ** 2,
            self.delta * (residual - 0.5 * self.delta)
        )
        
        # 时间加权
        weighted_loss = loss * self.weights.to(loss.device)
        return torch.mean(weighted_loss)

def create_optimizer(model: ETFModel, bert_lr=1e-5, agg_lr=1e-3, gru_lr=1e-3, head_lr=1e-3):
    params_group = [
        # BERT参数 (较低学习率)
        {'params': model.news_encoder.bert.parameters(), 'lr': bert_lr},
        
        # 新闻聚合层
        {'params': model.news_encoder.aggregate.parameters(), 'lr': agg_lr},
        
        # 特征融合层
        {'params': model.fusion.parameters(), 'lr': gru_lr},
        
        # GRU参数
        {'params': model.gru.parameters(), 'lr': gru_lr},
        
        # 预测头参数
        {'params': model.prediction_heads.parameters(), 'lr': head_lr}
    ]
    
    return torch.optim.AdamW(params_group, weight_decay=0.01)


class Trainer():
    model: ETFModel
    optimizer: torch.optim.Optimizer
    criterion: WeightedHuberLoss
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader

    def __init__(self, config: dict, device: torch.device):
        self.config = config['trainer_config']
        self.device = device
        self.model = ETFModel(config['model_config']).to(device)
        self.optimizer = create_optimizer(self.model)
        self.criterion = WeightedHuberLoss().to(device)
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',          # Reduce LR when the monitored metric stops decreasing
            factor=0.5,         # Multiply LR by this factor when reducing
            patience=2,         # Number of epochs with no improvement after which LR will be reduced
        )

        self.model_name = config['etf_code'] + '_etf_model'

    def init_dataloader(self, full_dataset: ETFDataset):
        train_size = int(0.8 * len(full_dataset))
        val_size = int(0.1 * len(full_dataset))
        
        # 按时间划分数据集 (避免未来信息泄露) Sequential split to maintain time order
        train_dataset = torch.utils.data.Subset(full_dataset, range(0, train_size))
        val_dataset = torch.utils.data.Subset(full_dataset, range(train_size, train_size + val_size))
        test_dataset = torch.utils.data.Subset(full_dataset, range(train_size + val_size, len(full_dataset)))
        
        # shuffle=False确保样本不被打乱顺序 Create data loader with shuffle=False to maintain sequence order
        self.train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], 
                shuffle=True, num_workers=4, pin_memory=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], 
                            shuffle=False, num_workers=2)
        self.test_loader = DataLoader(test_dataset, batch_size=self.config['batch_size'], 
                            shuffle=False, num_workers=2)
        
    def train(self):
        best_val_loss = float('inf')
        epochs_no_improve = 0
        history = {'train_loss': [], 'val_loss': []}

        grad_clip = self.config['grad_clip']
        num_epochs = self.config['num_epochs']
        patience = self.config['patience']

        model_save_path = self.config['model_save_path'] + self.model_name + '.pth'
        checkpoint_path = self.config['checkpoint_path'] + self.model_name + '_checkpoint.pth'
        checkpoint_interval = self.config['checkpoint_interval']

        # 检查是否有检查点，若有则加载检查点并继续训练
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            history = checkpoint['history']
            print(f"** Resuming training from epoch {start_epoch}")
        except FileNotFoundError:
            start_epoch = 0
            print("** No checkpoint found, starting fresh training.")
        
        for epoch in range(start_epoch, num_epochs):
            print("-" * 30)
            # 训练阶段
            self.model.train()
            train_loss = 0
            for batch_idx, batch in enumerate(self.train_loader):
                if batch_idx % 4 == 0:
                    print(f"--- Epoch {epoch+1}/{num_epochs} | Train Batch {batch_idx+1}/{len(self.train_loader)}")
                tech_data = batch['tech_data'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                news_weights = batch['news_weights'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                self.optimizer.zero_grad()
                
                # 前向传播
                outputs = self.model(tech_data, input_ids, attention_mask, news_weights)
                # 计算损失
                loss = self.criterion(outputs, targets)
                train_loss += loss.item()
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                
                # 参数更新
                self.optimizer.step()
            
            avg_train_loss = train_loss / len(self.train_loader)
            history['train_loss'].append(avg_train_loss)
            
            # 验证阶段
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_idx, batch in enumerate(self.val_loader):
                    if batch_idx % 4 == 0:
                        print(f"*** Epoch {epoch+1}/{num_epochs} | Validation Batch {batch_idx+1}/{len(self.val_loader)}")
                    tech_data = batch['tech_data'].to(self.device)
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    news_weights = batch['news_weights'].to(self.device)
                    targets = batch['targets'].to(self.device)
                    
                    outputs = self.model(tech_data, input_ids, attention_mask, news_weights)
                    loss = self.criterion(outputs, targets)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(self.val_loader)
            history['val_loss'].append(avg_val_loss)
            self.scheduler.step(avg_val_loss)
            
            # 打印进度
            print(f'*** Epoch {epoch+1}/{num_epochs} | '
                f'Train Loss: {avg_train_loss:.6f} | '
                f'Val Loss: {avg_val_loss:.6f} | '
                f'LR: agg_lr={self.optimizer.param_groups[1]["lr"]:.2e} gru_lr={self.optimizer.param_groups[2]["lr"]:.2e}')
            
            # 早停检查
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(self.model.state_dict(), model_save_path)
                print(f'---Saving best model at epoch {epoch+1} ...')
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f'!!!Early stopping at epoch {epoch+1} ...')
                    break

            # 训练中途保存模型检查点，意外中断后下次从检查点继续
            if (epoch + 1) % checkpoint_interval == 0 :
                print(f'---Saving checkpoint at epoch {epoch + 1} ...')
                # 保存检查点
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'history': history
                }
                torch.save(checkpoint, checkpoint_path)
    
    def load_trained_model(self):
        # 加载最佳模型
        model_save_path = self.config['model_save_path'] + self.model_name + '.pth'
        self.model.load_state_dict(torch.load(model_save_path, map_location=self.device, weights_only=False))

    def test(self):
        # 测试评估
        self.model.eval()
        test_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                if batch_idx % 4 == 0:
                    print(f"@@@ Batch {batch_idx+1}/{len(self.test_loader)}")
                tech_data = batch['tech_data'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                news_weights = batch['news_weights'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                outputs = self.model(tech_data, input_ids, attention_mask, news_weights)
                loss = self.criterion(outputs, targets)
                test_loss += loss.item()
                
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        avg_test_loss = test_loss / len(self.test_loader)
        print(f'@@@ Test Loss: {avg_test_loss:.6f}')
        
        # 计算方向准确性
        preds_array = np.vstack(all_preds)
        targets_array = np.vstack(all_targets)
        
        # 计算每日方向准确率
        daily_acc = []
        for i in range(5):
            correct = np.sign(preds_array[:, i]) == np.sign(targets_array[:, i])
            acc = correct.mean()
            daily_acc.append(acc)
            print(f'@@@ Day {i+1} Direction Accuracy: {acc:.4f}')
        
        print(f'@@@ Average Direction Accuracy: {np.mean(daily_acc):.4f}')

    def test_next_day_dir_accuracy(self):
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                if batch_idx % 4 == 0:
                    print(f"@@@ Batch {batch_idx+1}/{len(self.test_loader)}")
                tech_data = batch['tech_data'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                news_weights = batch['news_weights'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                outputs = self.model(tech_data, input_ids, attention_mask, news_weights)
                
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        # 计算方向准确性
        preds_array = np.vstack(all_preds)
        targets_array = np.vstack(all_targets)
        
        # 计算每日方向准确率
        correct_0_05 = ((np.sign(preds_array[:, 0]) == np.sign(targets_array[:, 0])) | 
                   (np.abs(preds_array[:, 0] - targets_array[:, 0]) <= 0.05))
        acc_0_05 = correct_0_05.mean()
        correct_0_1 = ((np.sign(preds_array[:, 0]) == np.sign(targets_array[:, 0])) | 
                   (np.abs(preds_array[:, 0] - targets_array[:, 0]) <= 0.1))
        acc_0_1 = correct_0_1.mean()
        correct_0_5 = ((np.sign(preds_array[:, 0]) == np.sign(targets_array[:, 0])) | 
                   (np.abs(preds_array[:, 0] - targets_array[:, 0]) <= 0.5))
        acc_0_5 = correct_0_5.mean()
        correct_1_0 = ((np.sign(preds_array[:, 0]) == np.sign(targets_array[:, 0])) | 
                   (np.abs(preds_array[:, 0] - targets_array[:, 0]) <= 1.0))
        acc_1_0 = correct_1_0.mean()
        
        print(f'@@@ Threshold=±0.05% Average +1 Day Direction Accuracy: {np.mean(acc_0_05):.4f}')
        print(f'@@@ Threshold=±0.1% Average +1 Day Direction Accuracy: {np.mean(acc_0_1):.4f}')
        print(f'@@@ Threshold=±0.5% Average +1 Day Direction Accuracy: {np.mean(acc_0_5):.4f}')
        print(f'@@@ Threshold=±1.0% Average +1 Day Direction Accuracy: {np.mean(acc_1_0):.4f}')

        # 计算每日数值准确率
        correct_0_05 = (np.abs(preds_array[:, 0] - targets_array[:, 0]) <= 0.05)
        acc_0_05 = correct_0_05.mean()
        correct_0_1 = (np.abs(preds_array[:, 0] - targets_array[:, 0]) <= 0.1)
        acc_0_1 = correct_0_1.mean()
        correct_0_5 = (np.abs(preds_array[:, 0] - targets_array[:, 0]) <= 0.5)
        acc_0_5 = correct_0_5.mean()
        correct_1_0 = (np.abs(preds_array[:, 0] - targets_array[:, 0]) <= 1.0)
        acc_1_0 = correct_1_0.mean()
        
        print(f'@@@ Threshold=±0.05% Average +1 Day Value Accuracy: {np.mean(acc_0_05):.4f}')
        print(f'@@@ Threshold=±0.1% Average +1 Day Value Accuracy: {np.mean(acc_0_1):.4f}')
        print(f'@@@ Threshold=±0.5% Average +1 Day Value Accuracy: {np.mean(acc_0_5):.4f}')
        print(f'@@@ Threshold=±1.0% Average +1 Day Value Accuracy: {np.mean(acc_1_0):.4f}')