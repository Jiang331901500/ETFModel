import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from etf_model import ETFModel
from etf_dataset import ETFDataset

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
        
        # 分类头参数
        {'params': model.classification_head.parameters(), 'lr': head_lr}
    ]
    
    return torch.optim.AdamW(params_group, weight_decay=0.01)


class Trainer():
    model: ETFModel
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader

    def __init__(self, config: dict, device: torch.device):
        self.config = config['trainer_config']
        self.device = device
        self.model = ETFModel(config['model_config']).to(device)
        self.optimizer = create_optimizer(self.model)
        self.criterion = nn.CrossEntropyLoss().to(device)
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',          # Reduce LR when the monitored metric stops decreasing
            factor=0.5,         # Multiply LR by this factor when reducing
            patience=2,         # Number of epochs with no improvement after which LR will be reduced
        )

        self.model_name = config['etf_code'] + '_etf_model'

        # 定义分类边界
        self.boundaries = torch.tensor([-1, -0.5, -0.1, 0.1, 0.5, 1], device=device)

    def init_dataloader(self, full_dataset: ETFDataset):
        train_size = int(0.8 * len(full_dataset))
        val_size = int(0.1 * len(full_dataset))
        
        # 按时间划分数据集 (避免未来信息泄露) Sequential split to maintain time order
        train_dataset = torch.utils.data.Subset(full_dataset, range(0, train_size))
        val_dataset = torch.utils.data.Subset(full_dataset, range(train_size, train_size + val_size))
        test_dataset = torch.utils.data.Subset(full_dataset, range(train_size + val_size, len(full_dataset)))
        
        self.train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], 
                shuffle=True, num_workers=4, pin_memory=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], 
                            shuffle=False, num_workers=2)
        self.test_loader = DataLoader(test_dataset, batch_size=self.config['batch_size'], 
                            shuffle=False, num_workers=2)
    
    def _convert_to_class_labels(self, targets):
        """将连续涨跌幅转换为分类标签"""
        # 只取第一天的涨跌幅数据
        delta_day1 = targets[:, 0]  # shape: (batch_size,)
        
        # 使用bucketize进行分箱处理
        labels = torch.bucketize(delta_day1.contiguous(), self.boundaries)
        return labels
    
    def train(self):
        best_val_loss = float('inf')
        best_val_acc = 0.0
        epochs_no_improve = 0
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

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
            train_correct = 0
            train_total = 0
            for batch_idx, batch in enumerate(self.train_loader):
                if batch_idx % 4 == 0:
                    print(f"--- Epoch {epoch+1}/{num_epochs} | Train Batch {batch_idx+1}/{len(self.train_loader)}")
                tech_data = batch['tech_data'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                news_weights = batch['news_weights'].to(self.device)
                targets = batch['targets'].to(self.device)

                # 转换目标为分类标签 (只使用第一天数据)
                class_labels = self._convert_to_class_labels(targets)
                
                self.optimizer.zero_grad()
                
                # 前向传播
                outputs = self.model(tech_data, input_ids, attention_mask, news_weights)
                # 计算损失
                loss = self.criterion(outputs, class_labels)
                train_loss += loss.item()

                # 计算准确率
                _, predicted = torch.max(outputs.data, 1)
                train_total += class_labels.size(0)
                train_correct += (predicted == class_labels).sum().item()
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                
                # 参数更新
                self.optimizer.step()
            
            avg_train_loss = train_loss / len(self.train_loader)
            train_acc = train_correct / train_total
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_acc)
            
            # 验证阶段
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for batch_idx, batch in enumerate(self.val_loader):
                    if batch_idx % 4 == 0:
                        print(f"*** Epoch {epoch+1}/{num_epochs} | Validation Batch {batch_idx+1}/{len(self.val_loader)}")
                    tech_data = batch['tech_data'].to(self.device)
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    news_weights = batch['news_weights'].to(self.device)
                    targets = batch['targets'].to(self.device)

                    # 转换目标为分类标签
                    class_labels = self._convert_to_class_labels(targets)
                    
                    outputs = self.model(tech_data, input_ids, attention_mask, news_weights)
                    loss = self.criterion(outputs, class_labels)
                    val_loss += loss.item()

                    # 计算准确率
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += class_labels.size(0)
                    val_correct += (predicted == class_labels).sum().item()
            
            avg_val_loss = val_loss / len(self.val_loader)
            val_acc = val_correct / val_total
            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(val_acc)

            self.scheduler.step(avg_val_loss)
            
            # 打印进度
            print(f'*** Epoch {epoch+1}/{num_epochs} | '
                f'Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | '
                f'Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | '
                f'LR: agg_lr={self.optimizer.param_groups[1]["lr"]:.2e} gru_lr={self.optimizer.param_groups[2]["lr"]:.2e}')
            
            # 早停检查 (使用验证准确率作为主要指标)
            if val_acc > best_val_acc or (val_acc == best_val_acc and avg_val_loss < best_val_loss):
                best_val_acc = val_acc
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
                    'train_acc': train_acc,
                    'val_acc': val_acc,
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
        test_correct = 0
        test_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                if batch_idx % 4 == 0:
                    print(f"@@@ Batch {batch_idx+1}/{len(self.test_loader)}")
                tech_data = batch['tech_data'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                news_weights = batch['news_weights'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                # 转换目标为分类标签
                class_labels = self._convert_to_class_labels(targets)
                
                outputs = self.model(tech_data, input_ids, attention_mask, news_weights)
                loss = self.criterion(outputs, class_labels)
                test_loss += loss.item()
                
                # 计算准确率
                _, predicted = torch.max(outputs.data, 1)
                test_total += class_labels.size(0)
                test_correct += (predicted == class_labels).sum().item()
                
                # 收集预测结果用于进一步分析
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(class_labels.cpu().numpy())
        
        avg_test_loss = test_loss / len(self.test_loader)
        test_acc = test_correct / test_total
        print(f'@@@ Test Loss: {avg_test_loss:.6f}')
        print(f'@@@ Test Accuracy: {test_acc:.4f}')
        
        # 返回预测结果供进一步分析
        return np.array(all_preds), np.array(all_labels)
