import torch
from torch.utils.data import Dataset
import pandas as pd

class ETFDataset(Dataset):
    def __init__(self, etf_df, news_dict, tokenizer, window, pred_days):
        self.tokenizer = tokenizer
        self.window = window
        self.pred_days = pred_days
        self.etf_df = etf_df
        self.news_dict = news_dict
        self.dates = etf_df.index[window-1:-pred_days]  # 确保有足够的历史数据和未来预测目标
        
        # 技术特征列 (不包括目标列)
        self.tech_cols = ['close', 'vol', 'amount', 'delta', 
                          'ma5', 'ma20', 'ma60', 
                          'macd', 'signal', 'rsi', 'volatility', 'vol_change']
        
    def __len__(self):
        return len(self.dates)
    
    def __getitem__(self, idx):
        end_date = self.dates[idx]
        start_date = self.etf_df.index[idx]
        
        # 技术特征序列
        tech_data = self.etf_df.loc[start_date:end_date][self.tech_cols]
        tech_tensor = torch.tensor(tech_data.values, dtype=torch.float32)
        
        # 新闻数据序列
        all_input_ids = []
        all_attention_mask = []
        all_news_weights = []
        for single_date in self.etf_df.loc[start_date:end_date]['date'].to_list():
            date_str = single_date.strftime('%Y-%m-%d')
            # 获取当天的新闻列表
            news = self.news_dict.get(date_str)
            all_input_ids.append(news['input_ids'])
            all_attention_mask.append(news['attention_mask'])
            all_news_weights.append(news['news_weights'])
        
        # 堆叠为三维张量 [days=90, news=20, seq_len]
        input_ids = torch.stack(all_input_ids)
        attention_mask = torch.stack(all_attention_mask)
        # [days=90, news=20]
        news_weights = torch.stack(all_news_weights)
        
        # 目标值 (未来5日涨跌幅， 假设目标列为target_1, target_2, ..., target_5)
        targets = self.etf_df.loc[end_date, [f'target_{i+1}' for i in range(self.pred_days)]]
        target_tensor = torch.tensor(targets.to_list(), dtype=torch.float32)
        
        return {
            'tech_data': tech_tensor,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'news_weights': news_weights,
            'targets': target_tensor
        }