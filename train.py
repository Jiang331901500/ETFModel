import torch
import pandas as pd
from trainer import Trainer
from data_preprocessor import DataPreprocessor
from etf_dataset import ETFDataset
from transformers import AutoTokenizer
import os

# 全局配置参数
config = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    'etf_code': '510500',

    'data_dir': 'data',
    'sample_sequence_window': 30,
    'max_news_per_day': 10,
    'max_news_length': 386,

    'trainer_config': {
        'batch_size': 3,
        'num_epochs': 2000,
        'patience': 10,
        'grad_clip': 1.0,

        'checkpoint_path': '/root/autodl-tmp/checkpoint/' if os.path.exists('/root/autodl-tmp') else 'checkpoint/',
        'checkpoint_interval': 1,
        'model_save_path': '/root/autodl-tmp/model/' if os.path.exists('/root/autodl-tmp') else 'model/',
    },

    'model_config': {
        'news_emb_aggregate_output_size': 128,
        'fusion_hidden': 256,
        'tech_feature_dim': 12,
        'pred_days': 5,
        'gru_hidden': 256,
        
    }
}

# 主函数
if __name__ == "__main__":
    pd.options.mode.copy_on_write = True
    device = torch.device(config['device'])
    os.environ["TOKENIZERS_PARALLELISM"] = "true"  # 禁用并行处理，消除警告
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone-chinese")
    # 准备数据
    dp = DataPreprocessor(config, tokenizer, from_pkl=True)
    dp.load_data_frame()
    ds = ETFDataset(dp.etf_df, dp.preprocess_news(), tokenizer, config['sample_sequence_window'], config['model_config']['pred_days'])
    # 初始化训练器
    trainer = Trainer(config, device)
    trainer.init_dataloader(ds)
    # 训练模型
    trainer.train()
    # 测试
    trainer.load_trained_model()
    trainer.test()
    