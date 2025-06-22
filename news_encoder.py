import torch.nn as nn
import torch
from transformers import AutoTokenizer, AutoModel

class NewsEncoder(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.bert = AutoModel.from_pretrained("yiyanghkust/finbert-tone-chinese")
        self.embed_dim = self.bert.config.hidden_size
        
        # 冻结底层参数
        self.bert.requires_grad_(False)
        
        # 新闻聚合层（压缩维度，平衡新闻特征与技术指标特征）
        self.aggregate = nn.Sequential(
            nn.Linear(self.embed_dim, config['news_emb_aggregate_output_size']),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
    
    def forward(self, input_ids, attention_mask, news_weights):
        """
        input_ids: [batch, num_news, seq_len]
        attention_mask: [batch, num_news, seq_len]
        news_weights: [batch, num_news] 每条新闻的预计算权重
        """
        batch_size, num_news, seq_len = input_ids.shape
        
        # 展平处理
        flat_ids = input_ids.view(-1, seq_len) # [batch * num_news, seq_len]
        flat_mask = attention_mask.view(-1, seq_len) # [batch * num_news, seq_len]

        # 确保bert底层的dropout不工作
        self.bert.eval()
        with torch.no_grad():
            # BERT处理
            outputs = self.bert(
                input_ids=flat_ids,
                attention_mask=flat_mask,
                return_dict=True
            )
        
        # 获取[CLS]标记
        cls_vectors = outputs.last_hidden_state[:, 0, :]
        
        # 恢复形状 [batch, num_news, embed_dim]
        cls_vectors = cls_vectors.view(batch_size, num_news, -1)
        
        # 加权平均池化
        # 使用权重作为注意力分数
        weighted = torch.sum(
            cls_vectors * news_weights.unsqueeze(-1), 
            dim=1
        )  # [batch, embed_dim]
        
        return self.aggregate(weighted)  # [batch, news_emb_aggregate_output_size]
