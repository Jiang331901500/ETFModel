import torch.nn as nn
import torch
from news_encoder import NewsEncoder

class ETFModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.pred_days = config['pred_days']
        
        # 新闻编码器
        self.news_encoder = NewsEncoder(config)
        
        # 特征融合层 (技术特征12维 + 新闻特征128维)
        fusion_hidden: int = config['fusion_hidden']
        self.fusion = nn.Sequential(
            nn.Linear(config['tech_feature_dim'] + config['news_emb_aggregate_output_size'], fusion_hidden),
            nn.LayerNorm(fusion_hidden),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 时序建模 (双向GRU)
        gru_hidden: int = config['gru_hidden']
        self.gru = nn.GRU(
            input_size=fusion_hidden,
            hidden_size=gru_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # 下一日涨跌情况的分类头
        self.classification_head = nn.Sequential(
            nn.Linear(gru_hidden*2, 128),
            nn.GELU(),
            nn.Linear(128, 7)  # 7个类别：大跌、跌、微跌、平稳、微涨、涨、大涨
        )
    
    def forward(self, tech_data, input_ids, attention_mask, news_weights):
        """        
        :param tech_data: 技术特征数据 [batch, seq_len, tech_dim]
        :param input_ids: 新闻输入ID [batch, seq_len, num_news, token_len]
        :param attention_mask: 新闻注意力掩码 [batch, seq_len, num_news, token_len]
        :param news_weights: 新闻权重 [batch, seq_len, num_news]
        :return: 预测结果 [batch, pred_days]
        """
        
        # 合并所有天的新闻，批量编码
        # input_ids: [batch, seq_len, num_news, token_len] -> [batch * seq_len, num_news, token_len]
        batch_size, seq_len, num_news, token_len = input_ids.shape
        input_ids_reshaped = input_ids.view(batch_size * seq_len, num_news, token_len)
        attention_mask_reshaped = attention_mask.view(batch_size * seq_len, num_news, token_len)
        news_weights_reshaped = news_weights.view(batch_size * seq_len, num_news)

        # 编码所有天的新闻 [batch * seq_len, num_news, token_len] -> [batch * seq_len, 256]
        news_features = self.news_encoder(
            input_ids=input_ids_reshaped,
            attention_mask=attention_mask_reshaped,
            news_weights=news_weights_reshaped
        )

        # 恢复为 [batch, seq_len, output_dim]
        news_tensor = news_features.view(batch_size, seq_len, -1)
        
        # 融合技术特征和新闻特征
        combined = torch.cat([tech_data, news_tensor], dim=-1)
        fused = self.fusion(combined) # [batch, seq_len, output_dim]
        
        # GRU时序处理
        gru_out, _ = self.gru(fused)
        
        # 取最后一个时间步 [batch, hidden_dim*2]
        last_state = gru_out[:, -1, :]
        
        # 多日预测
        return self.classification_head(last_state)  # [batch, pred_class]