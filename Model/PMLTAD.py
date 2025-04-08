import torch
import torch.nn as nn
import torch.nn.functional as F

class PMLTAD(nn.Module):
    def __init__(self, model_param):
        super(PMLTAD, self).__init__()
        self.day = model_param["day"]
        self.future_day = model_param["future_day"]
        self.device = model_param["device"]
        self.embedding_size = model_param["embedding_size"]
        self.week_embedding_size = model_param["week_embedding_size"]
        self.a_feat_size = model_param["a_feat_size"]
        self.u_feat_size = model_param["u_feat_size"]
        self.hidden_size = model_param["hidden_size"]
        self.num_heads = model_param["num_attention_head"]
        self.bce_weight = model_param["bce_weight"]
        self.recon_weight = model_param["recon_weight"]

        # Embedding layers
        self.a_embedding = nn.Embedding(self.a_feat_size, self.embedding_size)
        self.u_embedding = nn.Embedding(self.u_feat_size, self.embedding_size)
        self.week_embedding = nn.Embedding(7, self.week_embedding_size)
        self.day_embedding = nn.Embedding(31, self.week_embedding_size)

        # 用户特征提取模块 - 自注意力
        self.self_attn_behavior = nn.MultiheadAttention(self.embedding_size, self.num_heads, batch_first=True)
        self.self_attn_time = nn.MultiheadAttention(self.week_embedding_size, self.num_heads, batch_first=True)
        self.self_attn_user = nn.MultiheadAttention(self.embedding_size, self.num_heads, batch_first=True)

        # 短期行为增强模块 - Transformer 编码器 + 解码器
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_size, nhead=self.num_heads, batch_first=True)
        self.behavior_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        decoder_layer = nn.TransformerDecoderLayer(d_model=self.embedding_size, nhead=self.num_heads, batch_first=True)
        self.behavior_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

        # 周期感知模块：用点乘模拟 A_s；时间衰减 A_t 通过 learnable slope
        self.time_decay_weight = nn.Parameter(torch.rand(1))
        self.attn_projection = nn.Linear(self.week_embedding_size, self.embedding_size)

        # 周期趋势预测
        self.trend_predictor = nn.Linear(self.embedding_size * 3, self.a_feat_size)

        # 活跃天数预测模块
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.fc_days = nn.Linear(self.embedding_size * 3, 1)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        self.reconstruction_loss = nn.MSELoss()
        self.prediction_loss = nn.MSELoss()
        self.trend_loss = nn.BCELoss()

    def forward(self, a_idx, u_idx, week_idx, day_idx, behavior_mask, target_day_count=None, target_behavior_trend=None):
        # Embedding
        a_emb = self.a_embedding(a_idx)  # 行为特征
        u_emb = self.u_embedding(u_idx)  # 用户特征
        week_emb = self.week_embedding(week_idx)
        day_emb = self.day_embedding(day_idx)
        time_emb = week_emb + day_emb  # 周期性时间嵌入

        # Step1: 自注意力建模
        beh_feat, _ = self.self_attn_behavior(a_emb, a_emb, a_emb)
        time_feat, _ = self.self_attn_time(time_emb, time_emb, time_emb)
        user_feat, _ = self.self_attn_user(u_emb, u_emb, u_emb)

        # Step2: 短期行为增强
        enc_beh = self.behavior_encoder(beh_feat)
        dec_beh = self.behavior_decoder(enc_beh, enc_beh)
        loss_rec = self.reconstruction_loss(dec_beh, beh_feat)

        # Step3: 周期感知注意力
        sim = F.cosine_similarity(time_feat[:, :self.day], time_feat[:, self.day:], dim=-1).unsqueeze(-1)
        decay = torch.exp(-self.time_decay_weight * torch.abs(torch.arange(self.day).unsqueeze(0).to(self.device)))
        A = sim + decay  # [batch, future_day, day]
        period_guided = torch.bmm(A, enc_beh[:, :self.day])

        # Step4: 多任务预测
        h_concat = torch.cat([period_guided, time_feat[:, self.day:], user_feat.mean(dim=1, keepdim=True).repeat(1, self.future_day, 1)], dim=-1)
        behavior_pred = self.sigmoid(self.trend_predictor(h_concat))  # 行为概率
        pooled = self.maxpool(h_concat.transpose(1, 2)).squeeze(-1)
        day_count_pred = self.sigmoid(self.fc_days(pooled)).squeeze(-1)

        # 计算损失
        loss = None
        if target_day_count is not None and target_behavior_trend is not None:
            loss_pred = self.prediction_loss(day_count_pred, target_day_count)
            loss_trend = self.trend_loss(behavior_pred, target_behavior_trend)
            loss = loss_pred + self.bce_weight * loss_trend + self.recon_weight * loss_rec

        return (loss, behavior_pred, day_count_pred, dec_beh) if loss is not None else (behavior_pred, day_count_pred, dec_beh)
