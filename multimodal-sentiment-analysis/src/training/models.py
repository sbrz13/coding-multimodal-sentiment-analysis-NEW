import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, CLIPVisionModel, CLIPVisionConfig


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        output = x / keep_prob * random_tensor
        return output


class FeatureNoiseInjection(nn.Module):
    def __init__(self, noise_scale=0.1, prob=0.5):
        super().__init__()
        self.noise_scale = noise_scale
        self.prob = prob

    def forward(self, x):
        if not self.training or random.random() > self.prob:
            return x
        noise = torch.randn_like(x) * self.noise_scale
        return x + noise


class TextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(config.model_config["text_model"])
        hidden_size = self.roberta.config.hidden_size
        proj_dim = config.model_config["projection_dim"]
        text_dropout = config.model_config.get("text_dropout_rate", config.model_config["dropout_rate"])

        if config.model_config["freeze_encoders"]:
            num_layers = self.roberta.config.num_hidden_layers
            for i, layer in enumerate(self.roberta.encoder.layer):
                if i < num_layers - 4:
                    for param in layer.parameters():
                        param.requires_grad = False
            for param in self.roberta.embeddings.parameters():
                param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(hidden_size, proj_dim),
            nn.GELU(),
            nn.Dropout(text_dropout),
            nn.Linear(proj_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )

        self.seq_projection = nn.Sequential(
            nn.Linear(hidden_size, proj_dim),
            nn.GELU(),
            nn.LayerNorm(proj_dim),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        sequence_output = outputs.last_hidden_state
        projected = self.projection(cls_embedding)
        seq_projected = self.seq_projection(sequence_output)
        return projected, seq_projected, attention_mask


class ImageEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        vision_model_name = config.model_config["vision_model"]
        self.use_clip = "clip" in vision_model_name.lower()

        if self.use_clip:
            self.vit = CLIPVisionModel.from_pretrained(vision_model_name)
            hidden_size = self.vit.config.hidden_size
        else:
            self.vit = AutoModel.from_pretrained(vision_model_name)
            hidden_size = self.vit.config.hidden_size

        proj_dim = config.model_config["projection_dim"]
        image_dropout = config.model_config.get("image_dropout_rate", config.model_config["dropout_rate"])
        self.feature_noise = FeatureNoiseInjection(
            noise_scale=config.model_config.get("image_noise_scale", 0.1),
            prob=config.model_config.get("image_noise_prob", 0.5),
        )

        if config.model_config["freeze_encoders"]:
            num_layers = self.vit.config.num_hidden_layers
            for i, layer in enumerate(self.vit.encoder.layers):
                if i < num_layers - 4:
                    for param in layer.parameters():
                        param.requires_grad = False
            for param in self.vit.embeddings.parameters():
                param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(hidden_size, proj_dim),
            nn.GELU(),
            nn.Dropout(image_dropout),
            nn.Linear(proj_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )

        self.seq_projection = nn.Sequential(
            nn.Linear(hidden_size, proj_dim),
            nn.GELU(),
            nn.LayerNorm(proj_dim),
        )

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        patch_sequence = outputs.last_hidden_state[:, 1:, :]

        cls_embedding = self.feature_noise(cls_embedding)
        patch_sequence = self.feature_noise(patch_sequence)

        projected = self.projection(cls_embedding)
        seq_projected = self.seq_projection(patch_sequence)
        return projected, seq_projected


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, dropout_rate, ffn_ratio=4, drop_path_rate=0.0):
        super().__init__()
        self.t2i_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout_rate, batch_first=True
        )
        self.i2t_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout_rate, batch_first=True
        )

        self.t_gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.i_gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())

        self.t_norm1 = nn.LayerNorm(dim)
        self.i_norm1 = nn.LayerNorm(dim)

        ffn_hidden = dim * ffn_ratio
        self.ffn_t = nn.Sequential(
            nn.Linear(dim, ffn_hidden), nn.GELU(), nn.Dropout(dropout_rate),
            nn.Linear(ffn_hidden, dim), nn.Dropout(dropout_rate),
        )
        self.ffn_i = nn.Sequential(
            nn.Linear(dim, ffn_hidden), nn.GELU(), nn.Dropout(dropout_rate),
            nn.Linear(ffn_hidden, dim), nn.Dropout(dropout_rate),
        )
        self.t_norm2 = nn.LayerNorm(dim)
        self.i_norm2 = nn.LayerNorm(dim)

        self.drop_path_t = DropPath(drop_path_rate)
        self.drop_path_i = DropPath(drop_path_rate)

    def forward(self, text_seq, text_mask, image_seq):
        key_padding_mask = None
        if text_mask is not None:
            key_padding_mask = (text_mask == 0)

        t2i_out, t2i_w = self.t2i_attn(
            query=text_seq, key=image_seq, value=image_seq
        )
        i2t_out, i2t_w = self.i2t_attn(
            query=image_seq, key=text_seq, value=text_seq,
            key_padding_mask=key_padding_mask
        )

        t_gate = self.t_gate(torch.cat([text_seq, t2i_out], dim=-1))
        i_gate = self.i_gate(torch.cat([image_seq, i2t_out], dim=-1))

        text_fused = self.t_norm1(text_seq + self.drop_path_t(t_gate * t2i_out))
        image_fused = self.i_norm1(image_seq + self.drop_path_i(i_gate * i2t_out))

        text_fused = self.t_norm2(text_fused + self.drop_path_t(self.ffn_t(text_fused)))
        image_fused = self.i_norm2(image_fused + self.drop_path_i(self.ffn_i(image_fused)))

        return text_fused, image_fused


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, dropout_rate, ffn_ratio=4, drop_path_rate=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout_rate, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        ffn_hidden = dim * ffn_ratio
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_hidden), nn.GELU(), nn.Dropout(dropout_rate),
            nn.Linear(ffn_hidden, dim), nn.Dropout(dropout_rate),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x, mask=None):
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = (mask == 0)
        out, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.drop_path(out))
        x = self.norm2(x + self.drop_path(self.ffn(x)))
        return x


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        proj_dim = config.model_config["projection_dim"]
        num_heads = config.model_config["num_attention_heads"]
        attn_dim = config.model_config["attention_dim"]
        dropout_rate = config.model_config["dropout_rate"]

        assert attn_dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.W_Q = nn.Linear(proj_dim, attn_dim, bias=False)
        self.W_K = nn.Linear(proj_dim, attn_dim, bias=False)
        self.W_V = nn.Linear(proj_dim, attn_dim, bias=False)
        self.out_proj = nn.Linear(attn_dim, attn_dim, bias=False)
        self.layer_norm = nn.LayerNorm(attn_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, text_feat, image_feat):
        B = text_feat.size(0)

        q = self.W_Q(text_feat)
        k = self.W_K(image_feat)
        v = self.W_V(image_feat)

        q = q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(B, -1)
        z = self.out_proj(context)
        z = self.layer_norm(z)
        return z


class ModalityReliabilityGate(nn.Module):
    def __init__(self, text_dim, image_dim, hidden_dim=256):
        super().__init__()
        self.text_reliability = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.image_reliability = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.joint_gate = nn.Sequential(
            nn.Linear(text_dim + image_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, text_feat, image_feat):
        text_score = self.text_reliability(text_feat)
        image_score = self.image_reliability(image_feat)
        scores = torch.cat([text_score, image_score], dim=-1)
        weights = torch.softmax(scores, dim=-1)
        text_weight = weights[:, 0:1]
        image_weight = weights[:, 1:2]
        joint_gate = self.joint_gate(torch.cat([text_feat, image_feat], dim=-1))
        image_weight = image_weight * joint_gate
        text_weight = text_weight * (1 - joint_gate) + text_weight * joint_gate
        total = text_weight + image_weight + 1e-8
        text_weight = text_weight / total
        image_weight = image_weight / total
        return text_weight, image_weight


class SequenceLevelAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        proj_dim = config.model_config["projection_dim"]
        num_heads = config.model_config["num_attention_heads"]
        num_layers = config.model_config.get("cross_attn_layers", 4)
        dropout_rate = config.model_config["dropout_rate"]
        drop_path_rate = config.model_config.get("drop_path_rate", 0.1)

        self.text_self_attn = SelfAttentionBlock(proj_dim, num_heads, dropout_rate, drop_path_rate=drop_path_rate)
        self.image_self_attn = SelfAttentionBlock(proj_dim, num_heads, dropout_rate, drop_path_rate=drop_path_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        self.cross_layers = nn.ModuleList([
            CrossAttentionBlock(proj_dim, num_heads, dropout_rate, drop_path_rate=dpr[i])
            for i in range(num_layers)
        ])

        self.text_pool = nn.Sequential(
            nn.Linear(proj_dim, proj_dim // 2),
            nn.Tanh(),
            nn.Linear(proj_dim // 2, 1),
        )
        self.image_pool = nn.Sequential(
            nn.Linear(proj_dim, proj_dim // 2),
            nn.Tanh(),
            nn.Linear(proj_dim // 2, 1),
        )

        self.gate = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.Sigmoid(),
        )

        self.fusion_proj = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(proj_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, text_sequence, text_mask, image_sequence):
        text_proj = text_sequence
        image_proj = image_sequence

        text_proj = self.text_self_attn(text_proj, text_mask)
        image_proj = self.image_self_attn(image_proj)

        for cross_layer in self.cross_layers:
            text_proj, image_proj = cross_layer(text_proj, text_mask, image_proj)

        text_attn_logits = self.text_pool(text_proj).squeeze(-1)
        image_attn_logits = self.image_pool(image_proj).squeeze(-1)

        if text_mask is not None:
            text_attn_logits = text_attn_logits.masked_fill(text_mask == 0, float('-inf'))

        text_attn = torch.softmax(text_attn_logits, dim=1)
        image_attn = torch.softmax(image_attn_logits, dim=1)

        text_weighted = (text_proj * text_attn.unsqueeze(-1)).sum(dim=1)
        image_weighted = (image_proj * image_attn.unsqueeze(-1)).sum(dim=1)

        gate = self.gate(torch.cat([text_weighted, image_weighted], dim=-1))
        gated = gate * image_weighted + (1 - gate) * text_weighted

        fused = self.fusion_proj(gated)
        fused = self.dropout(fused)
        return fused


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, text_feat, image_feat):
        text_feat = F.normalize(text_feat, dim=-1)
        image_feat = F.normalize(image_feat, dim=-1)

        B = text_feat.size(0)
        logits = torch.matmul(text_feat, image_feat.T) / self.temperature
        labels = torch.arange(B, device=text_feat.device)

        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        return (loss_i2t + loss_t2i) / 2


class JointClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        attn_dim = config.model_config["attention_dim"]
        proj_dim = config.model_config["projection_dim"]
        num_classes = config.model_config["num_classes"]
        dropout_rate = config.model_config["dropout_rate"]

        combined_dim = attn_dim + proj_dim

        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, combined_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(combined_dim, combined_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(combined_dim // 2, combined_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(combined_dim // 4, num_classes),
        )

    def forward(self, z_head, z_seq):
        combined = torch.cat([z_head, z_seq], dim=-1)
        logits = self.classifier(combined)
        return logits


class TextOnlyClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        proj_dim = config.model_config["projection_dim"]
        num_classes = config.model_config["num_classes"]
        dropout_rate = config.model_config.get("text_dropout_rate", config.model_config["dropout_rate"])
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(proj_dim, proj_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(proj_dim // 2, num_classes),
        )

    def forward(self, text_feat):
        logits = self.classifier(text_feat)
        return logits


class ImageOnlyClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        proj_dim = config.model_config["projection_dim"]
        num_classes = config.model_config["num_classes"]
        image_dropout = config.model_config.get("image_dropout_rate", config.model_config["dropout_rate"])
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(image_dropout),
            nn.Linear(proj_dim, proj_dim // 2),
            nn.GELU(),
            nn.Dropout(image_dropout),
            nn.Linear(proj_dim // 2, num_classes),
        )

    def forward(self, image_feat):
        logits = self.classifier(image_feat)
        return logits


class CrossAttentionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.text_encoder = TextEncoder(config)
        self.image_encoder = ImageEncoder(config)
        self.multihead_fusion = MultiHeadCrossAttention(config)
        self.seq_fusion = SequenceLevelAttention(config)
        self.classifier = JointClassifier(config)
        self.contrastive_loss = ContrastiveLoss(
            temperature=config.model_config.get("contrastive_temperature", 0.07)
        )
        self.contrastive_weight = config.model_config.get("contrastive_weight", 0.3)

        proj_dim = config.model_config["projection_dim"]
        self.use_reliability_gate = config.model_config.get("use_reliability_gate", True)
        if self.use_reliability_gate:
            self.reliability_gate = ModalityReliabilityGate(
                text_dim=proj_dim, image_dim=proj_dim, hidden_dim=proj_dim // 2
            )

    def forward(self, input_ids, attention_mask, pixel_values):
        text_feat, text_seq, text_mask = self.text_encoder(input_ids, attention_mask)
        image_feat, image_seq = self.image_encoder(pixel_values)

        if self.use_reliability_gate:
            text_weight, image_weight = self.reliability_gate(text_feat, image_feat)
            text_feat = text_weight * text_feat
            image_feat = image_weight * image_feat
            text_seq = text_weight.unsqueeze(-1) * text_seq
            image_seq = image_weight.unsqueeze(-1) * image_seq

        z_head = self.multihead_fusion(text_feat, image_feat)
        z_seq = self.seq_fusion(text_seq, text_mask, image_seq)

        logits = self.classifier(z_head, z_seq)
        return logits, text_feat, image_feat, z_head

    def compute_loss(self, logits, labels, text_feat, image_feat, criterion):
        ce_loss = criterion(logits, labels)
        cl_loss = self.contrastive_loss(text_feat, image_feat)
        total_loss = ce_loss + self.contrastive_weight * cl_loss
        return total_loss, ce_loss, cl_loss

    def get_probabilities(self, input_ids, attention_mask, pixel_values):
        logits, _, _, _ = self.forward(input_ids, attention_mask, pixel_values)
        probs = F.softmax(logits, dim=-1)
        return probs


class TextOnlyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.text_encoder = TextEncoder(config)
        self.classifier = TextOnlyClassifier(config)

    def forward(self, input_ids, attention_mask):
        text_feat, _, _ = self.text_encoder(input_ids, attention_mask)
        logits = self.classifier(text_feat)
        return logits, text_feat

    def get_probabilities(self, input_ids, attention_mask):
        logits, text_feat = self.forward(input_ids, attention_mask)
        probs = F.softmax(logits, dim=-1)
        return probs


class ImageOnlyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_encoder = ImageEncoder(config)
        self.classifier = ImageOnlyClassifier(config)

    def forward(self, pixel_values):
        image_feat, _ = self.image_encoder(pixel_values)
        logits = self.classifier(image_feat)
        return logits, image_feat

    def get_probabilities(self, pixel_values):
        logits, image_feat = self.forward(pixel_values)
        probs = F.softmax(logits, dim=-1)
        return probs
