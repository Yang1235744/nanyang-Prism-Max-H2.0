"""字节级文本编码器 —— 零依赖，词表 259（UTF-8 字节 + PAD/BOS/EOS）。

完全自训：不加载任何外部 tokenizer / CLIP / T5。
支持中英文混合 prompt，词表覆盖全部 ASCII + 常见 UTF-8 多字节序列。
"""
import torch
import torch.nn as nn

PAD_ID, BOS_ID, EOS_ID = 256, 257, 258
VOCAB_SIZE = 259


def encode_text(text: str, max_len: int = 128) -> list[int]:
    """将文本编码为字节级 token ID 列表。"""
    raw = text.encode("utf-8")[: max_len - 2]
    ids = [BOS_ID] + list(raw) + [EOS_ID]
    ids += [PAD_ID] * (max_len - len(ids))
    return ids


def decode_ids(ids: list[int]) -> str:
    """逆向解码（仅用于调试，生成任务不需要）。"""
    out = []
    for i in ids:
        if i == PAD_ID:
            break
        if i <= 255:
            out.append(bytes([i]))
    try:
        return b"".join(out).decode("utf-8", errors="ignore")
    except Exception:
        return ""


class ByteTextEncoder(nn.Module):
    """轻量 Transformer 文本编码器，输出与 DiT dim 对齐的上下文向量。"""

    def __init__(self, dim: int = 512, depth: int = 4, heads: int = 4, max_len: int = 128):
        super().__init__()
        self.max_len = max_len
        self.tok_emb = nn.Embedding(VOCAB_SIZE, dim, padding_idx=PAD_ID)
        self.pos_emb = nn.Parameter(torch.zeros(max_len, dim))
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.ln = nn.LayerNorm(dim)
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: [B, L] -> [B, L, dim]，padding 位置由 mask 屏蔽。"""
        mask = ids == PAD_ID
        pos = torch.arange(ids.shape[1], device=ids.device)
        h = self.tok_emb(ids) + self.pos_emb[pos]
        return self.ln(self.encoder(h, src_key_padding_mask=mask))
