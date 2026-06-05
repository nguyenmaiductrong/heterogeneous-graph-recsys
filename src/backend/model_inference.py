from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = Path("checkpoints/downloaded/epoch_003.pt")


class EmbeddingModel:
    """Phục vụ embedding cho recommender.

    Ưu tiên embedding ENCODER-OUTPUT đã precompute (`<stem>.user_emb.npy` /
    `<stem>.item_emb.npy`, sinh bởi scripts/export_embeddings.py) — đây là biểu diễn
    SAU message passing BPATMP, đúng với cái training tối ưu và evaluate đo, và đã
    content-init cho node ít/không có lịch sử (cold-start).

    Nếu chưa có file precompute, fallback về bảng seed `input_proj.*` kèm cảnh báo:
    biểu diễn này CHƯA qua encoder nên không khớp metric và KHÔNG xử lý cold-start —
    chỉ dùng tạm. Chạy scripts/export_embeddings.py để có embedding đúng.
    """

    def __init__(self, checkpoint_path: Path) -> None:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.checkpoint_path = checkpoint_path
        self.epoch = ckpt.get("epoch")
        self.metrics = ckpt.get("metrics", {})

        stem = Path(checkpoint_path).with_suffix("")
        u_path = Path(f"{stem}.user_emb.npy")
        i_path = Path(f"{stem}.item_emb.npy")
        if u_path.exists() and i_path.exists():
            self.embedding_source = "encoder_output"
            self.user = F.normalize(torch.from_numpy(np.load(u_path)).float(), dim=1)
            self.product = F.normalize(torch.from_numpy(np.load(i_path)).float(), dim=1)
            logger.info(
                "Loaded encoder-output embeddings: user=%s product=%s (cold-start ready).",
                tuple(self.user.shape), tuple(self.product.shape),
            )
        else:
            self.embedding_source = "input_proj_seed"
            state = ckpt["model_state_dict"]
            # input_proj.{user,product} có thêm 1 hàng cold-slot ở cuối (index = num_nodes)
            # — KHÔNG phải node thật, phải cắt bỏ để chỉ mục khớp catalog.
            self.user = F.normalize(state["input_proj.user.weight"][:-1].float(), dim=1)
            self.product = F.normalize(state["input_proj.product.weight"][:-1].float(), dim=1)
            logger.warning(
                "Khong thay %s/%s — fallback bang seed input_proj. Bieu dien NAY chua qua "
                "encoder (khong khop metric, KHONG co cold-start). Chay "
                "scripts/export_embeddings.py --checkpoint %s de sinh embedding dung.",
                u_path.name, i_path.name, checkpoint_path,
            )

    @property
    def n_users(self) -> int:
        return int(self.user.size(0))

    @property
    def n_products(self) -> int:
        return int(self.product.size(0))

    def user_vector(self, user_id: str) -> Optional[torch.Tensor]:
        if not user_id.startswith("u-"):
            return None
        try:
            user_idx = int(user_id.removeprefix("u-"))
        except ValueError:
            return None
        if 0 <= user_idx < self.n_users:
            return self.user[user_idx]
        return None

    def product_vectors(self, product_indices: list[int]) -> torch.Tensor:
        idx = torch.tensor(product_indices, dtype=torch.long)
        return self.product[idx]

    def product_vector(self, product_idx: int) -> torch.Tensor:
        return self.product[int(product_idx)]


@lru_cache(maxsize=1)
def load_embedding_model() -> EmbeddingModel:
    checkpoint = Path(os.environ.get("BPATMP_CHECKPOINT", DEFAULT_CHECKPOINT))
    return EmbeddingModel(checkpoint)


def model_available() -> bool:
    checkpoint = Path(os.environ.get("BPATMP_CHECKPOINT", DEFAULT_CHECKPOINT))
    return checkpoint.exists()
