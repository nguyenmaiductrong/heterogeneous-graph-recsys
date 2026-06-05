#!/usr/bin/env python
"""Xuất embedding ENCODER-OUTPUT (sau message passing BPATMP) cho inference.

Đây là biểu diễn ĐÚNG mà training tối ưu và evaluate đo — KHÁC với bảng embedding
seed `input_proj.*` (chưa qua lan truyền). Item/user ít hoặc không có lịch sử hành vi
được content-init qua structural edges (category/brand) ngay trong encoder forward, nên
file xuất ra đã xử lý cold-start. Backend chỉ việc nạp các vector này và chấm điểm.

Chạy:
  python scripts/export_embeddings.py --checkpoint checkpoints/downloaded/epoch_003.pt

Ghi ra cạnh checkpoint:
  <stem>.user_emb.npy   [n_users, d]  float32
  <stem>.item_emb.npy   [n_items, d]  float32
  <stem>.emb_meta.json  {n_users, n_items, dim, ref_time, edge_window, checkpoint}
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/training.yaml")
    parser.add_argument("--checkpoint", required=True, help="Local .pt file")
    parser.add_argument(
        "--edge-window",
        default="auto",
        choices=["auto", "train", "trainval"],
        help="Graph dùng để forward. 'auto' = trainval nếu có (đủ lịch sử nhất), không thì train.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    logger.info("Loaded checkpoint epoch=%s", ckpt.get("epoch"))

    import sys
    sys.path.insert(0, str(Path(__file__).parents[1]))
    from scripts.run_training import build_hetero_data
    from src.model.bpatmp import BPATMPModel
    from src.graph.neighbor_sampler import BehaviorAwareNeighborSampler, NeighborSamplerConfig
    from src.core.contracts import configure_dims
    from src.training.trainer import export_embeddings

    # Chọn cửa sổ graph: ưu tiên trainval (toàn bộ lịch sử quan sát được) cho serving.
    edge_window = args.edge_window
    if edge_window == "auto":
        data_dir = Path(cfg["data"]["data_dir"])
        edge_window = "trainval" if (data_dir / "purchase_trainval_src.npy").exists() else "train"
    logger.info("edge_window=%s", edge_window)

    hetero, node_counts, behavior_data = build_hetero_data(cfg, edge_window=edge_window)
    ref_time = float(
        np.concatenate([behavior_data[b]["ts"] for b in ["view", "cart", "purchase"]]).max()
    )
    logger.info("ref_time=%s n_users=%d n_items=%d", ref_time, node_counts["user"], node_counts["product"])

    mc = cfg["model"]
    configure_dims(mc["embed_dim"])
    model = BPATMPModel(
        num_nodes_dict=node_counts,
        embed_dim=mc["embed_dim"],
        n_layers=mc["n_layers"],
        dropout=mc["dropout"],
        n_intents=mc.get("n_intents", 32),
        rank=mc.get("rank", 32),
        use_grad_checkpoint=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    sc = cfg.get("sampler", {})
    sampler = BehaviorAwareNeighborSampler(
        data=hetero,
        config=NeighborSamplerConfig(
            hop1_budget=sc.get("hop1_budget", 10),
            hop2_budget=sc.get("hop2_budget", 5),
            hop1_sample_replace=sc.get("hop1_sample_replace", True),
        ),
        device=device,
    )

    n_users = int(node_counts["user"])
    n_items = int(node_counts["product"])
    user_emb, item_emb = export_embeddings(
        model,
        sampler,
        torch.arange(n_users, dtype=torch.long),
        n_items,
        device,
        batch_size=cfg["training"].get("eval_batch_size", 2048),
        use_bf16=cfg["training"].get("use_bf16", True),
        ref_time=ref_time,
    )

    stem = ckpt_path.with_suffix("")
    u_path = Path(f"{stem}.user_emb.npy")
    i_path = Path(f"{stem}.item_emb.npy")
    np.save(u_path, user_emb.numpy().astype(np.float32))
    np.save(i_path, item_emb.numpy().astype(np.float32))
    meta = {
        "n_users": n_users,
        "n_items": n_items,
        "dim": int(user_emb.size(1)),
        "ref_time": ref_time,
        "edge_window": edge_window,
        "checkpoint": str(ckpt_path),
        "epoch": ckpt.get("epoch"),
        "source": "encoder_output",
    }
    Path(f"{stem}.emb_meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("Saved encoder embeddings: %s, %s (+ emb_meta.json)", u_path.name, i_path.name)


if __name__ == "__main__":
    main()
