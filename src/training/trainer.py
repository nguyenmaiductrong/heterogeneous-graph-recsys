from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm import tqdm

from src.model.bpatmp import BPATMPModel
from src.core.contracts import BEHAVIOR_TYPES, EvalInput
from src.graph.neighbor_sampler import BehaviorAwareNeighborSampler
from src.training.losses import (
    BPATMPLoss,
    bpr_loss,
    build_user_history_csr,
    cold_consistency_loss,
    sample_aligned_negatives_local,
)

_BEHAVIOR_EDGE_NAMES = {"view", "cart", "purchase"}
_REV_BEHAVIOR_EDGE_NAMES = {"rev_view", "rev_cart", "rev_purchase"}


def _make_cold_subgraph(
    subgraph,
    p_hist: float,
    generator: torch.Generator | None,
):
    """Tao ban sao subgraph mo phong cold: xoa toan bo behavior edge cua mot tap
    seed user ngau nhien (p_hist) -> ho thanh few/zero-shot. CHI xoa edge, GIU NGUYEN
    node .x nen searchsorted/CSR/align voi forward warm khong lech (R2). Lam ngoai
    grad-checkpoint nen khong dinh hazard recompute (R3). Structural edge giu nguyen.
    """
    sub = subgraph.clone()
    device = sub["user"].x.device
    n_user = sub["user"].x.size(0)
    if p_hist <= 0 or n_user == 0:
        return sub
    drop_user = torch.rand(n_user, device=device, generator=generator) < p_hist
    if not drop_user.any():
        return sub
    drop_idx = drop_user.nonzero(as_tuple=True)[0]
    for et in list(sub.edge_types):
        _, name, _ = et
        ei = sub[et].edge_index
        if name in _BEHAVIOR_EDGE_NAMES:
            keep = ~torch.isin(ei[0], drop_idx)
        elif name in _REV_BEHAVIOR_EDGE_NAMES:
            keep = ~torch.isin(ei[1], drop_idx)
        else:
            continue
        sub[et].edge_index = ei[:, keep]
        for attr in ("edge_ts", "ts", "edge_attr"):
            if hasattr(sub[et], attr):
                val = getattr(sub[et], attr)
                if val is not None and val.size(0) == keep.size(0):
                    setattr(sub[et], attr, val[keep])
    return sub
from src.core.evaluator import TemporalSplitEvaluator

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Seed Python / NumPy / PyTorch RNGs so a run is reproducible.

    With ``deterministic=True`` also forces deterministic CUDA kernels and
    disables cuDNN autotuning — slower, and a few ops may not support it.
    """
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    logger.info("Seed set to %d (deterministic=%s)", seed, deterministic)


@dataclass
class TrainConfig:
    # Basic training
    epochs: int = 30
    batch_size: int = 512
    lr: float = 3e-4
    weight_decay: float = 1e-3
    l2_lambda: float = 1e-4
    num_neg: int = 1
    max_grad_norm: float = 1.0
    amp: bool = True
    patience: int = 5
    eval_every: int = 1
    eval_batch_size: int = 512
    num_workers: int = 4
    save_dir: str = "checkpoints/rees46"
    seed: int = 42
    deterministic: bool = False

    # W&B
    use_wandb: bool = False
    wandb_project: str = "bpatmp-recsys"
    wandb_entity: str = "nguyenmaiductrong37"
    wandb_run_name: str = "bpatmp-training"
    wandb_artifact_name: str = "bpatmp-checkpoint"
    wandb_save_every: int = 5

    # Loss: L_total = L_BPR + lambda_cl * L_InfoNCE(view, purchase)
    cl_weight: float = 0.1   # lambda_cl
    cl_tau: float = 0.1      # InfoNCE temperature
    bpr_alpha: float = 0.5   # exponent in w_b = (N_p / N_b) ** alpha
    bpr_w_min: float = 0.05  # floor for w_b
    cl_every_k: int = 1      # 1 = every step, K>1 = run CL every K steps
    use_bf16: bool = True
    max_view_triplets: int = -1

    # Cold-robust training (chi kich hoat khi lambda_cold > 0)
    cold_p_id: float = 0.15       # xac suat thay self-embedding seed bang cold slot
    cold_p_hist: float = 0.30     # xac suat "cold-out" mot seed user (xoa behavior edge)
    cold_lambda: float = 0.0      # trong so distillation L_cold (0 = tat, train nhu cu)
    cold_every_k: int = 2         # chay forward cold moi K step

    # Evaluation
    eval_subsample: int = 10000
    eval_seed: int = 42
    eval_ks: list[int] = field(default_factory=lambda: [1, 5, 10, 20, 50])
    primary_metric: str = "NDCG@20"

    # A100 Optimizations
    gradient_accumulation: int = 1
    warmup_epochs: int = 0
    min_lr: float = 1e-6
    use_fused_adamw: bool = True
    compile_model: bool = False
    allow_tf32: bool = True
    cudnn_benchmark: bool = True
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    log_every: int = 50
    empty_cache_freq: int = 0

    @classmethod
    def from_yaml(cls, cfg: dict) -> "TrainConfig":
        t = cfg.get("training", {})
        loss = cfg.get("loss", {})
        w = cfg.get("wandb", {})
        e = cfg.get("evaluation", {})
        a100 = cfg.get("a100", {})
        # `low_history` la ten moi; giu fallback `cold` cho config/checkpoint cu.
        cold = cfg.get("low_history", cfg.get("cold", {}))

        eval_ks = e.get("ks", [1, 5, 10, 20, 50])
        primary_metric = str(e.get("primary_metric", cls.primary_metric))

        return cls(
            # Basic training
            epochs=t.get("epochs", cls.epochs),
            batch_size=t.get("batch_size", cls.batch_size),
            lr=t.get("lr", cls.lr),
            weight_decay=t.get("weight_decay", cls.weight_decay),
            l2_lambda=t.get("l2_lambda", cls.l2_lambda),
            num_neg=t.get("num_neg", cls.num_neg),
            max_grad_norm=t.get("max_grad_norm", cls.max_grad_norm),
            amp=t.get("amp", cls.amp),
            patience=t.get("patience", cls.patience),
            eval_every=t.get("eval_every", cls.eval_every),
            eval_batch_size=t.get("eval_batch_size", cls.eval_batch_size),
            num_workers=t.get("num_workers", cls.num_workers),
            save_dir=t.get("save_dir", cls.save_dir),
            seed=int(t.get("seed", cls.seed)),
            deterministic=bool(t.get("deterministic", cls.deterministic)),

            # W&B
            use_wandb=w.get("enabled", cls.use_wandb),
            wandb_project=w.get("project", cls.wandb_project),
            wandb_entity=w.get("entity", cls.wandb_entity),
            wandb_run_name=w.get("run_name", cls.wandb_run_name),
            wandb_artifact_name=w.get("artifact_name", cls.wandb_artifact_name),
            wandb_save_every=w.get("save_every", cls.wandb_save_every),

            # Loss
            cl_weight=loss.get("lambda_cl", t.get("cl_weight", cls.cl_weight)),
            cl_tau=float(loss.get("tau", cls.cl_tau)),
            bpr_alpha=loss.get("alpha", cls.bpr_alpha),
            bpr_w_min=loss.get("w_min", cls.bpr_w_min),
            cl_every_k=int(t.get("cl_every_k", a100.get("cl_every_k", cls.cl_every_k))),
            use_bf16=t.get("use_bf16", cls.use_bf16),
            max_view_triplets=t.get("max_view_triplets", cls.max_view_triplets),

            # Cold-robust training
            cold_p_id=float(cold.get("p_id", cls.cold_p_id)),
            cold_p_hist=float(cold.get("p_hist", cls.cold_p_hist)),
            cold_lambda=float(cold.get("lambda_cold", cls.cold_lambda)),
            cold_every_k=int(cold.get("cold_every_k", cls.cold_every_k)),

            # Evaluation
            eval_subsample=t.get("eval_subsample", cls.eval_subsample),
            eval_seed=t.get("eval_seed", cls.eval_seed),
            eval_ks=list(eval_ks),
            primary_metric=primary_metric,

            # A100 Optimizations
            gradient_accumulation=t.get("gradient_accumulation", cls.gradient_accumulation),
            warmup_epochs=t.get("warmup_epochs", cls.warmup_epochs),
            min_lr=t.get("min_lr", cls.min_lr),
            use_fused_adamw=a100.get("use_fused_adamw", t.get("optimizer", "") == "adamw_fused"),
            compile_model=a100.get("compile_model", cls.compile_model),
            allow_tf32=a100.get("allow_tf32", cls.allow_tf32),
            cudnn_benchmark=a100.get("cudnn_benchmark", cls.cudnn_benchmark),
            pin_memory=t.get("pin_memory", cls.pin_memory),
            persistent_workers=t.get("persistent_workers", cls.persistent_workers),
            prefetch_factor=t.get("prefetch_factor", cls.prefetch_factor),
            log_every=w.get("log_every", cls.log_every),
            empty_cache_freq=a100.get("empty_cache_freq", cls.empty_cache_freq),
        )


def _find_latest_checkpoint(save_dir: Path) -> Path | None:
    ckpts = sorted(save_dir.glob("epoch_*.pt"))
    return ckpts[-1] if ckpts else None


def _save_checkpoint(
    save_dir: Path,
    epoch: int,
    model: BPATMPModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    loss: float,
    metrics: dict,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "loss": loss,
            "metrics": metrics,
        },
        save_dir / f"epoch_{epoch:03d}.pt",
    )


def _load_checkpoint(
    ckpt_path: Path,
    model: BPATMPModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> int:
    ckpt = torch.load(ckpt_path, map_location=device)
    try:
        model.load_state_dict(ckpt["model_state_dict"])
    except RuntimeError as exc:
        # R5: cold-slot doi kich thuoc bang embedding (+1) -> checkpoint cu khong khop.
        logger.warning(
            "Checkpoint %s khong tuong thich model hien tai (%s). Bo qua resume, train tu dau.",
            ckpt_path, exc,
        )
        return 0
    try:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    except (ValueError, KeyError, RuntimeError):
        logger.warning(
            "Optimizer/scaler state incompatible with checkpoint — starting with fresh optimizer state."
        )
    resumed_epoch = int(ckpt["epoch"])
    logger.info(
        "Resumed from %s (epoch %d, loss=%.4f)",
        ckpt_path,
        resumed_epoch,
        ckpt.get("loss", float("nan")),
    )
    return resumed_epoch + 1


class InteractionDataset(Dataset):
    def __init__(self, triplets: torch.Tensor) -> None:
        assert triplets.ndim == 2 and triplets.size(1) in (3, 4)
        self.triplets = triplets

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.triplets[idx]


class TemporalBatchSampler(Sampler):
    """Chia triplet DA SORT theo ts thanh cac chunk lien tuc; moi epoch xao tron
    THU TU chunk (noi dung chunk giu nguyen lat thoi gian).

    Voi i.i.d. shuffle, min(batch_ts) ~ dau train window nen filter nhan qua
    `t_e < ref_time` trong model drop ~100% behavior edge luc train (eval lai giu
    het -> train/eval mismatch). Batch theo lat thoi gian giu ref_time sat thoi
    diem cua batch: lich su TRUOC lat van di qua filter, leakage-free khong doi.
    """

    def __init__(
        self,
        n: int,
        batch_size: int,
        generator: torch.Generator | None = None,
        drop_last: bool = True,
    ) -> None:
        self.n = n
        self.batch_size = batch_size
        self.generator = generator
        n_full, rem = divmod(n, batch_size)
        self.n_batches = n_full if (drop_last or rem == 0) else n_full + 1

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self):
        for b in torch.randperm(self.n_batches, generator=self.generator).tolist():
            start = b * self.batch_size
            yield list(range(start, min(start + self.batch_size, self.n)))


def _format_main_metrics(metrics: dict[str, float]) -> str:
    return " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())


def _make_generator(device: torch.device, seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None
    try:
        gen = torch.Generator(device=device)
    except RuntimeError as exc:
        if device.type == "cuda":
            logger.warning(
                "Could not create CUDA sampler generator for deterministic eval (%s); "
                "falling back to the default sampler RNG.",
                exc,
            )
            return None
        gen = torch.Generator()
    gen.manual_seed(int(seed))
    return gen


def train_epoch(
    model: BPATMPModel,
    sampler: BehaviorAwareNeighborSampler,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: BPATMPLoss,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    num_neg: int = 1,
    max_grad_norm: float = 1.0,
    amp: bool = True,
    use_bf16: bool = True,
    cl_every_k: int = 1,
    history_ptr: torch.Tensor | None = None,
    history_item: torch.Tensor | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    generator: torch.Generator | None = None,
    cold_p_id: float = 0.0,
    cold_p_hist: float = 0.0,
    cold_lambda: float = 0.0,
    cold_every_k: int = 2,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_cl_loss = 0.0
    total_cold_loss = 0.0
    n_cold_steps = 0
    n_steps = 0
    cold_on = cold_lambda > 0.0 and (cold_p_id > 0.0 or cold_p_hist > 0.0)
    n_skipped = 0  # batches dropped because no positive items found in subgraph
    purchase_id = BEHAVIOR_TYPES.index("purchase")

    pbar = tqdm(dataloader, desc="train", leave=False, dynamic_ncols=True)
    for step, raw_batch in enumerate(pbar):
        raw_batch = raw_batch.to(device)
        users_g = raw_batch[:, 0]
        items_g = raw_batch[:, 1]
        beh_ids = raw_batch[:, 2]

        # ref_time = batch_min: moi edge giu lai co t_e < min(batch_ts) <= t_pos cho
        # MOI positive trong batch -> khong leakage. Batch la lat thoi gian lien tuc
        # (TemporalBatchSampler) nen batch_min nam sat dau lat: lich su TRUOC lat
        # song sot qua filter `t_e < ref_time` cua model. Positive chi mat context
        # ben trong lat (~1/n_batches cua window) — khong dang ke.
        ref_time = float(raw_batch[:, 3].min().item()) if raw_batch.size(1) >= 4 else None

        unique_users = users_g.unique()
        subgraph = sampler.sample(
            unique_users, seed_type="user", generator=generator
        ).to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        _amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
        with torch.amp.autocast("cuda", dtype=_amp_dtype, enabled=amp and device.type == "cuda"):
            user_emb, item_emb, beh_embs = model(
                subgraph,
                return_beh_embs=True,
                ref_time=ref_time,
            )

            user_x = subgraph["user"].x.contiguous()
            u_loc = torch.searchsorted(user_x, users_g.contiguous())

            prod_x = subgraph["product"].x
            sorted_px, sort_ord = prod_x.sort()
            pos_p = torch.searchsorted(sorted_px, items_g.contiguous()).clamp(
                max=sorted_px.size(0) - 1
            )
            found_p = sorted_px[pos_p] == items_g
            pp_loc = sort_ord[pos_p]

            if not found_p.any():
                n_skipped += 1
                continue

            u_loc = u_loc[found_p]
            pp_loc = pp_loc[found_p]
            bev = beh_ids[found_p]
            users_g_kept = users_g[found_p]

            N_items = item_emb.size(0)
            behavior_losses: dict[str, torch.Tensor] = {}

            for beh_id, beh_name in enumerate(BEHAVIOR_TYPES):
                mask = bev == beh_id
                if not mask.any():
                    continue

                u_b = u_loc[mask]
                pp_b = pp_loc[mask]
                B_b = u_b.size(0)
                if N_items <= 1:
                    continue

                u_emb_b = user_emb[u_b]
                pos_emb_b = item_emb[pp_b]

                if history_ptr is not None and history_item is not None:
                    neg_loc = sample_aligned_negatives_local(
                        pp_b=pp_b,
                        user_b_global=users_g_kept[mask],
                        N_items=N_items,
                        num_neg=num_neg,
                        prod_x=subgraph["product"].x,
                        history_ptr=history_ptr,
                        history_item=history_item,
                        user_emb_b=u_emb_b.detach(),
                        item_emb_local=item_emb.detach(),
                        generator=generator,
                    )
                else:
                    neg_loc = torch.randint(
                        0, N_items - 1, (B_b, num_neg), device=device, generator=generator
                    )
                    neg_loc[neg_loc >= pp_b.unsqueeze(-1)] += 1

                neg_emb_b = item_emb[neg_loc]
                pos_s = (u_emb_b * pos_emb_b).sum(-1, keepdim=True)
                neg_s = torch.bmm(neg_emb_b, u_emb_b.unsqueeze(-1)).squeeze(-1)
                behavior_losses[beh_name] = bpr_loss(pos_s, neg_s)

            users_per_beh = {b: u_loc[bev == bid].unique() for bid, b in enumerate(BEHAVIOR_TYPES)}

            # CL: compare view vs purchase embeddings for users with both behaviors
            run_cl = loss_fn.lambda_cl > 0 and (cl_every_k <= 1 or (step % cl_every_k == 0))
            view_emb_cl: torch.Tensor | None = None
            purch_emb_cl: torch.Tensor | None = None
            if run_cl and beh_embs is not None:
                view_u = users_per_beh.get("view", torch.empty(0, dtype=torch.long, device=device))
                purch_u = users_per_beh.get("purchase", torch.empty(0, dtype=torch.long, device=device))
                common = view_u[torch.isin(view_u, purch_u)]
                if common.numel() >= 2:
                    view_emb_cl = beh_embs["view"][common]
                    purch_emb_cl = beh_embs["purchase"][common]

            run_cold = cold_on and (cold_every_k <= 1 or (step % cold_every_k == 0))

        if not behavior_losses:
            n_skipped += 1
            continue

        loss, log = loss_fn(
            behavior_losses=behavior_losses,
            view_emb=view_emb_cl,
            purchase_emb=purch_emb_cl,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite training loss at step={step}: "
                + ", ".join(f"{k}={v}" for k, v in log.items())
            )

        # Backward warm TRUOC khi forward cold: cold loss distill ve warm emb DA
        # detach nen hai graph doc lap; giu ca hai activation graph cung luc la
        # nguyen nhan OOM. Gradient hai backward tu cong don truoc khi step.
        if amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # ---- Cold-robust: forward phu cold-simulated cho distillation ----
        # Main forward (tren) van WARM -> BPR/CL khong doi (regression-safe).
        # Forward nay mo phong cold (ID-dropout + history-dropout), align theo
        # cung node order voi warm vi chi xoa edge / thay self-embedding.
        if run_cold:
            warm_user_t = user_emb.detach()
            warm_item_t = item_emb.detach()
            del user_emb, item_emb, beh_embs
            cold_mask = None
            if cold_p_id > 0:
                cm_u = torch.rand(
                    subgraph["user"].x.size(0), device=device, generator=generator
                ) < cold_p_id
                cm_p = torch.rand(
                    subgraph["product"].x.size(0), device=device, generator=generator
                ) < cold_p_id
                cold_mask = {"user": cm_u, "product": cm_p}
            sub_cold = _make_cold_subgraph(subgraph, cold_p_hist, generator)
            with torch.amp.autocast(
                "cuda", dtype=_amp_dtype, enabled=amp and device.type == "cuda"
            ):
                ue_cold, ie_cold = model(sub_cold, ref_time=ref_time, cold_mask=cold_mask)
                l_cold = cold_consistency_loss(ue_cold, warm_user_t) + cold_consistency_loss(
                    ie_cold, warm_item_t
                )
            del sub_cold, ue_cold, ie_cold
            if not torch.isfinite(l_cold):
                raise FloatingPointError(f"Non-finite cold loss at step={step}")
            if amp:
                scaler.scale(cold_lambda * l_cold).backward()
            else:
                (cold_lambda * l_cold).backward()
            log["loss/cold"] = float(l_cold.item())
            total_cold_loss += float(l_cold.item())
            n_cold_steps += 1

        if amp:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += log["loss/total"]
        total_cl_loss += float(log.get("loss/cl", 0.0))
        n_steps += 1
        pbar.set_postfix(
            loss=f"{log['loss/total']:.4f}",
            cl=f"{log.get('loss/cl', 0.0):.4f}",
        )

    if n_skipped > 0:
        logger.warning(
            "train_epoch: %d/%d batches skipped (no positive item in subgraph)",
            n_skipped,
            n_skipped + n_steps,
        )

    return {
        "train/loss": total_loss / max(n_steps, 1),
        "train/cl_loss": total_cl_loss / max(n_steps, 1),
        "train/cold_loss": total_cold_loss / max(n_cold_steps, 1),
        "train/skipped_batches": float(n_skipped),
    }


@torch.no_grad()
def export_embeddings(
    model: BPATMPModel,
    sampler: BehaviorAwareNeighborSampler,
    user_ids: torch.Tensor,
    n_items: int,
    device: torch.device,
    batch_size: int = 512,
    use_bf16: bool = True,
    ref_time: float | None = None,
    sampler_seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    d = model.embed_dim
    _amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    sampler_generator = _make_generator(device, sampler_seed)

    item_emb = torch.zeros(n_items, d)
    for start in range(0, n_items, batch_size):
        end = min(start + batch_size, n_items)
        seeds = torch.arange(start, end, device=device)
        sub = sampler.sample(
            seeds,
            seed_type="product",
            generator=sampler_generator,
        ).to(device)
        with torch.amp.autocast("cuda", dtype=_amp_dtype, enabled=device.type == "cuda"):
            _, item_local = model(sub, ref_time=ref_time)
        item_emb[start:end] = item_local.float().cpu()

    user_emb = torch.zeros(len(user_ids), d)
    for start in range(0, len(user_ids), batch_size):
        end = min(start + batch_size, len(user_ids))
        seeds = user_ids[start:end].to(device)
        sub = sampler.sample(
            seeds,
            seed_type="user",
            generator=sampler_generator,
        ).to(device)
        with torch.amp.autocast("cuda", dtype=_amp_dtype, enabled=device.type == "cuda"):
            u_local, _ = model(sub, ref_time=ref_time)
        user_emb[start:end] = u_local.float().cpu()

    return user_emb, item_emb


@torch.no_grad()
def eval_epoch(
    model: BPATMPModel,
    sampler: BehaviorAwareNeighborSampler,
    eval_user_ids: torch.Tensor,
    ground_truth: dict[int, int | list[int]],
    exclude_items: dict[int, list[int]],
    n_items: int,
    evaluator: TemporalSplitEvaluator,
    device: torch.device,
    batch_size: int = 512,
    use_bf16: bool = True,
    subsample: int = 0,
    seed: int = 42,
    sampler_seed: int | None = None,
    ref_time: float | None = None,
    user_is_cold: torch.Tensor | None = None,
    item_is_cold: torch.Tensor | None = None,
) -> dict[str, float]:
    valid_users = list(ground_truth.keys())

    if 0 < subsample < len(valid_users):
        gen = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(valid_users), generator=gen)[:subsample].tolist()
        valid_users = [valid_users[i] for i in idx]
        ground_truth = {u: ground_truth[u] for u in valid_users}
        exclude_items = {u: exclude_items.get(u, []) for u in valid_users}

    eval_user_ids_filtered = torch.tensor(
        valid_users, dtype=torch.long, device=eval_user_ids.device
    )

    user_emb, item_emb = export_embeddings(
        model,
        sampler,
        eval_user_ids_filtered,
        n_items,
        device,
        batch_size,
        use_bf16=use_bf16,
        ref_time=ref_time,
        sampler_seed=seed if sampler_seed is None else sampler_seed,
    )

    eval_input = EvalInput(
        user_embeddings=user_emb,
        item_embeddings=item_emb,
        eval_user_ids=eval_user_ids_filtered,
        ground_truth=ground_truth,
        exclude_items=exclude_items,
    )

    user_segment = None
    if user_is_cold is not None:
        user_segment = user_is_cold.to("cpu")[eval_user_ids_filtered.cpu()].bool()

    return evaluator.evaluate(
        eval_input,
        batch_size=batch_size,
        mode="full_tiled",
        user_segment=user_segment,
        item_is_cold=item_is_cold,
    )


def _setup_a100_optimizations(cfg: TrainConfig, device: torch.device) -> None:
    """Apply A100-specific optimizations."""
    if device.type != "cuda":
        return

    # TF32 for faster matmul on A100/H100
    if cfg.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("TF32 enabled for matmul operations")

    # cuDNN benchmark for faster convolutions (skipped when deterministic)
    if cfg.cudnn_benchmark and not cfg.deterministic:
        torch.backends.cudnn.benchmark = True
        logger.info("cuDNN benchmark enabled")

    # Log GPU info
    gpu_name = torch.cuda.get_device_name(device)
    gpu_mem = torch.cuda.get_device_properties(device).total_memory / 1e9
    logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")


def _get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr: float = 1e-6,
):
    """Cosine schedule with linear warmup."""
    import math

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(
    model: BPATMPModel,
    sampler: BehaviorAwareNeighborSampler,
    train_triplets: torch.Tensor,
    eval_user_ids: torch.Tensor,
    ground_truth: dict[int, int | list[int]],
    exclude_items: dict[int, list[int]],
    n_items: int,
    n_users: int,
    behavior_counts: dict[str, int],
    cfg: TrainConfig,
    device: torch.device,
    eval_ref_time: float | None = None,
    user_is_cold: torch.Tensor | None = None,
    item_is_cold: torch.Tensor | None = None,
) -> None:
    # Reproducibility: re-seed at the start of training so the RNG state here
    # is fixed regardless of how much randomness data/model setup consumed.
    set_seed(cfg.seed, deterministic=cfg.deterministic)

    # Apply A100 optimizations
    _setup_a100_optimizations(cfg, device)

    model.to(device)

    # Compile model with torch.compile (PyTorch 2.0+)
    if cfg.compile_model and hasattr(torch, "compile"):
        # mode="default" + dynamic=True: hetero subgraph có shape thay đổi mỗi step,
        # "reduce-overhead" dùng CUDA graphs nên sẽ recompile liên tục → chậm hơn.
        logger.info("Compiling model with torch.compile (mode=default, dynamic=True)...")
        try:
            from torch import _dynamo as _td
            _td.config.cache_size_limit = 64
        except ImportError:
            pass
        model = torch.compile(model, mode="default", dynamic=True)

    # Generator for sampling inside train_epoch (neighbor + negative sampling);
    # on CUDA falls back to the global RNG if a device generator can't be made.
    train_generator = _make_generator(device, cfg.seed)

    loader_generator = torch.Generator()
    loader_generator.manual_seed(cfg.seed)
    has_ts = train_triplets.size(1) >= 4
    if has_ts:
        train_triplets = train_triplets[train_triplets[:, 3].argsort()]
    dataset = InteractionDataset(train_triplets)
    if has_ts:
        loader = DataLoader(
            dataset,
            batch_sampler=TemporalBatchSampler(
                len(dataset), cfg.batch_size, generator=loader_generator
            ),
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory and device.type == "cuda",
            persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
            prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        )
        logger.info(
            "TemporalBatchSampler: %d time-slice batches of %d (triplets sorted by ts)",
            len(dataset) // cfg.batch_size,
            cfg.batch_size,
        )
    else:
        # Khong co cot ts -> khong the batch theo thoi gian, giu i.i.d. shuffle.
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory and device.type == "cuda",
            drop_last=True,
            persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
            prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
            generator=loader_generator,
        )

    loss_fn = BPATMPLoss(
        behavior_counts=behavior_counts,
        lambda_cl=cfg.cl_weight,
        tau=cfg.cl_tau,
        alpha=cfg.bpr_alpha,
        w_min=cfg.bpr_w_min,
    ).to(device)
    logger.info(
        "Loss: BPR(alpha=%.2f, w_min=%.2f, weights=%s) + lambda_cl=%.3f (tau=%.2f)",
        cfg.bpr_alpha, cfg.bpr_w_min,
        loss_fn.bpr.task_weights.tolist(),
        loss_fn.lambda_cl, loss_fn.cl.tau,
    )

    history_ptr, history_item = build_user_history_csr(train_triplets, n_users=n_users)
    history_ptr = history_ptr.to(device)
    history_item = history_item.to(device)

    emb_params = list(model.input_proj.parameters()) + list(model.beh_proj.parameters())
    emb_ids = {id(p) for p in emb_params}
    other_params = [p for p in model.parameters() if id(p) not in emb_ids]
    use_fused = cfg.use_fused_adamw and device.type == "cuda"
    optimizer = torch.optim.AdamW(
        [
            {"params": other_params, "weight_decay": cfg.weight_decay},
            {"params": emb_params, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
        fused=use_fused,
    )
    logger.info("Optimizer: AdamW (fused=%s, wd=%.1e on non-embedding)", use_fused, cfg.weight_decay)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=cfg.amp and not cfg.use_bf16 and device.type == "cuda"
    )

    steps_per_epoch = max(1, len(loader))
    num_training_steps = cfg.epochs * steps_per_epoch
    num_warmup_steps = max(0, cfg.warmup_epochs) * steps_per_epoch
    scheduler = _get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps, min_lr=cfg.min_lr / max(cfg.lr, 1e-12)
    )
    logger.info(
        "LR scheduler: cosine warmup — warmup_steps=%d, total_steps=%d, peak_lr=%.2e",
        num_warmup_steps,
        num_training_steps,
        cfg.lr,
    )
    evaluator = TemporalSplitEvaluator(ks=list(cfg.eval_ks), device=str(device))

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    wandb_manager = None
    wandb_run = None
    if cfg.use_wandb:
        from src.training.checkpoint_manager import CheckpointManager

        wandb_manager = CheckpointManager(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            run_name=cfg.wandb_run_name,
            artifact_name=cfg.wandb_artifact_name,
            save_every_n_epochs=cfg.wandb_save_every,
            local_dir=str(save_dir),
        )
        wandb_run = wandb_manager.init_wandb(
            config={
                "epochs": cfg.epochs,
                "batch_size": cfg.batch_size,
                "lr": cfg.lr,
                "weight_decay": cfg.weight_decay,
                "l2_lambda": cfg.l2_lambda,
                "num_neg": cfg.num_neg,
                "amp": cfg.amp,
                "cl_weight": cfg.cl_weight,
            }
        )
        logger.info("W&B enabled — project=%s run=%s", cfg.wandb_project, wandb_run.id)

    start_epoch = 0
    if wandb_manager is not None:
        start_epoch = wandb_manager.load_checkpoint(model, optimizer, scaler, device)

    if start_epoch == 0 and wandb_manager is None:
        latest_ckpt = _find_latest_checkpoint(save_dir)
        if latest_ckpt is not None:
            start_epoch = _load_checkpoint(latest_ckpt, model, optimizer, scaler, device)

    pm = cfg.primary_metric
    best_primary = -1.0
    no_improve = 0
    metrics = {}

    epoch_pbar = tqdm(range(start_epoch, cfg.epochs), desc="epochs", dynamic_ncols=True)
    for epoch in epoch_pbar:
        train_log = train_epoch(
            model,
            sampler,
            loader,
            optimizer,
            loss_fn,
            scaler,
            device,
            num_neg=cfg.num_neg,
            max_grad_norm=cfg.max_grad_norm,
            amp=cfg.amp,
            use_bf16=cfg.use_bf16,
            cl_every_k=cfg.cl_every_k,
            history_ptr=history_ptr,
            history_item=history_item,
            scheduler=scheduler,
            generator=train_generator,
            cold_p_id=cfg.cold_p_id,
            cold_p_hist=cfg.cold_p_hist,
            cold_lambda=cfg.cold_lambda,
            cold_every_k=cfg.cold_every_k,
        )
        train_loss = train_log["train/loss"]
        train_log["train/lr"] = float(optimizer.param_groups[0]["lr"])

        row = f"Epoch {epoch:03d} | " + _format_main_metrics(train_log)

        postfix: dict[str, str] = {"loss": f"{train_loss:.4f}"}

        if (epoch + 1) % cfg.eval_every == 0:
            metrics = eval_epoch(
                model,
                sampler,
                eval_user_ids,
                ground_truth,
                exclude_items,
                n_items,
                evaluator,
                device,
                cfg.eval_batch_size,
                use_bf16=cfg.use_bf16,
                subsample=cfg.eval_subsample,
                seed=cfg.eval_seed,
                sampler_seed=cfg.eval_seed,
                ref_time=eval_ref_time,
                user_is_cold=user_is_cold,
                item_is_cold=item_is_cold,
            )

            row += " | " + _format_main_metrics(metrics)

            primary_val = metrics.get(pm, -1.0)
            postfix[pm.replace("@", "_")] = f"{primary_val:.4f}"
            postfix["best_primary"] = f"{max(best_primary, primary_val):.4f}"

            if primary_val > best_primary:
                best_primary = primary_val
                no_improve = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "metrics": metrics,
                    },
                    save_dir / "best.pt",
                )
                row += "  <- best"
            else:
                no_improve += 1

        epoch_pbar.set_postfix(postfix)
        logger.info(row)

        _save_checkpoint(save_dir, epoch, model, optimizer, scaler, train_loss, metrics)

        if wandb_manager is not None:
            wandb_run.log({**train_log, **metrics, "epoch": epoch})
            cloud_ok = wandb_manager.save_checkpoint(
                model,
                optimizer,
                epoch,
                scaler=scaler,
                loss=train_loss,
                metrics=metrics,
            )
            if not cloud_ok:
                logger.error(
                    "Epoch %d: W&B checkpoint NOT verified. "
                    "Local file preserved. DO NOT close Colab yet.",
                    epoch,
                )

        if no_improve >= cfg.patience:
            logger.info(
                "Early stopping at epoch %d. Best %s=%.4f",
                epoch,
                pm,
                best_primary,
            )
            break

    best_path = save_dir / "best.pt"
    if best_path.exists():
        logger.info("Loading best.pt for final FULL-rank evaluation on all val users...")
        best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"])
        final_val_metrics = eval_epoch(
            model,
            sampler,
            eval_user_ids,
            ground_truth,
            exclude_items,
            n_items,
            evaluator,
            device,
            cfg.eval_batch_size,
            use_bf16=cfg.use_bf16,
            subsample=0,
            seed=cfg.eval_seed,
            sampler_seed=cfg.eval_seed,
            ref_time=eval_ref_time,
            user_is_cold=user_is_cold,
            item_is_cold=item_is_cold,
        )
        logger.info(
            "FINAL VAL full-rank eval on best.pt: %s",
            _format_main_metrics(final_val_metrics),
        )
        with open(save_dir / "final_val_metrics.json", "w") as f:
            json.dump(final_val_metrics, f, indent=2)
        if wandb_run is not None:
            wandb_run.log(
                {f"final/val/{k}": v for k, v in final_val_metrics.items()}
            )

    if wandb_run is not None:
        wandb_run.finish()

    logger.info(
        "Training complete. Best %s (subsample)=%.4f. "
        "Run test eval: python scripts/evaluate.py --checkpoint %s --split test",
        pm,
        best_primary,
        save_dir / "best.pt",
    )
