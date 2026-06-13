import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast


def bpr_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
) -> torch.Tensor:
    if pos_scores.dim() == 1:
        pos_scores = pos_scores.unsqueeze(-1)
    if neg_scores.dim() == 1:
        neg_scores = neg_scores.unsqueeze(-1)
    return -F.logsigmoid(pos_scores - neg_scores).mean()


class MultiTaskBPRLoss(nn.Module):
    """Per-behavior BPR with count-aware weights:

        w_b = clip((N_purchase / N_b) ** alpha, w_min, 1.0)

    purchase weight is 1.0 by construction. View/cart get scaled down because
    REES46 has ~100x more views than purchases.
    """

    BEHAVIOR_ORDER = ["view", "cart", "purchase"]

    def __init__(
        self,
        behavior_counts: dict[str, int],
        l2_lambda: float = 0.0,
        alpha: float = 0.5,
        w_min: float = 0.05,
    ):
        super().__init__()
        self.l2_lambda = l2_lambda
        n_purchase = max(int(behavior_counts.get("purchase", 1)), 1)
        weights = []
        for b in self.BEHAVIOR_ORDER:
            n_b = max(int(behavior_counts.get(b, 1)), 1)
            w = (n_purchase / n_b) ** alpha
            w = max(min(w, 1.0), w_min)
            weights.append(w)
        self.register_buffer(
            "task_weights", torch.tensor(weights, dtype=torch.float32)
        )

    @autocast("cuda", enabled=False)
    def forward(
        self,
        behavior_losses: dict[str, torch.Tensor],
        model_params: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        log_dict = {}
        total = torch.tensor(0.0, device=self.task_weights.device)
        n_present = 0

        for idx, beh in enumerate(self.BEHAVIOR_ORDER):
            if beh not in behavior_losses:
                continue
            n_present += 1
            beh_loss = behavior_losses[beh].float()
            w = self.task_weights[idx]
            total = total + w * beh_loss
            log_dict[f"loss/{beh}"] = beh_loss.item()
            log_dict[f"weight/{beh}"] = w.item()

        if n_present > 0:
            total = total * (len(self.BEHAVIOR_ORDER) / n_present)

        if model_params is not None and self.l2_lambda > 0:
            l2_term = self.l2_lambda * model_params.float()
            total = total + l2_term
            log_dict["loss/l2"] = l2_term.item()

        log_dict["loss/total"] = total.item()
        return total, log_dict


class SimpleInfoNCE(nn.Module):
    """Symmetric InfoNCE giua view va purchase embeddings.

    Khong co hard-k mining ben trong — hard negatives da duoc xu ly
    o buoc negative sampling roi.
    """

    def __init__(self, tau: float = 0.1) -> None:
        super().__init__()
        self.tau = tau

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        """z_a, z_b: (N, d) — cung tap users, khac behavior."""
        N = z_a.size(0)
        if N < 2:
            return z_a.new_zeros(())
        z_a = F.normalize(z_a.float(), dim=-1)
        z_b = F.normalize(z_b.float(), dim=-1)
        sim = (z_a @ z_b.T) / self.tau
        labels = torch.arange(N, device=z_a.device)
        return 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels))


class BPATMPLoss(nn.Module):
    """Loss don gian cho BPATMP:

        L = L_BPR  +  lambda_cl * L_InfoNCE(view, purchase)

    AdamW weight_decay xu ly regularization, khong can them L2 rieng.
    """

    def __init__(
        self,
        behavior_counts: dict[str, int],
        lambda_cl: float = 0.1,
        tau: float = 0.1,
        alpha: float = 0.5,
        w_min: float = 0.05,
    ) -> None:
        super().__init__()
        self.lambda_cl = lambda_cl
        self.bpr = MultiTaskBPRLoss(behavior_counts, l2_lambda=0.0, alpha=alpha, w_min=w_min)
        self.cl = SimpleInfoNCE(tau=tau)

    @autocast("cuda", enabled=False)
    def forward(
        self,
        behavior_losses: dict[str, torch.Tensor],
        view_emb: torch.Tensor | None = None,
        purchase_emb: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        total, log_dict = self.bpr(behavior_losses)
        bpr_val = log_dict.pop("loss/total")
        log_dict["loss/bpr"] = bpr_val

        if (
            self.lambda_cl > 0
            and view_emb is not None
            and purchase_emb is not None
            and view_emb.size(0) >= 2
        ):
            l_cl = self.cl(view_emb, purchase_emb)
            total = total + self.lambda_cl * l_cl
            log_dict["loss/cl"] = l_cl.item()
        else:
            log_dict["loss/cl"] = 0.0

        log_dict["loss/total"] = total.item()
        return total, log_dict


def cold_consistency_loss(
    cold_emb: torch.Tensor,
    warm_emb: torch.Tensor,
) -> torch.Tensor:
    """Distillation cosine: keo embedding cold-simulated ve warm (warm da detach o caller).

    Dung 1 - cosine thay vi L2 tho de tranh collapse ve norm 0 (R4).
    cold_emb, warm_emb: (M, d) cung thu tu node.
    """
    if cold_emb.numel() == 0 or cold_emb.size(0) == 0:
        return cold_emb.new_zeros(())
    c = F.normalize(cold_emb.float(), dim=-1)
    w = F.normalize(warm_emb.float(), dim=-1)
    return (1.0 - (c * w).sum(dim=-1)).mean()


def build_user_history_csr(
    triplets: torch.Tensor,
    n_users: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CSR over (user -> seen items) across ALL behaviors so train-time
    negatives can be masked the same way eval's exclude_items does."""
    user = triplets[:, 0].long()
    item = triplets[:, 1].long()
    order = user.argsort()
    user_s = user[order]
    item_s = item[order]
    counts = torch.bincount(user_s, minlength=n_users)
    ptr = torch.zeros(n_users + 1, dtype=torch.long)
    ptr[1:] = counts.cumsum(0)
    return ptr, item_s


def sample_aligned_negatives_local(
    pp_b: torch.Tensor,
    user_b_global: torch.Tensor,
    N_items: int,
    num_neg: int,
    prod_x: torch.Tensor,
    history_ptr: torch.Tensor,
    history_item: torch.Tensor,
    user_emb_b: torch.Tensor,
    item_emb_local: torch.Tensor,
    frac_hard: float = 0.5,
    hard_pool: int = 200,
    skip_top: int = 0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Uniform + hard negatives, voi global history masking.

    Bo popularity sampling (contribution nho, them complexity).
    Distribution: uniform (1-frac_hard) | in-batch hard (frac_hard).
    Hard negative SAMPLE ngau nhien tu pool top-`hard_pool` thay vi lay top-k
    tuyet doi: dinh ranking day dac cac item user SE tuong tac (future positive
    cua val khong the mask bang history) — BPR de chung xuong la pha truc tiep
    NDCG, gay hien tuong val dat dinh som roi tut dan.

    skip_top > 0: BO han `skip_top` rank dau khoi pool — cang train tot, future
    positive cang don vao chinh vung top-k ma NDCG cham diem; sample tu rank
    [skip_top, skip_top + hard_pool) van la negative rat kho (top ~0.5% cua
    catalog) nhung khong de truc tiep cac item dang duoc rank dung.
    """
    device = pp_b.device
    B = pp_b.size(0)
    if N_items <= 1 or B == 0:
        return pp_b.unsqueeze(1).expand(B, num_neg).contiguous()

    n_hard = max(0, int(num_neg * frac_hard))
    n_rand = num_neg - n_hard

    rand_negs = torch.randint(0, N_items, (B, n_rand), device=device, generator=generator)

    if n_hard > 0:
        with torch.no_grad():
            scores = user_emb_b @ item_emb_local.T
            scores.scatter_(1, pp_b.unsqueeze(1), float("-inf"))
            skip = max(0, min(int(skip_top), N_items - 1 - n_hard))
            pool_size = min(max(hard_pool, n_hard), N_items - 1 - skip)
            _, pool = scores.topk(skip + pool_size, dim=-1)
            pool = pool[:, skip:]
            choice = torch.randint(
                0, pool_size, (B, n_hard), device=device, generator=generator
            )
            hard_negs = pool.gather(1, choice)
    else:
        hard_negs = torch.empty((B, 0), dtype=torch.long, device=device)

    negs_local = torch.cat([rand_negs, hard_negs], dim=-1)

    starts = history_ptr[user_b_global]
    ends = history_ptr[user_b_global + 1]
    lens = ends - starts
    max_len = int(lens.max().item()) if lens.numel() > 0 else 0

    if max_len > 0:
        offsets = torch.arange(max_len, device=device)
        pad_idx = (starts.unsqueeze(1) + offsets.unsqueeze(0)).clamp(
            max=history_item.size(0) - 1
        )
        valid = offsets.unsqueeze(0) < lens.unsqueeze(1)
        seen = history_item[pad_idx].masked_fill(~valid, -1)

        for _ in range(2):
            negs_global = prod_x[negs_local.clamp(max=N_items - 1).long()]
            bad = (negs_global.unsqueeze(2) == seen.unsqueeze(1)).any(dim=-1)
            if not bad.any():
                break
            repl = torch.randint(0, N_items, bad.shape, device=device, generator=generator)
            negs_local = torch.where(bad, repl, negs_local)

    same = negs_local == pp_b.unsqueeze(1)
    if same.any():
        repl = torch.randint(0, N_items, negs_local.shape, device=device, generator=generator)
        negs_local = torch.where(same, repl, negs_local)

    return negs_local
