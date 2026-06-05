"""Smoke test cho cold-start: model cold slot + forward(cold_mask) + eval phan nhom.

Khong can Spark/data that — dung HeteroData synthetic nho.
Chay: PYTHONPATH=. python scripts/smoke_cold.py
"""
from __future__ import annotations

import torch
from torch_geometric.data import HeteroData

from src.core.contracts import configure_dims
from src.model.bpatmp import (
    BPATMPModel,
    BEHAVIOR_EDGES,
    STRUCTURAL_EDGES,
    BEHAVIOR_ORIGIN,
)

EMBED = 32
configure_dims(EMBED)

N_USER, N_PROD, N_CAT, N_BRAND = 6, 8, 3, 4
NODE_COUNTS = {"user": N_USER, "product": N_PROD, "category": N_CAT, "brand": N_BRAND}


def build_subgraph(cold_item_idx: int | None = None) -> HeteroData:
    """Subgraph chua tat ca node; product cold_item_idx khong co behavior edge,
    chi co structural edge (category/brand) -> kiem tra content-init."""
    g = HeteroData()
    g["user"].x = torch.arange(N_USER)
    g["product"].x = torch.arange(N_PROD)
    g["category"].x = torch.arange(N_CAT)
    g["brand"].x = torch.arange(N_BRAND)

    torch.manual_seed(0)
    # behavior edges: user-product, tranh cold item
    warm_items = [i for i in range(N_PROD) if i != cold_item_idx]
    for (st, name, dt) in BEHAVIOR_EDGES:
        if name.startswith("rev_"):
            continue
        n_e = 20
        src = torch.randint(0, N_USER, (n_e,))
        dst = torch.tensor([warm_items[int(torch.randint(0, len(warm_items), (1,)))] for _ in range(n_e)])
        ts = torch.randint(1_500_000_000, 1_580_000_000, (n_e,)).long()
        g[(st, name, dt)].edge_index = torch.stack([src, dst])
        g[(st, name, dt)].edge_ts = ts
        # reverse
        rev = ("product", f"rev_{name}", "user")
        g[rev].edge_index = torch.stack([dst, src])
        g[rev].edge_ts = ts

    # structural edges cho MOI product (ke ca cold item)
    prod = torch.arange(N_PROD)
    cat = torch.randint(0, N_CAT, (N_PROD,))
    brand = torch.randint(0, N_BRAND, (N_PROD,))
    g[("product", "belongs_to", "category")].edge_index = torch.stack([prod, cat])
    g[("category", "contains", "product")].edge_index = torch.stack([cat, prod])
    g[("product", "producedBy", "brand")].edge_index = torch.stack([prod, brand])
    g[("brand", "brands", "product")].edge_index = torch.stack([brand, prod])
    # structural edges co the mang edge_attr = behavior origin (purchase=2 mac dinh)
    for et in STRUCTURAL_EDGES:
        ne = g[et].edge_index.size(1)
        g[et].edge_attr = torch.full((ne, 1), BEHAVIOR_ORIGIN["purchase"], dtype=torch.long)
    return g


def main() -> None:
    model = BPATMPModel(
        num_nodes_dict=NODE_COUNTS, embed_dim=EMBED, n_layers=2,
        n_intents=8, rank=8, use_grad_checkpoint=False,
    )
    model.eval()

    # 1) cold slot ton tai, bang embedding = num_nodes+1
    assert model.cold_slot_idx == {"user": N_USER, "product": N_PROD}, model.cold_slot_idx
    assert model.input_proj["user"].weight.shape[0] == N_USER + 1
    assert model.input_proj["product"].weight.shape[0] == N_PROD + 1
    assert model.input_proj["category"].weight.shape[0] == N_CAT
    print("[1] cold slot + bang embedding +1 OK")

    cold_item = 7
    sub = build_subgraph(cold_item_idx=cold_item)
    ref_time = 1_581_000_000.0

    # 2) forward thuong: shape dung
    ue, ie = model(sub, ref_time=ref_time)
    assert ue.shape == (N_USER, EMBED) and ie.shape == (N_PROD, EMBED), (ue.shape, ie.shape)
    assert torch.isfinite(ue).all() and torch.isfinite(ie).all()
    print(f"[2] forward thuong OK — user{tuple(ue.shape)} item{tuple(ie.shape)}")

    # 3) cold item (chi co structural edge) -> embedding huu han, khac 0
    cold_emb = ie[cold_item]
    assert torch.isfinite(cold_emb).all() and cold_emb.abs().sum() > 0, cold_emb
    print(f"[3] cold-item content-init OK — ||emb||={cold_emb.norm():.4f}")

    # 4) forward voi cold_mask: thay self-embedding cua mot so node, shape khong doi
    cm = {
        "user": torch.zeros(N_USER, dtype=torch.bool),
        "product": torch.zeros(N_PROD, dtype=torch.bool),
    }
    cm["product"][cold_item] = True
    cm["user"][0] = True
    ue2, ie2 = model(sub, ref_time=ref_time, cold_mask=cm)
    assert ue2.shape == ue.shape and ie2.shape == ie.shape
    assert torch.isfinite(ue2).all() and torch.isfinite(ie2).all()
    # node bi cold-mask phai khac voi forward thuong (vi self-embedding doi)
    assert not torch.allclose(ue2[0], ue[0]), "cold_mask khong thay doi user[0]"
    print("[4] forward(cold_mask) OK — masked node thay doi, shape giu nguyen")

    # 5) cold_mask phai differentiable qua cold slot
    model.train()
    model.zero_grad(set_to_none=True)
    ue3, ie3, _ = model(sub, ref_time=ref_time, return_beh_embs=True, cold_mask=cm)
    loss = ie3[cold_item].sum() + ue3[0].sum()
    loss.backward()
    w = model.input_proj["product"].weight
    slot = model.cold_slot_idx["product"]
    grad_slot = w.grad[slot]
    assert grad_slot is not None and torch.isfinite(grad_slot).all() and grad_slot.abs().sum() > 0, (
        "grad khong toi cold slot product"
    )
    uw = model.input_proj["user"].weight
    assert uw.grad[model.cold_slot_idx["user"]].abs().sum() > 0, "grad khong toi cold slot user"
    print("[5] gradient chay toi cold slot OK")

    # 6) state_dict round-trip
    sd = model.state_dict()
    m2 = BPATMPModel(
        num_nodes_dict=NODE_COUNTS, embed_dim=EMBED, n_layers=2,
        n_intents=8, rank=8, use_grad_checkpoint=False,
    )
    m2.load_state_dict(sd)
    print("[6] state_dict round-trip OK")

    print("\nSMOKE PHASE B: PASSED")


def test_eval() -> None:
    from src.core.contracts import EvalInput
    from src.core.evaluator import TemporalSplitEvaluator

    torch.manual_seed(1)
    n_eval, n_items = 40, 200
    ue = torch.randn(n_eval, EMBED)
    ie = torch.randn(n_items, EMBED)
    eval_user_ids = torch.arange(n_eval)
    gt = {i: [int(torch.randint(0, n_items, (1,))), int(torch.randint(0, n_items, (1,)))] for i in range(n_eval)}
    excl = {i: [int(torch.randint(0, n_items, (1,)))] for i in range(n_eval)}
    ev = EvalInput(user_embeddings=ue, item_embeddings=ie, eval_user_ids=eval_user_ids,
                   ground_truth=gt, exclude_items=excl)

    evaluator = TemporalSplitEvaluator(ks=[1, 5, 10, 20], device="cpu")

    base = evaluator.evaluate(ev)
    seg = torch.zeros(n_eval, dtype=torch.bool)
    seg[::3] = True  # ~1/3 cold users
    item_cold = torch.zeros(n_items, dtype=torch.bool)
    item_cold[::2] = True
    segged = evaluator.evaluate(ev, user_segment=seg, item_is_cold=item_cold)

    # overall khong doi (regression)
    for k in (1, 5, 10, 20):
        assert abs(base[f"HR@{k}"] - segged[f"HR@{k}"]) < 1e-9, (k, base[f"HR@{k}"], segged[f"HR@{k}"])
        assert abs(base[f"NDCG@{k}"] - segged[f"NDCG@{k}"]) < 1e-9
    print("[D1] overall metric khong doi khi them segment OK")

    # warm/cold trung binh co trong so = overall
    n_cold = int(seg.sum()); n_warm = n_eval - n_cold
    for k in (1, 20):
        mix = (segged[f"cold_user/HR@{k}"] * n_cold + segged[f"warm_user/HR@{k}"] * n_warm) / n_eval
        assert abs(mix - segged[f"HR@{k}"]) < 1e-9, (k, mix, segged[f"HR@{k}"])
    assert segged["cold_user/n"] == n_cold and segged["warm_user/n"] == n_warm
    print(f"[D2] warm/cold split dung (cold={n_cold}, warm={n_warm}) OK")

    # cold-item recall hop le
    for k in (1, 5, 10, 20):
        r = segged[f"cold_item/Recall@{k}"]
        assert 0.0 <= r <= 1.0, (k, r)
    # recall khong giam theo k
    assert segged["cold_item/Recall@20"] >= segged["cold_item/Recall@1"] - 1e-9
    print(f"[D3] cold_item Recall@20={segged['cold_item/Recall@20']:.3f} (n_gt={segged['cold_item/n_gt']:.0f}) OK")

    print("\nSMOKE PHASE D: PASSED")


def test_train() -> None:
    """Phase C: chay train_epoch voi cold enabled tren graph synthetic + sampler that."""
    import torch
    from torch.utils.data import DataLoader
    from src.graph.neighbor_sampler import BehaviorAwareNeighborSampler, NeighborSamplerConfig
    from src.training.losses import BPATMPLoss, build_user_history_csr
    from src.training.trainer import (
        InteractionDataset, train_epoch, _make_cold_subgraph,
    )
    from src.training.losses import cold_consistency_loss

    device = torch.device("cpu")
    g = build_subgraph(cold_item_idx=None)  # graph day du, moi item co behavior edge

    # train triplets tu behavior edges
    rows = []
    for (st, name, dt) in [("user", "view", "product"), ("user", "cart", "product"), ("user", "purchase", "product")]:
        ei = g[(st, name, dt)].edge_index
        ts = g[(st, name, dt)].edge_ts
        bid = {"view": 0, "cart": 1, "purchase": 2}[name]
        for j in range(ei.size(1)):
            rows.append([int(ei[0, j]), int(ei[1, j]), bid, int(ts[j])])
    triplets = torch.tensor(rows, dtype=torch.long)

    # C1: _make_cold_subgraph giu node, bot behavior edge, giu structural
    gen = torch.Generator().manual_seed(0)
    n_beh_before = sum(g[et].edge_index.size(1) for et in g.edge_types if et[1] in ("view", "cart", "purchase"))
    n_struct_before = sum(g[et].edge_index.size(1) for et in g.edge_types if et[1] in ("belongs_to", "contains", "producedBy", "brands"))
    sub_cold = _make_cold_subgraph(g, p_hist=0.6, generator=gen)
    assert torch.equal(sub_cold["user"].x, g["user"].x) and torch.equal(sub_cold["product"].x, g["product"].x)
    n_beh_after = sum(sub_cold[et].edge_index.size(1) for et in sub_cold.edge_types if et[1] in ("view", "cart", "purchase"))
    n_struct_after = sum(sub_cold[et].edge_index.size(1) for et in sub_cold.edge_types if et[1] in ("belongs_to", "contains", "producedBy", "brands"))
    assert n_beh_after < n_beh_before, (n_beh_after, n_beh_before)
    assert n_struct_after == n_struct_before
    print(f"[C1] _make_cold_subgraph: behavior {n_beh_before}->{n_beh_after}, structural giu {n_struct_after} OK")

    # C2: cold_consistency_loss: giong nhau->0, gradient chay
    a = torch.randn(5, EMBED, requires_grad=True)
    assert cold_consistency_loss(a, a).item() < 1e-6
    l = cold_consistency_loss(a, torch.randn(5, EMBED))
    l.backward()
    assert a.grad is not None and a.grad.abs().sum() > 0
    print("[C2] cold_consistency_loss OK")

    # C3: train_epoch chay duoc voi cold enabled, loss huu han, cold_loss > 0
    model = BPATMPModel(num_nodes_dict=NODE_COUNTS, embed_dim=EMBED, n_layers=2,
                        n_intents=8, rank=8, use_grad_checkpoint=False).to(device)
    sampler = BehaviorAwareNeighborSampler(
        data=g, config=NeighborSamplerConfig(hop1_budget=8, hop2_budget=4), device=device,
    )
    loss_fn = BPATMPLoss(behavior_counts={"view": 60, "cart": 20, "purchase": 20}, lambda_cl=0.1)
    hist_ptr, hist_item = build_user_history_csr(triplets, n_users=N_USER)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    loader = DataLoader(InteractionDataset(triplets), batch_size=16, shuffle=True, drop_last=False)

    out = train_epoch(
        model, sampler, loader, opt, loss_fn, scaler, device,
        num_neg=2, amp=False, use_bf16=False, cl_every_k=1,
        history_ptr=hist_ptr, history_item=hist_item,
        cold_p_id=0.3, cold_p_hist=0.3, cold_lambda=0.2, cold_every_k=1,
    )
    assert all(torch.isfinite(torch.tensor(v)) for v in out.values()), out
    assert out["train/cold_loss"] > 0.0, out
    print(f"[C3] train_epoch cold OK — loss={out['train/loss']:.4f} cold_loss={out['train/cold_loss']:.4f} skipped={out['train/skipped_batches']:.0f}")

    # C4: lambda_cold=0 -> khong co cold step (regression-safe path)
    out0 = train_epoch(
        model, sampler, loader, opt, loss_fn, scaler, device,
        num_neg=2, amp=False, use_bf16=False, cl_every_k=1,
        history_ptr=hist_ptr, history_item=hist_item,
        cold_p_id=0.3, cold_p_hist=0.3, cold_lambda=0.0, cold_every_k=1,
    )
    assert out0["train/cold_loss"] == 0.0, out0
    print("[C4] lambda_cold=0 -> khong chay cold (train nhu cu) OK")

    print("\nSMOKE PHASE C: PASSED")


if __name__ == "__main__":
    main()
    test_eval()
    test_train()
