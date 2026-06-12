#!/usr/bin/env python
"""Subsample artifact đã tiền xử lý theo USER — tạo bản nhẹ, công bằng cho mọi baseline.

    python scripts/subsample_dataset.py --src data --dst data_small --user-frac 0.25

Nguyên tắc thiết kế (để so sánh công bằng & không làm yếu mô hình một cách thiên lệch):
  * Sample USER uniform (seed cố định) — giữ nguyên tỉ lệ warm/cold user; mọi
    model (BPATMP + baseline) cùng đọc một thư mục output nên thấy đúng cùng
    user/cạnh/split/mask.
  * Giữ NGUYÊN vocab item + cạnh cấu trúc + metadata + candidate set: full-ranking
    vẫn trên cùng 100,775 item -> độ khó xếp hạng không đổi giữa các bản dữ liệu,
    không phải remap item, ground-truth item idx giữ nguyên.
  * User được giữ thì giữ TOÀN BỘ lịch sử (không sample cạnh) — không phá chuỗi
    hành vi theo thời gian, không thiên vị model dùng/không dùng lịch sử.
  * Cutoff thời gian, protocol mask (purchase-only / seen-all), cold mask đều
    được lọc theo user — KHÔNG đổi định nghĩa.

User idx được remap compact [0, n_keep) (bảng embedding user nhỏ theo tỉ lệ).
Item/category/brand idx giữ nguyên. Layout output y hệt data/ gốc nên
run_training.py / evaluate.py chỉ cần trỏ data_dir sang thư mục mới.
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("subsample")

BEHAVIORS = ("view", "cart", "purchase")
WINDOWS = ("train", "trainval")
MASK_PKLS = (
    "train_mask.pkl",
    "train_mask_purchase_only.pkl",
    "train_mask_seen_all.pkl",
    "train_mask_test_purchase_only.pkl",
)
COPY_FILES = ("item_is_cold.npy", "candidate_item_idx.npy")
COPY_MAPPINGS = (
    "item2idx.json", "brand2idx.json", "category2idx.json", "behavior2idx.json",
    "product_category.parquet", "product_brand.parquet",
)


def _view_keep_mask(src: np.ndarray, dst: np.ndarray, ts: np.ndarray,
                    frac: float, seed: int) -> np.ndarray:
    """Quyết định giữ/bỏ một cạnh view theo hash xác định của (src, dst, ts).

    Cùng một cạnh -> cùng quyết định ở MỌI file (train npy, trainval npy,
    train_events parquet) nên train ⊆ trainval và npy ↔ parquet nhất quán
    by-construction, không cần align thủ công.
    """
    h = (
        src.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)
        ^ dst.astype(np.uint64) * np.uint64(0xC2B2AE3D27D4EB4F)
        ^ ts.astype(np.uint64) * np.uint64(0x165667B19E3779F9)
        ^ np.uint64((seed * 0xD6E8FEB86659FD93) % 2**64)
    )
    h ^= h >> np.uint64(33)
    h *= np.uint64(0xFF51AFD7ED558CCD)
    h ^= h >> np.uint64(33)
    return (h.astype(np.float64) / float(2**64)) < frac


def _remap_user_pkl(path_in: Path, path_out: Path, keep_mask: np.ndarray, old2new: np.ndarray) -> int:
    with open(path_in, "rb") as f:
        d = pickle.load(f)
    out = {
        int(old2new[int(u)]): list(items)
        for u, items in d.items()
        if keep_mask[int(u)]
    }
    with open(path_out, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    return len(out)


def _filter_eval_npy(src: Path, dst: Path, split: str, keep_mask: np.ndarray, old2new: np.ndarray) -> int:
    u = np.load(src / f"{split}_user_idx.npy")
    p = np.load(src / f"{split}_product_idx.npy")
    t = np.load(src / f"{split}_timestamp.npy")
    m = keep_mask[u]
    np.save(dst / f"{split}_user_idx.npy", old2new[u[m]])
    np.save(dst / f"{split}_product_idx.npy", p[m])
    np.save(dst / f"{split}_timestamp.npy", t[m])
    return int(m.sum())


def _filter_gt_parquet(path_in: Path, path_out: Path, keep_mask: np.ndarray, old2new: np.ndarray) -> None:
    import pandas as pd

    df = pd.read_parquet(path_in)
    df = df[keep_mask[df["user_idx"].to_numpy()]].copy()
    df["user_idx"] = old2new[df["user_idx"].to_numpy()]
    df.to_parquet(path_out, index=False)


def _filter_train_events(path_in: Path, path_out: Path, keep_mask: np.ndarray, old2new: np.ndarray,
                         view_frac: float, view_seed: int) -> int:
    """Lọc directory parquet Spark theo fragment để RAM bounded; ghi 1 file duy nhất."""
    import pyarrow as pa
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    dataset = ds.dataset(path_in)
    writer: pq.ParquetWriter | None = None
    n_rows = 0
    try:
        for frag in dataset.get_fragments():
            tbl = frag.to_table()
            uid = tbl.column("user_idx").to_numpy()
            m = keep_mask[uid]
            if view_frac < 1.0:
                is_view = np.asarray(tbl.column("behavior").to_pandas() == "view")
                vk = _view_keep_mask(
                    uid, tbl.column("item_idx").to_numpy(),
                    tbl.column("timestamp").to_numpy(), view_frac, view_seed,
                )
                m = m & (~is_view | vk)
            if not m.any():
                continue
            tbl = tbl.filter(pa.array(m))
            new_uid = old2new[tbl.column("user_idx").to_numpy()]
            tbl = tbl.set_column(
                tbl.schema.get_field_index("user_idx"), "user_idx",
                pa.array(new_uid, type=pa.int64()),
            )
            if writer is None:
                writer = pq.ParquetWriter(path_out, tbl.schema)
            writer.write_table(tbl)
            n_rows += tbl.num_rows
    finally:
        if writer is not None:
            writer.close()
    return n_rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default="data")
    p.add_argument("--dst", default="data_small")
    p.add_argument("--user-frac", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max-view-train", type=int, default=0,
        help="Trần số cạnh view trong cửa sổ train SAU khi lọc user (0 = không cắt). "
             "Cắt thêm bằng hash(src,dst,ts) nên train/trainval/parquet nhất quán; "
             "cart/purchase/GT/mask không bị động tới.",
    )
    args = p.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if not (0 < args.user_frac < 1):
        raise ValueError("--user-frac phải trong (0, 1)")
    if dst.exists() and any(dst.iterdir()):
        raise FileExistsError(f"{dst} đã tồn tại và không rỗng — xoá trước khi chạy lại.")
    (dst / "node_mappings").mkdir(parents=True, exist_ok=True)
    (dst / "graph").mkdir(exist_ok=True)

    with open(src / "node_counts.json") as f:
        counts = json.load(f)
    n_users = counts["user"]

    rng = np.random.default_rng(args.seed)
    n_keep = int(round(n_users * args.user_frac))
    keep_ids = np.sort(rng.permutation(n_users)[:n_keep])
    keep_mask = np.zeros(n_users, dtype=bool)
    keep_mask[keep_ids] = True
    old2new = np.full(n_users, -1, dtype=np.int64)
    old2new[keep_ids] = np.arange(n_keep, dtype=np.int64)
    logger.info("Giữ %d/%d user (%.1f%%), seed=%d — item giữ nguyên %d",
                n_keep, n_users, 100 * args.user_frac, args.seed, counts["product"])

    # Tỉ lệ giữ view (tính trên cửa sổ train SAU lọc user; hash trên idx GỐC
    # để npy và parquet ra cùng quyết định)
    view_frac = 1.0
    if args.max_view_train > 0:
        vs = np.load(src / "view_train_src.npy")
        n_view_kept = int(keep_mask[vs].sum())
        del vs
        view_frac = min(1.0, args.max_view_train / max(n_view_kept, 1))
        logger.info("Cắt view: %s -> mục tiêu %s (frac=%.4f, hash seed=%d)",
                    f"{n_view_kept:,}", f"{args.max_view_train:,}", view_frac, args.seed)

    # 1. Cạnh hành vi (train + trainval)
    for win in WINDOWS:
        for beh in BEHAVIORS:
            s = np.load(src / f"{beh}_{win}_src.npy")
            d = np.load(src / f"{beh}_{win}_dst.npy")
            t = np.load(src / f"{beh}_{win}_ts.npy")
            m = keep_mask[s]
            if beh == "view" and view_frac < 1.0:
                m = m & _view_keep_mask(s, d, t, view_frac, args.seed)
            np.save(dst / f"{beh}_{win}_src.npy", old2new[s[m]])
            np.save(dst / f"{beh}_{win}_dst.npy", d[m])
            np.save(dst / f"{beh}_{win}_ts.npy", t[m])
            logger.info("%s_%s: %s -> %s cạnh", beh, win, f"{len(s):,}", f"{int(m.sum()):,}")

    # 2. Ground truth + eval npy
    for split in ("val", "test"):
        kept_rows = _filter_eval_npy(src, dst, split, keep_mask, old2new)
        n_gt = _remap_user_pkl(
            src / f"{split}_ground_truth.pkl", dst / f"{split}_ground_truth.pkl",
            keep_mask, old2new,
        )
        logger.info("%s: %s dòng GT, %s eval user", split, f"{kept_rows:,}", f"{n_gt:,}")

    # 3. Masks
    for name in MASK_PKLS:
        if (src / name).exists():
            n = _remap_user_pkl(src / name, dst / name, keep_mask, old2new)
            logger.info("%s: %s user", name, f"{n:,}")

    # 4. Cold mask: TÍNH LẠI từ cạnh đã giữ (định nghĩa = không có cạnh hành vi
    # train nào). Lọc mask gốc là sai khi có --max-view-train: user/item chỉ có
    # view mà view bị cắt hết sẽ thành cold trong bản nhỏ.
    user_cold = np.ones(n_keep, dtype=bool)
    item_cold = np.ones(counts["product"], dtype=bool)
    for beh in BEHAVIORS:
        user_cold[np.load(dst / f"{beh}_train_src.npy")] = False
        item_cold[np.load(dst / f"{beh}_train_dst.npy")] = False
    np.save(dst / "user_is_cold.npy", user_cold)
    np.save(dst / "item_is_cold.npy", item_cold)
    logger.info("Cold mask (tính lại): cold_user=%s/%s, cold_item=%s/%s",
                f"{int(user_cold.sum()):,}", f"{n_keep:,}",
                f"{int(item_cold.sum()):,}", f"{counts['product']:,}")
    for name in COPY_FILES:
        if name != "item_is_cold.npy" and (src / name).exists():
            shutil.copy2(src / name, dst / name)

    # 5. node_counts + node_mappings
    counts["user"] = n_keep
    with open(dst / "node_counts.json", "w") as f:
        json.dump(counts, f, indent=2)
    with open(src / "node_mappings" / "user2idx.json") as f:
        user2idx = json.load(f)
    user2idx_small = {k: int(old2new[v]) for k, v in user2idx.items() if keep_mask[v]}
    with open(dst / "node_mappings" / "user2idx.json", "w") as f:
        json.dump(user2idx_small, f)
    for name in COPY_MAPPINGS:
        shutil.copy2(src / "node_mappings" / name, dst / "node_mappings" / name)

    # 6. graph/ parquet (event log + GT + metadata) — cho baseline dùng parquet
    n_ev = _filter_train_events(
        src / "graph" / "train_events.parquet", dst / "graph" / "train_events.parquet",
        keep_mask, old2new, view_frac, args.seed,
    )
    logger.info("graph/train_events: %s dòng", f"{n_ev:,}")
    for split in ("val", "test"):
        _filter_gt_parquet(
            src / "graph" / f"{split}_ground_truth.parquet",
            dst / "graph" / f"{split}_ground_truth.parquet",
            keep_mask, old2new,
        )
    shutil.copy2(src / "graph" / "item_metadata.parquet", dst / "graph" / "item_metadata.parquet")

    # 7. Manifest để truy vết bản nhỏ sinh ra từ đâu
    with open(dst / "SUBSAMPLE_MANIFEST.json", "w") as f:
        json.dump({
            "source": str(src.resolve()),
            "user_frac": args.user_frac,
            "seed": args.seed,
            "n_users": n_keep,
            "n_users_source": n_users,
            "items_kept": "all (candidate set & full-ranking khó như bản gốc)",
            "max_view_train": args.max_view_train,
            "view_frac": view_frac,
            "note": "User sample uniform, giữ toàn bộ lịch sử user được chọn; "
                    "cutoff/protocol/mask không đổi. Mọi baseline phải đọc thư mục này.",
        }, f, indent=2, ensure_ascii=False)

    # 8. Verify
    logger.info("Verify bản nhỏ ...")
    for win in WINDOWS:
        for beh in BEHAVIORS:
            s = np.load(dst / f"{beh}_{win}_src.npy")
            d = np.load(dst / f"{beh}_{win}_dst.npy")
            assert s.size == 0 or (s.min() >= 0 and s.max() < n_keep), f"{beh}_{win} src OOB"
            assert d.size == 0 or (d.min() >= 0 and d.max() < counts["product"]), f"{beh}_{win} dst OOB"
    for split in ("val", "test"):
        with open(dst / f"{split}_ground_truth.pkl", "rb") as f:
            gt = pickle.load(f)
        u = np.load(dst / f"{split}_user_idx.npy")
        assert set(gt.keys()) == set(np.unique(u).tolist()), f"{split}: GT keys != eval npy users"
        assert all(0 <= min(v) and max(v) < counts["product"] for v in gt.values() if v)
    with open(dst / "val_ground_truth.pkl", "rb") as f:
        val_gt = pickle.load(f)
    with open(dst / "train_mask_purchase_only.pkl", "rb") as f:
        mask = pickle.load(f)
    leaked = sum(
        1 for u, items in val_gt.items() if set(items) & set(mask.get(u, []))
    )
    assert leaked == 0, f"{leaked} val user có positive nằm trong primary mask"
    uc = np.load(dst / "user_is_cold.npy")
    assert uc.shape[0] == n_keep
    if view_frac < 1.0:
        # view_train phải ⊆ view_trainval (cùng quyết định hash); so bằng set (s,d,t)
        tr = {(int(a), int(b), int(c)) for a, b, c in zip(
            np.load(dst / "view_train_src.npy")[:200000],
            np.load(dst / "view_train_dst.npy")[:200000],
            np.load(dst / "view_train_ts.npy")[:200000],
        )}
        tv_s = np.load(dst / "view_trainval_src.npy")
        tv = {(int(a), int(b), int(c)) for a, b, c in zip(
            tv_s, np.load(dst / "view_trainval_dst.npy"), np.load(dst / "view_trainval_ts.npy"),
        )}
        missing = len(tr - tv)
        assert missing == 0, f"{missing} cạnh view_train không có trong view_trainval"
        # parquet phải khớp số view với npy train
        import pyarrow.dataset as ds_mod
        import pyarrow.compute as pc
        dset = ds_mod.dataset(dst / "graph" / "train_events.parquet")
        n_view_pq = dset.count_rows(filter=pc.field("behavior") == "view")
        n_view_npy = int(np.load(dst / "view_train_src.npy").shape[0])
        assert n_view_pq == n_view_npy, f"parquet view={n_view_pq} != npy view_train={n_view_npy}"
        logger.info("Verify view-cut OK: train⊆trainval, parquet=%s view", f"{n_view_pq:,}")
    logger.info("Verify OK. Hoàn tất: %s", dst)


if __name__ == "__main__":
    main()
