from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable
from numbers import Integral
from torch import Tensor

EMBED_DIM: int = 128


def configure_dims(embed_dim: int) -> None:
    """Sync EMBED_DIM with config before any validation runs."""
    global EMBED_DIM
    EMBED_DIM = embed_dim


NODE_TYPES: list[str] = ["user", "product", "category", "brand"]
BEHAVIOR_TYPES: list[str] = ["view", "cart", "purchase"]

RELATION_TYPES: list[tuple[str, str, str]] = [
    ("user", "view", "product"),
    ("user", "cart", "product"),
    ("user", "purchase", "product"),
    ("product", "rev_view", "user"),
    ("product", "rev_cart", "user"),
    ("product", "rev_purchase", "user"),
    ("product", "belongs_to", "category"),
    ("category", "contains", "product"),
    ("product", "producedBy", "brand"),
    ("brand", "brands", "product"),
]

BEHAVIOR_TO_ID: dict[str, int] = {"view": 0, "cart": 1, "purchase": 2}


@dataclass
class EvalInput:
    user_embeddings: Tensor  # (num_eval_users, d)
    item_embeddings: Tensor  # (num_items, d)
    eval_user_ids: Tensor  # (num_eval_users,)
    ground_truth: dict[int, int | list[int] | tuple[int, ...] | set[int]]
    exclude_items: dict[int, list[int]]  # {user_id: [train_pos_items]}

    @staticmethod
    def _as_item_list(items: int | Iterable[int]) -> list[int]:
        if isinstance(items, Integral):
            return [int(items)]
        return [int(x) for x in items]

    def validate(self, total_num_users: int | None = None) -> None:
        """Validate EvalInput tensor shapes and index bounds.

        Parameters
        ----------
        total_num_users:
            The authoritative total user count from ``node_counts.json``
            (i.e. ``SplitResult.num_users``).  Pass this to enable a strict
            bounds check on ``eval_user_ids``, which holds *global* user
            indices in ``[0, total_num_users)`` — NOT positions into
            ``user_embeddings`` (which has size ``N_eval_users``).
            When omitted the global-ID bounds check is skipped.
        """
        n_eval_users = self.user_embeddings.size(0)
        num_items = self.item_embeddings.size(0)

        assert self.user_embeddings.size(1) == EMBED_DIM, (
            f"user_embeddings dim1={self.user_embeddings.size(1)}, expected {EMBED_DIM}"
        )
        assert self.item_embeddings.size(1) == EMBED_DIM, (
            f"item_embeddings dim1={self.item_embeddings.size(1)}, expected {EMBED_DIM}"
        )

        # user_embeddings is ordered by POSITION in eval_user_ids, not by
        # global user index.  Check the positional pairing, not a value bound.
        assert n_eval_users == self.eval_user_ids.size(0), (
            f"user_embeddings has {n_eval_users} rows but "
            f"eval_user_ids has {self.eval_user_ids.size(0)} entries — "
            "they must have the same length (embeddings are ordered by eval_user_ids position)"
        )
        assert len(self.ground_truth) == self.eval_user_ids.size(0), (
            f"ground_truth has {len(self.ground_truth)} entries but "
            f"eval_user_ids has {self.eval_user_ids.size(0)}"
        )

        # eval_user_ids holds GLOBAL user indices; check against total vocab size
        # when the authoritative count is provided.
        if total_num_users is not None:
            uid_max = int(self.eval_user_ids.max().item())
            assert uid_max < total_num_users, (
                f"eval_user_ids contains global index {uid_max} >= "
                f"total_num_users {total_num_users}.  "
                "This is a vocabulary mismatch between the saved mapping and "
                "the node counts passed to the model."
            )

        # Item bounds: ground_truth values are global item indices into item_embeddings.
        # The primary protocol uses multi-positive ground truth:
        # {user_idx: [all new purchase item_idx]}.
        missing_users = [
            int(u) for u in self.eval_user_ids.tolist() if int(u) not in self.ground_truth
        ]
        assert not missing_users, (
            f"ground_truth is missing {len(missing_users)} eval users, first={missing_users[:5]}"
        )

        gt_items: list[int] = []
        for user_items in self.ground_truth.values():
            items = self._as_item_list(user_items)
            assert items, "ground_truth contains an eval user with no positive items"
            gt_items.extend(items)

        item_min = min(gt_items)
        item_max = max(gt_items)
        assert item_min >= 0, f"ground_truth contains negative item index {item_min}"
        assert item_max < num_items, (
            f"ground_truth contains item index {item_max} >= "
            f"item_embeddings.size(0) {num_items}.  "
            "This will cause a CUDA OOB during evaluator scoring."
        )
