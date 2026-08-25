from __future__ import annotations

import json

import pytest

from rlff.proxy import (
    DETERMINISTIC_AUDIT_REWARD_PROVIDER,
    RLFFGroupAwareAgent,
)
from rlff.rollout_audit import (
    DeterministicAuditRewardProvider,
    _as_rows,
    audit_rollout_batches,
)
from rlff.runtime import role_advantage_tensor_data


class FakeTokenizer:
    eos_token_id = 99

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        _ = skip_special_tokens, clean_up_tokenization_spaces
        return " ".join(str(token_id) for token_id in token_ids)

    def convert_ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        return [f"token-{token_id}" for token_id in token_ids]


class FakeRTensor:
    def __init__(self, data: object) -> None:
        self.data = data

    def to_local(self) -> object:
        return self.data


def test_as_rows_localizes_areal_remote_tensor_handles() -> None:
    torch = pytest.importorskip("torch")

    rows = _as_rows(
        FakeRTensor(torch.tensor([[1, 1, 0], [1, 0, 0]])),
        field="attention_mask",
    )

    assert rows == [[1, 1, 0], [1, 0, 0]]


@pytest.mark.asyncio
async def test_deterministic_audit_reward_uses_trajectory_slot() -> None:
    provider = DeterministicAuditRewardProvider()
    payload = {
        "ids": {
            "trajectory_id": "group:trajectory:3",
            "completion_ids": ["completion-a", "completion-b"],
        }
    }

    assert await provider.score_proxy_completion_role(payload) == {
        "completion-a": 3.0,
        "completion-b": 3.0,
    }
    assert await provider.score_proxy_trajectory_role(payload) == 3.0


def test_deterministic_audit_provider_is_selected_by_serializable_name() -> None:
    workflow_kwargs = {
        "group_size": 4,
        "reward_provider_name": DETERMINISTIC_AUDIT_REWARD_PROVIDER,
    }

    assert json.loads(json.dumps(workflow_kwargs)) == workflow_kwargs
    agent = RLFFGroupAwareAgent(**workflow_kwargs)
    assert isinstance(agent._get_reward_provider(), DeterministicAuditRewardProvider)


def test_audit_validates_real_tensor_shapes_masks_logprobs_and_roles() -> None:
    torch = pytest.importorskip("torch")
    raw = {
        "input_ids": torch.tensor(
            [
                [10, 20, 21, 0],
                [11, 22, 0, 0],
                [12, 23, 24, 0],
                [13, 25, 0, 0],
            ]
        ),
        "attention_mask": torch.tensor(
            [
                [1, 1, 1, 0],
                [1, 1, 0, 0],
                [1, 1, 1, 0],
                [1, 1, 0, 0],
            ]
        ),
        "loss_mask": torch.tensor(
            [
                [0, 1, 1, 0],
                [0, 1, 0, 0],
                [0, 1, 1, 0],
                [0, 1, 0, 0],
            ]
        ),
        "logprobs": torch.tensor(
            [
                [0.0, -0.1, -0.2, 0.0],
                [0.0, -0.3, 0.0, 0.0],
                [0.0, -0.4, -0.5, 0.0],
                [0.0, -0.6, 0.0, 0.0],
            ]
        ),
        "versions": torch.tensor(
            [
                [-1, 0, 0, -1],
                [-1, 0, -1, -1],
                [-1, 1, 1, -1],
                [-1, 1, -1, -1],
            ]
        ),
        "rewards": torch.tensor([-1.0, -1.0, 1.0, 1.0]),
    }
    training = role_advantage_tensor_data(
        {key: value.detach().clone() for key, value in raw.items()}
    )

    records, summary = audit_rollout_batches(
        [raw],
        [training],
        tokenizer=FakeTokenizer(),
        characters=("Alice", "Bob"),
        group_size=2,
        max_rounds=1,
        context_length=16,
        max_new_tokens=4,
    )

    assert summary["status"] == "passed"
    assert summary["errors"] == []
    assert summary["interactions"] == 4
    assert summary["character_interaction_counts"] == {"Alice": 2, "Bob": 2}
    assert records[0]["character"] == "Alice"
    assert records[1]["character"] == "Bob"
    assert records[0]["completion_token_ids"] == [20, 21]
    assert records[0]["raw_loss_mask"] == [0, 1, 1]
    assert records[0]["training_loss_mask"] == [1.0, 1.0, 0.0]
    assert records[0]["advantages"] == [-1.0, -1.0, 0.0]
    assert records[0]["completion_tokens"][0]["training_position"] == 0


def test_audit_reports_non_contiguous_completion_mask() -> None:
    torch = pytest.importorskip("torch")
    raw = {
        "input_ids": torch.tensor([[10, 20, 21], [11, 22, 23]]),
        "attention_mask": torch.ones((2, 3), dtype=torch.int64),
        "loss_mask": torch.tensor([[0, 1, 0], [0, 1, 1]]),
        "logprobs": torch.tensor([[0.0, -0.1, 0.0], [0.0, -0.2, -0.3]]),
        "versions": torch.tensor([[-1, 0, -1], [-1, 1, 1]]),
        "rewards": torch.tensor([-1.0, 1.0]),
    }
    training = role_advantage_tensor_data(
        {key: value.detach().clone() for key, value in raw.items()}
    )

    _records, summary = audit_rollout_batches(
        [raw],
        [training],
        tokenizer=FakeTokenizer(),
        characters=("Alice",),
        group_size=2,
        max_rounds=1,
        context_length=16,
        max_new_tokens=4,
    )

    assert summary["status"] == "failed"
    assert "raw_loss_mask_not_completion_suffix" in {error["code"] for error in summary["errors"]}
