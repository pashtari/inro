"""The training scripts' shared helpers: FUTON's CP rank from a parameter budget."""

import sys
from pathlib import Path

import pytest

import inro

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_common import fill_rank, futon_rank  # noqa: E402  # as the scripts import it


@pytest.mark.parametrize("outputs", [1, 3])
def test_futon_rank_is_the_largest_multiple_of_5_within_the_budget(outputs):
    for components, budget in (((339, 510), 500_006), ((69, 115, 59), 131_456)):
        rank = futon_rank(components, budget, 64, outputs)

        def count(rank: int) -> int:
            model = inro.FUTON(
                in_features=len(components),
                out_features=outputs,
                basis=("lanczos", {"num_components": components, "radius": 3}),
                combiner=("cp", {"rank": rank}),
                head=("mlp", {"hidden_layers": 1, "hidden_features": 64}),
            )
            return inro.count_parameters(model)

        assert rank % 5 == 0
        assert count(rank) <= budget < count(rank + 5)


def test_futon_rank_of_a_linear_head_fills_the_budget():
    budget = 131_456
    for components in ([48, 57, 44], [97, 115, 88], [135, 78, 231]):
        rank = futon_rank(components, budget, None, 1)

        def count(rank: int) -> int:
            model = inro.FUTON(
                in_features=3,
                out_features=1,
                basis=("cosine", {"num_components": components}),
                combiner=("cp", {"rank": rank}),
                head="linear",
            )
            return inro.count_parameters(model)

        assert rank % 5 == 0
        assert count(rank) <= budget < count(rank + 5)
        entry = {
            "budget": budget,
            "kwargs": {
                "basis": ["cosine", {"num_components": components}],
                "combiner": ["cp", {"rank": "budget"}],
                "head": "linear",
            },
        }
        assert fill_rank(entry, outputs=1)["kwargs"]["combiner"][1]["rank"] == rank
