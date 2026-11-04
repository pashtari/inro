"""Super-resolution: forward model, minibatches, loss and metrics."""

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import inro


@pytest.fixture
def images():
    """Return a random 32 x 48 image, its rounded 4x downsampling, and the kernels."""
    torch.manual_seed(0)
    image = torch.randint(256, (3, 32, 48), dtype=torch.uint8)
    kernels = [inro.resize_matrix(32, 8), inro.resize_matrix(48, 12)]
    low = inro.downsample(image.movedim(0, -1).float(), kernels)
    low = low.round().clamp(0, 255).to(torch.uint8).movedim(-1, 0)
    return image, low, kernels


class Lookup(nn.Module):
    """INR stub that returns the pixels under its coordinates, plus a learnable offset.

    The offset gives a loss a gradient.

    Args:
        image: Image ``(3, H, W)`` whose pixels it returns.
    """

    def __init__(self, image: torch.Tensor) -> None:
        super().__init__()
        self.pixels = inro.ImageCoordinateDataset.preprocess(image)  # (H, W, 3)
        self.offset = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = self.pixels.shape[:2]
        rows = ((x[..., 0] + 1) / 2 * (height - 1)).round().long()
        columns = ((x[..., 1] + 1) / 2 * (width - 1)).round().long()
        return self.pixels[rows, columns] + self.offset


def test_resize_matrix_is_the_antialiased_bicubic_operator():
    torch.manual_seed(0)
    image = torch.rand(1, 3, 32, 48)
    expected = F.interpolate(image, size=(8, 12), mode="bicubic", antialias=True)
    kernels = [inro.resize_matrix(32, 8), inro.resize_matrix(48, 12)]
    downsampled = inro.downsample(image[0].movedim(0, -1), kernels).movedim(-1, 0)
    torch.testing.assert_close(downsampled, expected[0], atol=1e-5, rtol=1e-5)
    assert int((kernels[0] != 0).sum(1).max()) == 16  # taps per LR pixel at 4x


def test_blocks_downsample_to_the_low_resolution_pixels(images):
    image, low, kernels = images
    dataset = inro.SuperResolutionDataset(low, kernels, num_blocks=12, block_size=2)
    torch.manual_seed(0)
    batch = dataset[0]
    rendered = Lookup(image)(batch["input"])  # (B, L, L, 3) in [-1, 1]
    fitted = inro.downsample(rendered, (batch["weight_y"], batch["weight_x"]))
    assert batch["input"].shape == (12, dataset.footprint, dataset.footprint, 2)
    assert batch["weight_y"].shape == batch["weight_x"].shape == (12, 2, 16 + 4)
    assert batch["target"].shape == (12, 2, 2, 3)
    torch.testing.assert_close(fitted, batch["target"], atol=2 / 255, rtol=0)
    assert torch.equal(batch["_original"], low)


def test_loss_matches_the_low_resolution_pixels_through_the_kernels(images):
    """Leave the true image only the LR pixels' rounding, and pull an offset back."""
    image, low, kernels = images
    dataset = inro.SuperResolutionDataset(low, kernels, num_blocks=12, block_size=4)
    batch = torch.utils.data.default_collate([dataset[0]])  # as inro.train draws it
    field = Lookup(image)
    loss, metrics = inro.super_resolution_loss(batch, field)
    assert loss.item() <= 1 / 255 and set(metrics) == {"l1", "psnr"}
    assert float(metrics["psnr"]) > 50
    with torch.no_grad():
        field.offset.fill_(0.1)
    loss, _ = inro.super_resolution_loss(batch, field)
    loss.backward()
    assert float(field.offset.grad) > 0


def test_dataset_rejects_kernels_and_blocks_that_do_not_fit(images):
    _, low, kernels = images
    with pytest.raises(ValueError, match="do not map"):
        inro.SuperResolutionDataset(low, kernels[::-1], num_blocks=4)
    with pytest.raises(ValueError, match="block_size"):
        inro.SuperResolutionDataset(low, kernels, num_blocks=4, block_size=9)
    with pytest.raises(ValueError, match="footprint"):  # 4 * 5 + 16 = 36 > 32
        inro.SuperResolutionDataset(low, kernels, num_blocks=4, block_size=6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_dataset_moves_its_kernels_with_it(images):
    _, low, kernels = images
    dataset = inro.SuperResolutionDataset(low, kernels, 4, block_size=4).to("cuda")
    batch = dataset[0]
    trained = ("input", "weight_y", "weight_x", "target")
    assert all(batch[key].is_cuda for key in trained)
    assert batch["_original"].device.type == "cpu"


def test_luma_is_bt601_in_16_to_235():
    image = torch.zeros(3, 2, 2, dtype=torch.uint8)
    image[:, 0] = 255
    torch.testing.assert_close(
        inro.luma(image), torch.tensor([[[235.0, 235.0], [16.0, 16.0]]])
    )


@pytest.mark.parametrize("border", [0, 4])
def test_metrics_score_luma_and_rgb_without_the_border(border):
    torch.manual_seed(0)
    target = torch.randint(256, (3, 40, 48), dtype=torch.uint8)
    pred = (target.int() + torch.randint(-8, 9, target.shape)).clamp(0, 255).byte()
    metrics = inro.super_resolution_metrics(border)
    inner = (slice(None), slice(border, 40 - border), slice(border, 48 - border))
    assert metrics["psnr"](pred, target) == inro.psnr(
        inro.luma(pred[inner]), inro.luma(target[inner]), data_range=255
    )
    assert metrics["ssim"](pred, target) == inro.ssim(
        inro.luma(pred[inner]), inro.luma(target[inner]), data_range=255
    )
