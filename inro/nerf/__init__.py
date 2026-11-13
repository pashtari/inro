"""Novel view synthesis with radiance fields and volume renderers.

Pointwise implicit neural representations (INRs) serve as the density network of a
:class:`RadianceField`, and :func:`create_renderer` selects nerfacc when its CUDA
extension works, PyTorch otherwise. Training defaults follow torch-ngp's Blender recipe:
scaled cameras, random backgrounds, uniform ray marching, and Adam with exponential
learning-rate decay.

``inro.nerf.train`` and ``inro.nerf.evaluate`` fit and score posed images, while
``inro.train`` and ``inro.evaluate`` fit and score coordinate-value pairs.

Examples:
    >>> import inro
    >>> root = "data/blender/lego"
    >>> dataset = inro.nerf.BlenderDataset(root, downsample=4, skip=4)  # doctest: +SKIP
    >>> field = inro.nerf.RadianceField(  # doctest: +SKIP
    ...     inro.InstantNGP,
    ...     ("mlp", {"hidden_features": 64, "hidden_layers": 2}),
    ...     aabb=dataset.aabb,
    ... )
    >>> renderer = inro.nerf.create_renderer(  # doctest: +SKIP
    ...     aabb=dataset.aabb, near=dataset.near
    ... )
    >>> result = inro.nerf.train(field, renderer, dataset)  # doctest: +SKIP
    >>> test_set = inro.nerf.BlenderDataset(  # doctest: +SKIP
    ...     root, split="test", downsample=4
    ... )
    >>> scores = inro.nerf.evaluate(field, renderer, test_set)  # doctest: +SKIP
"""

from . import cameras, datasets, evaluation, fields, renderers, training
from .cameras import *
from .datasets import *
from .evaluation import *
from .fields import *
from .renderers import *
from .training import *

__all__ = [
    *cameras.__all__,
    *datasets.__all__,
    *fields.__all__,
    *renderers.__all__,
    *training.__all__,
    *evaluation.__all__,
]
