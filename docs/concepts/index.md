# Concepts

The library's design, from the abstraction every model shares to the conventions that make them interchangeable.
{ .inro-lead }

<div class="grid cards" markdown>

-   :material-function-variant:{ .lg .middle } **[INRs in Inro](inrs.md)**

    ---

    A model is a function of coordinates, a dataset is a set of coordinate/value pairs, and `inro.train` ties them together.

-   :material-sine-wave:{ .lg .middle } **[FUTON](futon.md)**

    ---

    The Fourier Tensor Network, stage by stage: bases, combiners, heads, and how to size a model.

-   :material-sigma:{ .lg .middle } **[Error bound of FUTON](error-bound.md)**

    ---

    How close a FUTON with a linear head can come to a signal, exactly and in computable bounds, with the proof and an experiment.

-   :material-view-grid-outline:{ .lg .middle } **[The model zoo](models.md)**

    ---

    The thirteen architectures, their families, references and constructor arguments.

-   :material-ruler-square:{ .lg .middle } **[Conventions](conventions.md)**

    ---

    Notation, coordinate and value ranges, tensor shapes, metrics, checkpoints and seeds.

</div>
