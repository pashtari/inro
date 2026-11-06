# FUTON, bases and combiners

The [FUTON concept page](../concepts/futon.md) derives the model; this page is the reference for `inro.models.futon`: `FUTON` and its encoding, `FUTONEncoding`, the bases and combiners, and the `SharedRows` and `Taps` features its bases return on grids and for local bases.

::: inro.models.futon

## Starts

::: inro.models.futon_inits

## Solvers

A FUTON with a CP combiner and a linear head, both without bias, is multilinear: with the other blocks held, its output is linear in each factor matrix and in the head, so fitting one block is a least-squares problem. `inro.LeastSquares` and `inro.MultiplicativeUpdate` sweep the blocks once per batch, and `inro.train` takes them in place of Adam, on the same datasets:

```python
model = inro.FUTON(
    2, 3,
    basis=("cosine", {"num_components": [H // 2, W // 2], "grid_size": (H, W)}),
    combiner=("cp", {"rank": 280}),
    head=("linear", {"bias": False}),
)
solver = inro.LeastSquares(model, momentum=0)
result = inro.train(model, dataset, dataset, optimizer=solver, num_steps=500, metrics={"psnr": inro.psnr})
```

On a grid with $M_c$ coordinates along axis $c$, the model is a CP tensor with one factor per axis, the response $\boldsymbol{T}_c \boldsymbol{U}^{(c)}$ of the basis's table $\boldsymbol{T}_c \in \mathbb{R}^{M_c \times K_c}$, and one for the channels, the head $\boldsymbol{V}$. With the other blocks held, the normal equations of the mean squared error for $\boldsymbol{U}^{(c)}$ are small, the size of the block itself:

$$
\boldsymbol{P}_c \boldsymbol{U}^{(c)} \boldsymbol{Q}_c = \boldsymbol{B}_c ,
\qquad
\boldsymbol{P}_c = \tfrac{1}{M_c} \boldsymbol{T}_c^\top \boldsymbol{T}_c ,
\qquad
\boldsymbol{Q}_c = \boldsymbol{V}^\top \boldsymbol{V} \odot \bigodot_{j \ne c} \tfrac{1}{M_j} \big(\boldsymbol{T}_j \boldsymbol{U}^{(j)}\big)^\top \big(\boldsymbol{T}_j \boldsymbol{U}^{(j)}\big) ,
$$

where $\boldsymbol{P}_c$ is the basis's Gram matrix and $\boldsymbol{B}_c$ the targets' correlation with the rest of the model. The head's are $\boldsymbol{V} \boldsymbol{G} = \boldsymbol{C}$, with $\boldsymbol{G}$ the elementwise product of every axis's Gram matrix and $\boldsymbol{C}$ the targets' correlation with the whole CP product.

=== "`LeastSquares`"

    Alternating least squares: every block takes its problem's solution, one side at a time, each with a ridge against collinear functions or components.

    - From the left, by the inverse of $\boldsymbol{P}_c$, which depends on the basis and the grid alone and is made once on a declared grid; from the right, by the Cholesky factorization of $\boldsymbol{Q}_c$.
    - Any basis will do, signed or not, and the parameters start from normal draws.
    - With an activation at the output the targets are Gauss-Newton ones, refitted every sweep, so the sweeps descend the activated model's error rather than minimize a fixed one.

=== "`MultiplicativeUpdate`"

    Lee and Seung's updates, for a nonnegative model. With everything nonnegative, the two parts of a block's gradient are too, and the update multiplies the block elementwise by their ratio:

    $$
    \boldsymbol{U}^{(c)} \leftarrow \boldsymbol{U}^{(c)} \odot \boldsymbol{B}_c \oslash \big(\boldsymbol{P}_c \boldsymbol{U}^{(c)} \boldsymbol{Q}_c\big).
    $$

    - That is the minimizer of the auxiliary function Lee and Seung build for a quadratic with a nonnegative Hessian: the block's problem is never worse after the update, the block stays nonnegative, and one at a fixed point stays there. An entry is set to zero where a minibatch's estimate of $\boldsymbol{B}_c$ is negative.
    - A block's problem is fixed for the sweep and an update costs two small products, so each block takes `inner` of them in a row, as Gillis and Glineur propose.
    - The model needs a nonnegative basis, the triangle with `normalize=False` so that its functions sum to one, nonnegative targets and nonnegative parameters. A signal in $[-1, 1]$ is fitted through the affine output activation `lambda z: 2 * z - 1`.

A few facts hold for both:

- **Batches.** A batch is taken as points of the grid the basis declares, or else of the grid of its distinct coordinates. The Gram matrices are the grid's, and a right-hand side is what the model explains on the grid plus what it misses at the batch's points, $\boldsymbol{B}_c = \boldsymbol{P}_c \boldsymbol{U}^{(c)} \boldsymbol{Q}_c + \boldsymbol{T}_c^\top \boldsymbol{E}_c / N$ for the residual's correlation $\boldsymbol{E}_c$ over the $N$ points. That is exact on a whole grid and, on a random share of it, the whole grid's in expectation, with the noise of the residual rather than of the signal. A sweep thus costs a few products of matrices over the grid whatever share of it the batch holds.
- **Monotony.** On a whole grid at `lr=1`, no update of `MultiplicativeUpdate`, or of `LeastSquares` without its ridge, raises the batch's error.
- **Scales.** Every axis's response is held at unit mean square, the head's columns taking the scales, which fixes the freedom of a CP model to trade scale between its blocks.
- **Momentum.** `momentum` averages each block's statistics over the batches, as online matrix factorization does (Mairal et al., 2010), with weights that sum to one from the first batch on. The newest batch's weight is the solver's `lr`: it starts at `1 - momentum`, and `inro.train` anneals it to a hundredth. Without momentum it stays one, and every sweep solves the batch's own problem.

The notebook `notebooks/futon_solvers.ipynb` compares these solvers with Adam.

::: inro.models.futon_solvers

## Error bounds

The [error bound page](../concepts/error-bound.md) states and proves the theorem these functions compute.

::: inro.models.futon_bounds
