# FUTON

FUTON, the Fourier Tensor Network, represents a signal as a truncated series in a fixed basis whose coefficient tensor is kept in low-rank form. This page derives the model, describes each stage as it is implemented, and explains how to size one.
{ .inro-lead }

<figure markdown="span">
  ![FUTON's forward pass](../assets/futon_illustration.svg){ width="960" }
  <figcaption>The three stages for one pixel of kodim19. Each array is colored by its entries' sign and size.</figcaption>
</figure>

## From a series to a network

Take a signal $s : [-1, 1]^C \to \mathbb{R}$ and a family of $K$ functions $\varphi_1, \dots, \varphi_K$ on $[-1, 1]$. Products of one function per axis form a separable basis of $[-1, 1]^C$, and the signal expands as a generalized Fourier series:

$$
s(\boldsymbol{x}) \approx \sum_{k_1=1}^{K} \cdots \sum_{k_C=1}^{K} \mathcal{W}[k_1, \dots, k_C] \, \prod_{c=1}^{C} \varphi_{k_c}(x_c).
$$

Stored in full, the coefficient tensor $\mathcal{W}$ has $K^C$ entries: for a $512 \times 768$ image at the sampling resolution that is the image itself, and for a $256^3$ volume it is $16.8$ million coefficients. FUTON's prior is that $\mathcal{W}$ is compressible, and it stores the tensor in canonical polyadic (CP) form of rank $R$,

$$
\mathcal{W} = \sum_{r=1}^{R} \boldsymbol{u}^{(1)}_r \circ \cdots \circ \boldsymbol{u}^{(C)}_r ,
$$

with one factor matrix $\boldsymbol{U}^{(c)} = [\boldsymbol{u}^{(c)}_1, \dots, \boldsymbol{u}^{(c)}_R] \in \mathbb{R}^{K \times R}$ per axis. Substituting the factorization and exchanging the sums turns the $C$-fold summation into a product of $C$ inner products:

$$
\begin{aligned}
s(\boldsymbol{x}) &\approx \sum_{r=1}^{R} \prod_{c=1}^{C} \big\langle \boldsymbol{u}^{(c)}_r, \boldsymbol{\varphi}(x_c) \big\rangle \\
&= \boldsymbol{1}^\top \Big( \bigodot_{c=1}^{C} \boldsymbol{U}^{(c)\top} \boldsymbol{\varphi}(x_c) \Big),
\end{aligned}
$$

where $\boldsymbol{\varphi}(x_c) = [\varphi_1(x_c), \dots, \varphi_K(x_c)]$ and $\odot$ is the elementwise product. Nothing in the forward pass is $K^C$ any more: each axis is projected to $R$ channels, the channels are multiplied across axes, and the $R$ products are summed. With $D$ output channels the final sum becomes a linear map $\boldsymbol{V} \in \mathbb{R}^{D \times R}$; replacing it with a small MLP gives the model used in the experiments.

This is the pipeline `decoder(combiner(basis(x)))`:

| Stage | Computes | Parameters | Keys |
| --- | --- | --- | --- |
| **Basis** | $\boldsymbol{\varphi}(x_c)$ per axis | none | `cosine`, `sinc`, `legendre`, `chebyshev`, `triangle`, `lanczos` |
| **Combiner** | $\boldsymbol{g}(\boldsymbol{x}) = \bigodot_c \boldsymbol{U}^{(c)\top} \boldsymbol{\varphi}(x_c)$ | $R \sum_c K_c$ | `cp`, `hadamard`, `tr`, `tucker` |
| **Decoder** | $\boldsymbol{V} \boldsymbol{g}(\boldsymbol{x}) + \boldsymbol{b}$, or an MLP | $RD + D$; one hidden layer adds $R^2 + R$ | `linear`, `mlp` |

```python
model = inro.FUTON(
    in_features=3,
    out_features=1,
    basis=("sinc", {"num_components": 128}),
    combiner=("cp", {"rank": 256}),
    decoder=("mlp", {"hidden_layers": 1, "hidden_features": 128}),
    output_activation=torch.tanh,
)
```

Each stage accepts a registry key, a `(key, kwargs)` pair, a module class or a module instance, resolved by `inro.build_module`. `inro.FUTONEncoding` holds the basis, built as `basis(in_features, **kwargs)`, and the combiner, built as `combiner(basis.num_components, **kwargs)`; `FUTON` adds the decoder, built as `decoder(encoding.out_features, out_features, **kwargs)`. Custom stages only need to expose `num_components` (a basis) or `out_features` (a combiner); see [custom components](../tutorials/custom-components.md).

## What a point costs

A point costs $O(R \sum_c K_c)$, linear in the spectral resolution and in the number of axes. A local basis reduces the basis term to its $2a$ nonzero taps per axis, $O(aRC)$.

On a grid the projections $\boldsymbol{U}^{(c)\top} \boldsymbol{\varphi}(x_c)$ are shared, since each depends on one coordinate alone:

| Points | How the basis is evaluated | Cost for $N$ points |
| --- | --- | --- |
| Scattered | Point by point | $O(NR \sum_c K_c)$ |
| On a grid, $S_c$ distinct coordinates per axis | Each distinct coordinate once, every point gathering its rows | $O(R \sum_c S_c K_c + NCR)$ |
| On a grid, local basis | The same, with tables of $2a$ taps per coordinate | $O(2aR \sum_c S_c + NCR)$ |

The shared path computes the same function and gradients. FUTON takes it whenever every axis of a batch repeats its coordinates, as an image's pixels or any batch drawn from them do. Coordinates that need gradients are always evaluated point by point.

A basis given **`grid_size`**, the sizes of the grid every input lies on, goes one step further: it evaluates its features on that grid once, when it is built, and finds each point's row by rounding. That spares a sort and a GPU synchronization per axis at every batch, and a break in the compiled graph.

!!! note "A declared grid on scattered points"

    A point off the declared grid reads its nearest grid point's row, so for scattered inputs a grid trades exactness for speed. The radiance-field benchmark rounds its samples to 2048 points per axis, finer than the ray-marching step, which keeps a trained field's PSNR within a few hundredths of a decibel and renders and trains it in a fraction of the memory.

## Bases

Six one-dimensional families are implemented. All take `num_components`, an int for every axis or one count per axis, and `normalize`.

<figure markdown="span">
  ![The six bases with four functions each](../assets/bases_all.svg){ width="900" }
  <figcaption markdown="span">Four functions of each family on $[-1, 1]$. Cosine, Legendre and Chebyshev are global and indexed by degree; sinc, triangle and Lanczos are shifted kernels indexed by their center.</figcaption>
</figure>

| Key | $\varphi_k(x)$ | Orthogonal on $[-1, 1]$ | Support | Smoothness | Cardinal |
| --- | --- | --- | --- | --- | --- |
| `cosine` | $\cos(k \pi u)$, $u = (x + 1) / 2$, $k = 0, \dots, K - 1$ | yes | global | $C^\infty$ | no |
| `sinc` | $\operatorname{sinc}\big((x - \mu_k) / w\big)$ | no | global | $C^\infty$ | yes |
| `legendre` | $P_k(x)$, Bonnet's recurrence | yes | global | $C^\infty$ | no |
| `chebyshev` | $T_k(x)$ | under the weight $(1 - x^2)^{-1/2}$ | global | $C^\infty$ | no |
| `triangle` | $\max(0, 1 - \lvert x - \mu_k \rvert / w)$ | no | 2 taps | $C^0$ | yes |
| `lanczos` | $\operatorname{sinc}(t) \operatorname{sinc}(t / a)$ for $\lvert t \rvert < a$, $t = (x - \mu_k) / w$ | no | $2a$ taps | $C^1$ | yes |

The kernel families are centered on the uniform grid $\mu_k = -1 + k w$ with spacing $w = 2 / (K - 1)$, so the grid of centers is the sampling grid of a signal with $K$ samples per axis. A **cardinal** basis satisfies $\varphi_k(\mu_j) = \delta_{jk}$: each function is one at its own center and zero at all others, so the coefficients of an interpolant are the samples themselves. For sinc this is the Whittaker–Shannon interpolation formula, and Lanczos is its windowed, compactly supported version: with radius $a$ (`radius`, default 3, as in the experiments), a coordinate touches only the $2a$ nearest centers.

<div class="inro-panels inro-panels--two" markdown>

<figure markdown="span">
  ![Sinc and Lanczos interpolation](../assets/interpolation.svg)
  <figcaption>Interpolating samples with sinc (top) and Lanczos-3 (bottom) kernels: each sample scales one shifted kernel, and the sum passes through the samples.</figcaption>
</figure>

<figure markdown="span">
  ![Locality](../assets/locality.svg)
  <figcaption>At one coordinate, all 16 sinc functions are nonzero, but only 6 of the Lanczos-3 functions are.</figcaption>
</figure>

</div>

Two options shape the features:

- **`normalize`** (default `True`) L2-normalizes the feature *vector* $\boldsymbol{\varphi}(x_c)$ at each point, not the basis functions. It stabilizes training and is what the experiments use. `normalize=False` recovers the plain series: exact cardinal interpolation for the cardinal bases, and the TensoRF-CP parameterization for the triangle basis with a CP combiner.
- **`sparse`** (`triangle` and `lanczos` only, default `True`) returns each axis as `Taps`, only the $2a$ nonzero features of each coordinate and the column of the first. A product with a factor gathers $2a$ consecutive rows per coordinate, which `torch.compile` fuses into one kernel. Sparse mode expects a flat batch `(N, C)`; `FUTON.forward` reshapes for you.

The paper proves that FUTON with the cosine basis is a universal approximator. In practice the choice of family barely moves accuracy on the grid, and moves training time a lot; the [basis ablation](../experiments/ablations.md#basis) measures both.

## Combiners

| Key | Computes | Output width | Parameters |
| --- | --- | --- | --- |
| `cp` | $\bigodot_c \boldsymbol{U}^{(c)\top} \boldsymbol{\varphi}(x_c)$ | `rank` | $R \sum_c K_c$ |
| `hadamard` | $\bigodot_c \boldsymbol{\varphi}(x_c)$, all $K_c$ equal | $K$ | none |
| `tr` | $\operatorname{vec}\big(\prod_c \mathcal{G}^{(c)} \times_2 \boldsymbol{\varphi}(x_c)\big)$ with cores $\mathcal{G}^{(c)} \in \mathbb{R}^{r \times K_c \times r}$ | $r^2$ | $r^2 \sum_c K_c$ |
| `tucker` | The projections contracted with a core | `out_features` | $\sum_c K_c r_c + \prod_c r_c \cdot$ `out_features` |

- **`CPCombiner`** is the paper's model: `factors[c]` holds $\boldsymbol{U}^{(c)}$, shape `(K_c, rank)`, initialized Kaiming-uniform with fan-in $K_c$.
- **`HadamardCombiner`** is CP with $\boldsymbol{U}^{(c)} = \boldsymbol{I}$, a diagonal coefficient tensor with no parameters of its own.
- **`TRCombiner`** chains tensor-ring cores into an $r \times r$ matrix per point and flattens it, leaving the closure of the ring to the decoder. A ring of rank $r$ has the size of a CP combiner of rank $r^2$.
- **`TuckerCombiner`** projects each axis to its own rank and contracts the projections with a core, so with a linear decoder the weight tensor is in Tucker format; CP is the case of a superdiagonal core.

The [combiner ablation](../experiments/ablations.md#combiner) finds CP both faster and more accurate than the ring and Tucker at equal size.

## Decoders

- **`linear`** is `nn.Linear(R, D)`: $\hat{s}(\boldsymbol{x}) = \boldsymbol{V} \boldsymbol{g}(\boldsymbol{x}) + \boldsymbol{b}$. With it, FUTON is *exactly* a rank-$R$ CP model of the signal in the chosen basis, Eq. (9) of the paper, and its capacity is the capacity of that tensor format.
- **`mlp`** is `inro.MLP(R, D, hidden_layers=h)`, a ReLU network whose hidden width defaults to $R$. The experiments use one hidden layer. The nonlinearity lets the model represent more than a rank-$R$ tensor at the same size: on images it is worth about 8 dB over a linear decoder of the same size ([decoder ablation](../experiments/ablations.md#decoder)).

`output_activation` is applied last; the benchmark models use `torch.tanh` to match targets in $[-1, 1]$. Inside a `RadianceField`, the density network must have no output activation, since the field applies its own.

## Sizing a model

With a CP combiner and a one-hidden-layer MLP decoder of width $M$ (`hidden_features`, $R$ by default), the parameter count is

$$
R \sum_{c=1}^{C} K_c \;+\; (RM + M) \;+\; (MD + D).
$$

Each task chooses FUTON's components per axis and decoder width as every baseline's settings are chosen, on the data the baselines were tuned on: the best quality at a training time and memory that stay low. The rank is the largest multiple of 5 that keeps FUTON smaller than every baseline of the task. The benchmarks run the sinc and the cosine basis at these sizes:

| Task | `num_components` | `rank` | Decoder | Parameters |
| --- | --- | --- | --- | --- |
| Images, $512 \times 768$ Kodak | `[H // 2, W // 2]` | 280 | MLP, 1 hidden layer of width 64 | 197,379 |
| Occupancy, voxel grids of 78 to 231 per side | `[D // 2, H // 2, W // 2]` | 360 to 475 by shape | MLP, 1 hidden layer of width 64 | 130,024 to 131,229 |
| Super-resolution, $2040 \times 1356$ DIV2K | `[H // 4, W // 4]` | 510 | MLP, 1 hidden layer of width 128 | 498,785 |
| Radiance fields, density network with 16 outputs | 100, read from a table of 2048 points per axis | 180 | MLP, 1 hidden layer of width 64 | 66,624 (73,027 with the color network) |

!!! tip "Rules of thumb from the ablations"

    **`num_components` is the bandwidth.** Half the resolution per axis, $K = N/2$, is the best of $N/4$, $N/2$ and $N$ on images and volumes. On radiance fields, whose $N$ is the 200 pixels of a view, $N/4$ edges ahead of it, by about 0.1 dB on the validation views, because the MLP decoder recovers what the coarser basis loses, but at a larger rank and more memory. $K$ should never exceed what the data observes: in super-resolution, components above the low-resolution grid get no gradient through the downsampling and keep their random start, so $K$ is the low-resolution size.

    **Spend a budget on rank first.** At a fixed budget the rank matters more than extra components, so spend a larger budget on $R$ before $K$.

With PyTorch's default Adam, these learning rates work:

| Basis | Images | Volumes | Super-resolution | Radiance fields |
| --- | --- | --- | --- | --- |
| `sinc` | 3e-2 | 3e-2 | 1e-2 | 1e-2 |
| `cosine` | 1e-1 | 3e-2 | 1e-2 | 1e-2 |

## Fitting without gradients

A FUTON with a CP combiner and a linear decoder, both without bias, is multilinear: with the other blocks held, its output is linear in each factor matrix and in the decoder, so fitting one block is a least-squares problem. `inro.LeastSquares` and `inro.MultiplicativeUpdate` sweep the blocks once per batch, and `inro.train` takes them in place of Adam, on the same datasets:

```python
model = inro.FUTON(
    2, 3,
    basis=("cosine", {"num_components": [H // 2, W // 2], "grid_size": (H, W)}),
    combiner=("cp", {"rank": 280}),
    decoder=("linear", {"bias": False}),
)
solver = inro.LeastSquares(model, momentum=0)
result = inro.train(model, dataset, dataset, optimizer=solver, num_steps=500, metrics={"psnr": inro.psnr})
```

On a grid with $M_c$ coordinates along axis $c$, the model is a CP tensor with one factor per axis, the response $\boldsymbol{T}_c \boldsymbol{U}^{(c)}$ of the basis's table $\boldsymbol{T}_c \in \mathbb{R}^{M_c \times K_c}$, and one for the channels, the decoder $\boldsymbol{V}$. With the other blocks held, the normal equations of the mean squared error for $\boldsymbol{U}^{(c)}$ are small, the size of the block itself:

$$
\boldsymbol{P}_c \boldsymbol{U}^{(c)} \boldsymbol{Q}_c = \boldsymbol{B}_c ,
\qquad
\boldsymbol{P}_c = \tfrac{1}{M_c} \boldsymbol{T}_c^\top \boldsymbol{T}_c ,
\qquad
\boldsymbol{Q}_c = \boldsymbol{V}^\top \boldsymbol{V} \odot \bigodot_{j \ne c} \tfrac{1}{M_j} \big(\boldsymbol{T}_j \boldsymbol{U}^{(j)}\big)^\top \big(\boldsymbol{T}_j \boldsymbol{U}^{(j)}\big) ,
$$

where $\boldsymbol{P}_c$ is the basis's Gram matrix and $\boldsymbol{B}_c$ the targets' correlation with the rest of the model. The decoder's are $\boldsymbol{V} \boldsymbol{G} = \boldsymbol{C}$, with $\boldsymbol{G}$ the elementwise product of every axis's Gram matrix and $\boldsymbol{C}$ the targets' correlation with the whole CP product.

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
- **Scales.** Every axis's response is held at unit mean square, the decoder's columns taking the scales, which fixes the freedom of a CP model to trade scale between its blocks.
- **Momentum.** `momentum` averages each block's statistics over the batches, as online matrix factorization does (Mairal et al., 2010), with weights that sum to one from the first batch on. The newest batch's weight is the solver's `lr`: it starts at `1 - momentum`, and `inro.train` anneals it to a hundredth. Without momentum it stays one, and every sweep solves the batch's own problem.

The notebook `notebooks/futon_solvers.ipynb` compares these solvers with Adam, and the [error bound](error-bound.md) page uses `LeastSquares` on the whole grid.

## Relation to the paper

The cosine basis, CP combiner and linear decoder implement Eq. (9) of the paper: `encoding.combiner.factors[c]` stores $\boldsymbol{U}^{(c)}$ and `decoder.weight` stores $\boldsymbol{V}$. Coordinates are mapped internally from $[-1, 1]$ to $[0, 1]$. The implementation drops the paper's $\sqrt{2}$ factor on the cosines and normalizes the feature vectors by default; `normalize=False` recovers Eq. (1) up to per-frequency constants that the factors absorb. The other bases, the Hadamard, tensor-ring and Tucker combiners and the MLP decoder are extensions, and the experiments of the paper use the sinc and cosine bases with the MLP decoder and `torch.tanh` at the output.
