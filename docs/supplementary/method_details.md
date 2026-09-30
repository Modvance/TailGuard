# Method and implementation details

## 1. Problem construction

Each dataset and imbalance setting is handled by one unified detector. We use
all 15 classes of MVTec Anomaly Detection and all 12 classes of VisA. The three
imbalance settings are:

- **Pareto:** class sizes follow a Pareto distribution with shape parameter
  $\alpha=0.6$.
- **Step-K4:** every tail class retains four normal training images.
- **Step-K1:** every tail class retains one normal training image.

Head classes account for 40% of the classes. Injected anomalies account for
10% of the constructed training set. Class identities, class frequencies, and
contamination labels are not provided to the model.

## 2. Reference eligibility

TailGuard treats the initial head and tail assignments as candidates rather
than final decisions. It determines three permissions progressively:

1. whether a training image may supervise normal reconstruction;
2. whether its patch features may enter a normal reference memory;
3. whether a test query may use that memory.

These permissions are referred to collectively as **reference eligibility**.
The three method stages implement the three decisions in sequence.

## 3. Dynamic Head Purification

Dynamic Head Purification (DHP) groups the initial head candidates by visual
appearance and observes their reconstruction errors early in training. Initial
tail candidates are protected from removal because low frequency alone does
not provide enough evidence that a sample is contaminated.

For sample $i$ in appearance group $g$, let $s_i(t)$ be the mean of the highest
1% responses in its reconstruction error map at iteration $t$. DHP normalizes
this statistic within the group:

$$
u_i(t)=\frac{s_i(t)-\operatorname{median}_{j\in g}s_j(t)}
{\operatorname{MAD}_{j\in g}s_j(t)+\epsilon},
$$

where MAD is the median absolute deviation. The group normalization avoids
using one reconstruction-error scale across visually different groups.

DHP does not assume that every group contains contamination. One-component and
two-component Gaussian mixture models are compared with the Bayesian
information criterion. The higher-error component is accepted only when the
separation, posterior confidence, and temporal persistence satisfy the fixed
decision rules. The model and optimizer then return to the selected stable
checkpoint, and high-risk samples are removed only from accepted head groups.
A per-group removal cap and a minimum retained size prevent excessive pruning.

If $\mathcal H_0$ is the initial head candidate set and
$\mathcal R_{\mathrm{DHP}}$ is the removed set, the purified head set is

$$
\mathcal H_c=\mathcal H_0\setminus\mathcal R_{\mathrm{DHP}},
\qquad
\mathcal R_{\mathrm{DHP}}\cap\mathcal T_0=\varnothing.
$$

Training resumes from the selected checkpoint using all retained samples.

## 4. Tail Reconciliation and Preservation

Protecting an initial tail candidate avoids premature deletion, but does not
determine its later role. A candidate may represent a genuine tail class or an
atypical member of a head group. Tail Reconciliation and Preservation (TRP)
therefore compares every protected candidate with the purified head structure.

Let $d_{ig}$ be the cosine distance from candidate $i$ to head group $g$, and
let $g_1$ be its nearest group. Relative group dominance is

$$
r_i=1-\frac{d_{ig_1}}
{\operatorname{median}_{g\ne g_1}d_{ig}}.
$$

This relative margin measures whether the nearest head group is distinctly
closer than the alternatives. A segmented Bayesian information criterion
comparison decides whether the ordered values support a split. Candidates
above the selected change point are affiliated with the nearest head group;
the remaining candidates form the reconciled tail set. When the evidence is
insufficient, candidates remain protected rather than being forced into a head
group. TRP changes later grouping and memory construction but does not delete
samples or alter the reconstruction model.

## 5. Conditional Tail Enhancement

Conditional Tail Enhancement (CTE) supplements underrepresented tail groups
with patch-level normal references. The purified head groups and reconciled
tail groups provide routing prototypes formed from global image features.
Patch memories are stored only for tail groups.

CTE maintains two reference views inside one module:

- the **coverage view** retains the initial tail candidates to preserve broad
  image-level evidence;
- the **reconciled view** uses the stricter TRP output for pixel localization.

A query is assigned to its nearest prototype. Tail memory contributes only
when the assigned prototype belongs to a tail group; a head-routed query keeps
the reconstruction output. If $s_{\mathrm{rec}}$ and $A_{\mathrm{rec}}$ are
the reconstruction image score and anomaly map, the final outputs are

$$
\begin{aligned}
s_{\mathrm{TG}}(x)
&=s_{\mathrm{rec}}(x)+\frac{\lambda}{2}
\left[m_{\mathrm{cov}}(x)+m_{\mathrm{rcl}}(x)\right],\\
A_{\mathrm{TG}}(x)
&=A_{\mathrm{rec}}(x)+\lambda
\mathbf 1[\hat c_{\mathrm{rcl}}(x)\in\mathcal C_{\mathrm{tail}}]
A_{\mathrm{mem}}^{\mathrm{rcl}}(x).
\end{aligned}
$$

Here, $m_{\mathrm{cov}}$ and $m_{\mathrm{rcl}}$ are routed image-level memory
scores, $\hat c_{\mathrm{rcl}}$ is the nearest reconciled prototype, and
$\mathcal C_{\mathrm{tail}}$ is the set of tail groups. Image detection uses
both views for coverage, while dense localization admits memory evidence only
from the reconciled view.

## 6. Fixed implementation settings

| Item | Setting |
|---|---|
| Encoder | Frozen DINOv2 with register tokens, ViT-B/14 |
| Reconstruction core | Dinomaly bottleneck and eight-block decoder |
| Input | Resize to 448 × 448, then center crop to 392 × 392 |
| Optimization | StableAdamW, batch size 16, 10,000 iterations |
| Learning-rate schedule | $2\times10^{-3}$ to $2\times10^{-4}$, cosine decay |
| DHP grouping | PCA-64; $k\in\{6,8,10,12,15,20\}$ |
| DHP monitoring | Iterations 100–1500, every 20 iterations |
| DHP safeguards | At most 10% removal per group; at least 20 samples retained |
| CTE fusion | $\lambda=1$; top patch ratio $\kappa=0.05$ |
| CTE storage | At most 20,000 patch features per tail group |
| Evaluation | I-AUROC, P-AUROC, and P-AUPRO |

The canonical values are fixed in
[`tailguard/config.py`](../../tailguard/config.py). They are shared across all
six settings and are not selected using test labels.
