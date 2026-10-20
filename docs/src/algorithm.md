# Algorithm

The model uses the exact conditional likelihood. Its denominator is computed with the
Gail–Lubin–Rubinstein (1981) recursion in a normalised log-space "mixture" form, which
carries log B, E[x_S] and Var[x_S] for each subset size. Every update is a convex
combination plus a rank-one PSD term. As a result there is no overflow, even for strata
with thousands of events, and the information matrix is positive semidefinite by
construction.

Other details:
- When m_k > n_k − m_k the recursion runs on the complement, so the cost is
  O(n_k · min(m_k, n_k − m_k) · p²) per stratum.
- Strata with one event (or one non-event) use a two-pass centred softmax in O(n_k p²).
- Covariates are centred within strata before fitting. The likelihood is invariant to
  this, and it removes cancellation in the information matrix when covariates have large
  stratum-level means.
- Strata are processed in chunks by several tasks, the most expensive chunks first. The
  chunk partition depends only on the data, and partial sums are reduced in a fixed
  order. Results are therefore bitwise identical for any number of threads (`julia -t N`).
- A single very large stratum (when p ≥ 8, or when its p × p × m_k work tensor would
  exceed 8 MiB) has its information matrix split by columns over tasks. Every entry is
  computed by the same operations, so the results stay bitwise identical. Per-task
  memory does not grow with the largest stratum.

