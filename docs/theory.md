# Theory

This note defines the routing problem Abyss addresses, derives the marginal-cost rule it approximates, and states every estimator the implementation uses. The classical results quoted in sections 3 and 7 hold for idealized models that differ from LLM serving in the ways listed in section 4, so they motivate the design without guaranteeing its performance.

## 1. Problem

A router assigns each arriving request to one of $R$ replicas. Request $j$ arrives at $a_j$ with $I_j$ prompt tokens, receives its first token at $f_j$, finishes at $d_j$ and produces $o_j$ output tokens. Its objectives are

```math
T_j = \beta_0 + \beta_1 \frac{I_j}{1000}, \qquad P_j = P,
```

and it misses if either is violated:

```math
\mathrm{ttft}_j = f_j - a_j, \qquad \mathrm{tpot}_j = \frac{d_j - f_j}{o_j - 1}, \qquad M_j = 1 - \mathbf{1}[\mathrm{ttft}_j \le T_j]\;\mathbf{1}[\mathrm{tpot}_j \le P_j].
```

A request abandoned by its client, or rejected, has $M_j = 1$. With per-request weights $w_j$ (1 by default), the routing policy $\pi$ should solve

```math
\min_{\pi} \; \mathbb{E}\Big[\sum_{j} w_j M_j\Big].
```

The evaluation reports attainment $A = 1 - \frac{1}{n}\sum_j M_j$ and token-weighted attainment $A_I = \sum_j I_j (1 - M_j) / \sum_j I_j$.

The state $s_r(t)$ of replica $r$ is its running and waiting requests, the progress of each, its KV memory and its prefix cache. Given its inputs, a replica evolves deterministically up to step-time noise.

## 2. Selfish and marginal-cost placement

Write $s_r \oplus q$ for the state of replica $r$ with request $q$ added. The own-cost rule places $q$ where $q$ alone is least likely to miss:

```math
r^{\mathrm{own}}(q) = \arg\min_{r} \; \mathbb{E}\big[M_q \mid s_r \oplus q\big].
```

Prefix-affinity, least-load and predicted-latency routers approximate this rule. The marginal-cost rule adds the change $q$ causes in the expected misses of the requests already on $r$, and the expected cost of the cache entries $q$ evicts:

```math
r^{\mathrm{mc}}(q) = \arg\min_{r} \;
\underbrace{w_q\,\mathbb{E}\big[M_q \mid s_r \oplus q\big]}_{\text{own}}
+ \underbrace{\sum_{j \in s_r} w_j \Big(\mathbb{E}\big[M_j \mid s_r \oplus q\big] - \mathbb{E}\big[M_j \mid s_r\big]\Big)}_{\text{externality}}
+ \underbrace{K_r(q)}_{\text{cache}}.
```

**One-step optimality.** Let $J(s) = \sum_r \sum_{j \in s_r} w_j\,\mathbb{E}[M_j \mid s_r]$ be the expected weighted misses of the requests present. If replicas evolve independently given their inputs, placing $q$ on $r$ changes only the terms of replica $r$, so

```math
J(s \oplus_r q) - J(s) = w_q\,\mathbb{E}\big[M_q \mid s_r \oplus q\big] + \sum_{j \in s_r} w_j \Big(\mathbb{E}\big[M_j \mid s_r \oplus q\big] - \mathbb{E}\big[M_j \mid s_r\big]\Big).
```

Minimizing this increment over $r$ is the marginal-cost rule without $K_r$. It is the greedy policy for the objective of section 1. It is not optimal over a sequence of arrivals; $K_r$ is a first-order correction for the effect of the placement on future requests through the prefix cache.

## 3. Why selfish placement can be much worse

**Pigou's example.** One unit of nonatomic traffic chooses between two parallel links with latencies $\ell_1(x) = 1$ and $\ell_2(x) = x$. With a share $x$ on the second link, the total cost is

```math
C(x) = (1 - x)\cdot 1 + x \cdot x .
```

Since $\ell_2(x) \le \ell_1(x)$ for all $x \le 1$, the selfish equilibrium is $x = 1$ with $C(1) = 1$. The optimum solves $C'(x) = 2x - 1 = 0$, giving $x^\star = \tfrac12$ and $C(x^\star) = \tfrac34$. The price of anarchy of this instance is

```math
\mathrm{PoA} = \frac{C(1)}{C(x^\star)} = \frac{4}{3}.
```

For affine latencies $4/3$ is the worst case over all networks (Roughgarden and Tardos, 2002). For polynomial latencies of degree $p$ the worst case is

```math
\mathrm{PoA}_p = \left(1 - \frac{p}{(p+1)^{(p+1)/p}}\right)^{-1} = \Theta\!\left(\frac{p}{\ln p}\right),
```

independent of the network topology (Roughgarden, 2003), and it is unbounded over all continuous latency functions.

**Marginal-cost tolls.** Charging each unit of traffic on link $e$ the toll $\tau_e = x_e\,\ell_e'(x_e)$ makes its perceived cost

```math
\ell_e(x_e) + x_e\,\ell_e'(x_e) = \frac{d}{dx_e}\big(x_e\,\ell_e(x_e)\big),
```

the marginal social cost. The equilibrium conditions under tolls are then the first-order optimality conditions of $\min \sum_e x_e \ell_e(x_e)$, so for convex costs the tolled equilibrium is optimal (Beckmann, McGuire and Winsten, 1956). In the example the toll on the second link is $x$, the perceived cost is $2x$, and the equilibrium moves to $x = \tfrac12$. The externality term of section 2 is the per-request analogue of $x_e \ell_e'(x_e)$.

**The externality in LLM serving.** With chunked prefill and a step budget of $B$ tokens, a replica decoding $n$ streams that admits an uncached prompt of $I$ tokens spends about $\lceil I / (B - n) \rceil$ steps on it, and every decode stream waits for each of those steps. If $t_p$ is the prefill time per token, each stream is delayed by roughly $I\,t_p$ and the aggregate delay is

```math
\Delta_{\mathrm{agg}} \approx n\, I\, t_p ,
```

which the own-cost rule does not see. For Llama 3.1 70B on four H100s, a 60k-token prompt occupies about eight steps of 0.6 s each, so every co-located stream stalls for several seconds.

## 4. What does not carry over

The results of section 3 assume nonatomic users, stationary flows and convex, continuous costs. LLM routing satisfies none of these:

- Requests are atomic, and one long prompt can saturate a replica on its own.
- Load is non-stationary and the state changes at every scheduler step.
- $M_j$ is a step function of latency, so expected misses are neither convex nor continuous in the placement.

Abyss uses the marginal-cost principle as a design rule. Its estimates come from simulation, and its evaluation in [evaluation.md](evaluation.md) is empirical.

## 5. Estimators

Abyss estimates each term of section 2 with two rollouts of a per-replica twin, a copy of the engine model of [design.md](design.md) that the router keeps synchronized with the observed token streams. All quantities below are evaluated at decision time $t$.

### 5.1 Miss probability

Predictions are converted to probabilities with a lognormal error model. For a predicted latency $x$, objective $s$ and the twin's measured log-error spread $\sigma$,

```math
\hat P(x;\, s, \sigma) = \Phi\!\left(\frac{\ln x - \ln s}{\sqrt{\sigma^2 + \sigma_0^2}}\right), \qquad \sigma_0 = 0.12 .
```

### 5.2 Own risk

Fork the twin of replica $r$, inject $q$, and step the fork until $q$ emits its first token or the horizon $t + 3T_q + 1$ passes. If the first token is reached at $\hat f_q$, then $\widehat{\mathrm{ttft}} = \hat f_q - t$. Otherwise, with $u_q$ prompt tokens left, fork time $t_H$, nominal prefill time per token $c_p$ and prefill scale $\kappa_p$,

```math
\widehat{\mathrm{ttft}} = (t_H - t) + u_q\, c_p\, \kappa_p .
```

The fork then runs up to $W = 24$ further steps. Let $k \le W$ be the number of steps run and $\bar\tau_W$ their mean duration. A long-run step time uses the decode step model $d(n, \bar c)$ for $n$ sequences at mean context $\bar c$, the decode scale $\kappa_d$ and the twin's recent prefill duty cycle $\rho$:

```math
\tau_\infty = \frac{\kappa_d\, d(n + 1, \bar c)}{1 - \min(\rho, 0.85)} .
```

With $\hat o_q$ the expected output length, the two are blended by the share of the response the window covers:

```math
\omega = \min\!\left(\frac{k}{\max(\hat o_q - 1, 1)},\, 1\right), \qquad
\widehat{\mathrm{tpot}} = \omega\,\bar\tau_W + (1 - \omega)\,\max\!\big(\tau_\infty,\, \tfrac12 \bar\tau_W\big),
```

with $\widehat{\mathrm{tpot}} = \tau_\infty$ when $k = 0$. The own term is

```math
\mathrm{own}_r(q) = w_q \Big[1 - \big(1 - \hat P(\widehat{\mathrm{ttft}};\, T_q, \sigma_T)\big)\big(1 - \hat P(\widehat{\mathrm{tpot}};\, P_q, \sigma_P)\big)\Big].
```

### 5.3 Externality

A second fork without $q$ is stepped to $\hat f_q$, the time at which the first fork produced $q$'s first token. Let $k_0$ be its step count and $\ell = (\hat f_q - t)/k_0$ its mean step time. The extra decode cost per step that $q$ adds once it decodes, with $n$ running requests at mean context $\bar c$, is

```math
\Delta d = \kappa_d \max\!\left(0,\; d\!\left(n + 1, \frac{n\bar c + I_q}{n + 1}\right) - d(n, \bar c)\right).
```

For each request $j$ on the replica:

- **No first token yet.** Let $b_j$ and $v_j$ be its first-token times in the forks without and with $q$, with $v_j = \hat f_q + \ell$ if the fork with $q$ has not reached it. Requests whose first token the fork without $q$ has not reached are skipped. If $v_j > b_j$,

```math
\Delta_j = w_j \Big[\hat P(v_j - a_j;\, T_j, \sigma_T) - \hat P(b_j - a_j;\, T_j, \sigma_T)\Big].
```

- **Decoding.** Let $\lambda_j$ be the tokens produced in the fork without $q$ but not in the fork with it, $m_j$ the estimated remaining tokens, $g_j$ the tokens observed so far and $e_j = t - f_j$ the time since its first token. The delay attributed to $q$ and the resulting mean time per token are

```math
\delta_j = \max(0, \lambda_j)\,\ell + \Delta d\,\min(m_j, \hat o_q), \qquad
N_j = \max(g_j - 1 + m_j,\, 1),
```

```math
\bar\tau_j = \frac{e_j + m_j\,\ell}{N_j}, \qquad \bar\tau_j' = \bar\tau_j + \frac{\delta_j}{N_j}, \qquad
\Delta_j = w_j \Big[\hat P(\bar\tau_j';\, P_j, \sigma_P) - \hat P(\bar\tau_j;\, P_j, \sigma_P)\Big].
```

The externality is $\mathrm{ext}_r(q) = \max\big(0, \sum_j \Delta_j\big)$. A request that misses in both forks contributes close to zero, and a request near its deadline contributes close to its weight.

### 5.4 Cache

With block size $b$ tokens, page size $p$ tokens, $B_q$ prompt blocks of which $h$ are cached on $r$, the pages $q$ needs are

```math
\pi_q = \left\lceil \frac{b\,(B_q - h) + (I_q - b B_q) + \hat o_q}{p} \right\rceil .
```

If $\pi_q$ exceeds the free pages $\phi_r$, the first $k = \lceil (\pi_q - \phi_r) / (b/p) \rceil$ blocks of the twin's LRU list are evicted. Each is valued by its reuse probability times its rebuild cost $c_b \kappa_p$:

```math
K_r(q) = \frac{\eta}{\bar T} \sum_{i=1}^{k} \hat p(h_i)\; c_b\, \kappa_p, \qquad \eta = 1,
```

where $\bar T$ is a moving average of the TTFT objectives seen, which puts $K_r$ on the same scale as the probabilities. For a block reused before, with moving-average reuse gap $g_h$ (floor 0.5 s), age $\alpha_h$ since its last use and the twin's measured residency $\theta$ from insertion to eviction,

```math
\hat p(h) = e^{-\alpha_h / 3 g_h}\left(1 - e^{-\theta / g_h}\right).
```

A block seen once uses the fleet-wide first-reuse gap $\bar g_1$ (floor 1 s) and the fraction $\hat\rho_1$ of blocks ever reused: $\hat p(h) = \hat\rho_1\, e^{-\alpha_h / 3\bar g_1}(1 - e^{-\theta/\bar g_1})$.

### 5.5 Price

A small tie-breaker separates replicas whose probabilities are all near zero:

```math
D_r(q) = \frac{\widehat{\mathrm{ttft}}}{T_q} + \frac14\,\frac{\widehat{\mathrm{tpot}}}{P_q} + \sum_{j\ \text{waiting}} \frac{v_j - b_j}{T_j} + \sum_{j\ \text{decoding}} \frac{\delta_j}{4 N_j P_j},
```

```math
\mathrm{price}_r(q) = \mathrm{own}_r(q) + \mathrm{ext}_r(q) + K_r(q) + \epsilon\, D_r(q), \qquad \epsilon = 0.05,
```

and $q$ is dispatched to $\arg\min_r \mathrm{price}_r(q)$.

### 5.6 Output length

Output lengths are unknown at dispatch. Abyss uses the empirical mean residual life over the last 4096 completed responses $o_1, \dots, o_m$:

```math
\hat m(g) = \frac{\sum_{i:\, o_i > g} o_i}{\big|\{i : o_i > g\}\big|} - g ,
```

capped at $\texttt{max\_tokens} - g$ and floored at 1, with $\hat o_q = \hat m(0)$. Before 16 completions a prior of 256 tokens is used.

### 5.7 Online calibration

Each twin learns two speed scales and two error spreads from the streams it relays.

**Error spreads.** When a first token or a completion is observed with predicted value $\hat x$ and actual value $x$,

```math
\sigma^2 \leftarrow 0.95\,\sigma^2 + 0.05 \left(\ln \frac{x}{\hat x}\right)^2, \qquad \sigma_{\text{init}} = 0.3 .
```

**Decode scale.** Over a window with at least 64 observed tokens and 16 simulated steps, let $\bar g$ be the observed mean gap per token, $S$ the simulated steps, $\Sigma$ the simulated busy time and $\Pi$ its prefill part. If $\Sigma - \Pi > 0.3\,\Sigma$,

```math
r = \mathrm{clip}\!\left(\frac{\bar g\, S - \Pi}{\Sigma - \Pi},\, \tfrac12,\, 2\right), \qquad
\kappa_d \leftarrow \mathrm{clip}\big(\kappa_d\, r^{0.2},\, 0.33,\, 3\big).
```

**Prefill scale.** When a first token arrives after an observed $y$ seconds from dispatch and the twin's own simulation produced it after $\hat y \ge 0.02$ s,

```math
\kappa_p \leftarrow \mathrm{clip}\!\left(\kappa_p \exp\!\Big(0.025\; \mathrm{clip}\big(\ln \tfrac{y}{\hat y},\, -\tfrac12,\, \tfrac12\big)\Big),\, 0.33,\, 3\right).
```

The comparison uses the twin's own simulation rather than the rollout made at routing time, because the rollout of the chosen replica is biased toward optimism by the selection itself.

**Prefill duty cycle.** Between synchronizations $\Delta t$ apart, with prefill load $L_p$ measured as simulated prefill seconds per second,

```math
\rho \leftarrow (1 - a)\,\rho + a \min(L_p, 1), \qquad a = 1 - e^{-\Delta t / 20}.
```

## 6. A worked example

The test `TestAbyssPricesTheStallOnRunningStreams` in `internal/policy/policy_test.go` builds the following case with Llama 3.1 70B on two replicas of four H100s each.

- Replica 0 holds the first 30 of 60 prompt blocks of an incoming 30,920-token prompt in its prefix cache and is decoding 48 streams with a per-token objective of 24 ms. They currently produce about one token every 21 ms.
- Replica 1 is idle and has nothing cached.

The twin predicts a time to first token of 2.05 s on replica 0 and 2.78 s on replica 1, against an objective of 5.6 s. The own risk is 0.001 on replica 0 and 0.014 on replica 1, so the own-cost rule selects replica 0. Placing the prompt there prefills its 15,560 uncached tokens in the same steps as the 48 decodes, and the rollouts estimate that 20.4 of the 48 streams are pushed past their objective. The prices are

```math
\mathrm{price}_0 = 20.76, \qquad \mathrm{price}_1 = 0.04,
```

so Abyss selects replica 1. With a 200 ms decode objective the externality on replica 0 falls to zero and the latency tie-breaker decides.

## 7. Relation to size-based assignment

For an M/G/1 queue with arrival rate $\lambda$, service time $S$ and load $\rho = \lambda\,\mathbb{E}[S]$, the Pollaczek-Khinchine formula gives the mean wait

```math
W_q = \frac{\lambda\, \mathbb{E}[S^2]}{2(1 - \rho)} .
```

The wait depends on the second moment of the job size, so a few very large jobs raise the wait of every job. Size-interval task assignment sends jobs above a cutoff $\theta$ to a dedicated pool, which lowers $\mathbb{E}[S^2]$ in the pool that serves short jobs; under high-variability job sizes it can outperform dynamic least-work-remaining assignment (Harchol-Balter, Crovella and Murta, 1999). Prompt lengths in the Mooncake conversation trace have a mean of 13,403 and a 99th percentile of 94,764 tokens.

The size-interval baseline in this repository places the $k$ replicas with the slowest prefill in the long pool $\mathcal{L}$, routes $q$ to $\mathcal{L}$ if $I_q > \theta$ and to its complement otherwise, and inside the pool chooses

```math
r^{\mathrm{SITA}}(q) = \arg\min_{r \in \mathrm{pool}(q)} \; c_r \Big( \sum_{j \in r,\ \text{waiting}} I_j + 64\,\big|\{ j \in r : \text{decoding} \}\big| + \max\big(I_q - b\, h_r(q),\, 1\big) \Big),
```

where $c_r$ is the replica's prefill time per token and $h_r(q)$ its cached blocks of $q$'s prompt. Least work is the same rule without pools.

The externality term of section 5.3 often produces the same split, because a long prefill next to running decodes is expensive, but it is not tied to a cutoff: when a replica has room, a long prompt costs little there. The cutoff rule is strong near capacity, where giving up on long prompts costs little in request count, and weaker with spare capacity, when there is room for long prompts that it does not use. The evaluation compares the two directly.

## References

- A. C. Pigou. *The Economics of Welfare*. Macmillan, 1920.
- M. Beckmann, C. B. McGuire, C. B. Winsten. *Studies in the Economics of Transportation*. Yale University Press, 1956.
- T. Roughgarden, É. Tardos. How Bad is Selfish Routing? *Journal of the ACM* 49(2), 2002.
- T. Roughgarden. The Price of Anarchy is Independent of the Network Topology. *Journal of Computer and System Sciences* 67(2), 2003.
- M. Harchol-Balter, M. E. Crovella, C. D. Murta. On Choosing a Task Assignment Policy for a Distributed Server System. *Journal of Parallel and Distributed Computing* 59(2), 1999.
