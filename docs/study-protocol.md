# Budgeted Profiling for LLM Software-Engineering Agents: A Study Protocol and Research Agenda

*(Workshop/Short Paper Draft, ~5 pages content in two-column style; no concrete API proposed)*

**Anonymous Authors**
Affiliation
Email

---

## Abstract

Large Language Model (LLM) agents are increasingly evaluated on repository-scale software engineering tasks, but most benchmark tool interfaces and incident/diagnosis datasets define *telemetry* narrowly as **logs/metrics/traces**. This is visible across operational RCA benchmarks (e.g., RCAEval explicitly frames RCA as reasoning over "metrics, logs, and traces") ([arXiv][1]) and agentic AIOps platforms (AIOpsLab's observability layer emphasizes Jaeger traces, application logs, and Prometheus metrics, exposed via an agent-cloud interface) ([Microsoft][2]). Meanwhile, industry observability standards are actively elevating **profiling** to a first-class "signal": OpenTelemetry has announced profiling support and highlights bidirectional links from logs/metrics/traces to profiles for root-cause and performance diagnosis ([OpenTelemetry][3]). In parallel, performance-centric coding benchmarks (e.g., SWE-fficiency) explicitly require agents to "profile or inspect execution" to localize hot paths, yet their evaluation harness focuses on speedup/correctness without characterizing *how* profiling should be provided under budget constraints ([GitHub][4]).

This paper argues that the missing scientific object is not "profiling exists" but **profiling as an agent tool under tight budgets**: what profile information should be exposed, at what granularity, with what output limits, and with what triggers—so that agents can close the loop between hypothesis, measurement, patching, and validation. We contribute (1) a precise problem statement for *budgeted agent-facing profiling*, (2) a set of focused research questions, and (3) a concrete, reproducible experimental protocol centered on SWE-fficiency's official CLI harness ([GitHub][4]), with optional cross-checks on SWE-Perf and PerfBench ([GitHub][5]). Our goal is to enable rigorous comparisons and motivate a community benchmark track for "agent-driven profiling," without requiring a new end-to-end framework or a finalized profiling API.

---

## 1. Introduction

### 1.1 Why profiling is the missing piece in agent evaluation

Repository-scale agent benchmarks have matured rapidly for functional bugs (e.g., SWE-bench), but performance and diagnosis remain structurally harder: you cannot validate a performance fix with a single unit test; you need **measurement infrastructure**, repeated runs, and careful interpretation under noise. PerfBench makes this explicit: validating performance improvements requires benchmarking and comparison, and their harness allows agents to generate performance benchmarks rather than relying on existing tests ([arXiv][6]). SWE-Perf similarly stresses runtime reduction while avoiding bugs in genuine repository contexts ([SWE-Perf][7]).

However, the most influential AIOps/RCA benchmark line largely operationalizes "observability" as **logs + metrics + traces**. RCAEval defines RCA as analyzing available telemetry "(i.e., metrics, logs, and traces)" during failure periods ([arXiv][1]). AIOpsLab's observability layer collects Jaeger traces, Filebeat/Logstash logs, and Prometheus metrics and exposes them through agent-facing APIs (example action: `get_logs(...)`) ([Microsoft][2]). OpenRCA similarly frames telemetry as KPI time series, trace graphs, and log text ([GitHub][8]). In short: **benchmarks and tool interfaces have standardized around L/M/T**.

This "L/M/T-only" framing is increasingly misaligned with practice. Profiling answers questions that L/M/T often cannot: which code paths burned CPU, which allocations caused GC pressure, which locks contended, which stack traces dominated. OpenTelemetry has publicly committed to making profiling a core signal and emphasizes bidirectional linking between profiles and logs/metrics/traces (e.g., "Metrics to profiles," "Traces to profiles," "Logs to profiles") ([OpenTelemetry][3]). OTel's profiling work is also moving into OTLP and collector pipelines (with caveats about stability), indicating a standard interface is forming ([OpenTelemetry][9]).

So the gap is not philosophical: **profiling is becoming standardized in observability stacks, but agent benchmarks and agent tool APIs largely ignore it**. This creates a research opportunity: to study *how to present profiling information to LLM agents under strict budgets* and how this changes success rates, speedup quality, and failure modes.

### 1.2 Why SWE-fficiency is the right primary testbed

SWE-fficiency is explicitly a repository-level benchmark for performance optimization (not bug fixing). Each task includes a full codebase, a performance workload, and a subset of guarding tests; patches are evaluated by applying them, running tests, and measuring speedups versus an expert patch using Speedup Ratio (SR), aggregated by harmonic mean ([GitHub][4]). The harness is operationally concrete: it provides a CLI workflow (gold run, predictions run, report generation) with standardized JSONL prediction format ([GitHub][4]). It also explicitly notes that the benchmark rejects instances whose speedups are not statistically significant, and recommends CPU/memory pinning for reproducibility ([GitHub][4]).

Crucially, SWE-fficiency tasks mirror real performance engineering: "profile or inspect execution, localize bottlenecks, propose correctness-preserving edits" ([SWE-fficiency][10]). That makes it ideal for our core question: **what profiling affordances actually help an agent, given a fixed budget?**

### 1.3 What this paper does (and deliberately does not do)

This paper is intentionally narrow. We do **not** propose a finalized profiling tool API or a large framework. Instead we:

1. Define **budgeted agent-facing profiling** as an experimental object.
2. Pose concrete **research questions** that isolate causal factors (presence/absence of profiling; output shape; budgets; triggering).
3. Provide a **step-by-step experimental protocol** using SWE-fficiency's official evaluation harness ([GitHub][4]), plus optional secondary checks on SWE-Perf ([GitHub][5]) and PerfBench ([arXiv][6]).

This is the kind of paper that a workshop should want: it clarifies the space, makes it measurable, and sets up follow-on work that others can cite and build on.

---

## 2. Background and Related Work

### 2.1 Agent benchmarks for performance optimization

**SWE-fficiency.** 498 tasks across 9 major Python libraries; metric is SR relative to expert speedup; evaluation is containerized with pinning recommendations and a standard CLI workflow ([GitHub][4]). The official CLI demonstrates the expected reproducible evaluation steps:

* gold baseline: `swefficiency eval --run_id ... --num_workers ...`
* model predictions: `swefficiency eval --run_id ... --prediction_path predictions.jsonl`
* report: `swefficiency report --gold_run ... --pred_run ...`
  and the predictions JSONL format with `instance_id`, `model_patch`, `model_name_or_path` ([GitHub][4]).

**SWE-Perf.** 140 repository-level instances derived from performance-improving PRs; evaluation measures runtime reduction without introducing bugs; the public repo documents environment setup and a two-stage evaluation pipeline (run evaluation; check evaluation) ([GitHub][5]).

**PerfBench.** 81 real-world .NET performance bug-fixing tasks. The benchmark emphasizes that performance fixes require benchmarking infrastructure; its harness allows agents to generate their own performance benchmarks and validates fixes by comparing execution metrics for developer vs agent fixes. Baseline OpenHands succeeds ~3%, while a performance-aware variant reaches ~20% ([arXiv][6]).

These benchmarks demonstrate that "performance" is now a first-class target. But they do not yet standardize *profiling as a tool interface* for agents—especially under strict budgets.

### 2.2 Agent benchmarks and platforms for RCA / AIOps

**RCAEval** explicitly defines RCA as analyzing telemetry data "metrics, logs, and traces" ([arXiv][1]), reinforcing the L/M/T convention.

**AIOpsLab** provides an observability layer and an agent-cloud interface (ACI). The public Microsoft Research writeup enumerates traces/logs/metrics sources (Jaeger, Filebeat+Logstash, Prometheus) and shows agent actions like `get_logs(...)` ([Microsoft][2]). It also acknowledges "data overload" and "flexible APIs to tune telemetry" ([Microsoft][2])—a theme directly analogous to our budgeted profiling problem.

**OpenRCA** similarly frames telemetry modalities around time series (KPIs), trace graphs, and log text ([GitHub][8]).

These works motivate our *gap statement*: profiling signals are largely absent from benchmark definitions and interfaces, even though real incident response often needs CPU/heap/lock evidence.

### 2.3 Profiling becoming a standardized "signal"

OpenTelemetry's profiling announcement explicitly highlights bidirectional links from metrics/traces/logs to profiles and positions profiling as delivering a new dimension of understanding with minimal effort ([OpenTelemetry][3]). Later updates ("State of Profiling") discuss profiles as a new OTLP signal type and collector pipelines for profiles (still unstable) ([OpenTelemetry][9]). Meanwhile, production tools demonstrate trace-profile correlation mechanisms (eBPF-based approaches capturing trace IDs; traces-to-profiles UIs that embed flame graphs per span) ([Do more with less. | Polar Signals][11]).

This is critical: it means the "profiling signal" is not just a one-off tool—it is converging toward **standard data models and correlation semantics**, which is exactly what agent tool interfaces can build upon.

---

## 3. Problem Statement: Budgeted Agent-Facing Profiling

We define an **agent-facing profiling tool** as any mechanism that, given a target program/workload, returns an **evidence artifact** describing resource consumption (e.g., CPU samples, allocations, lock contention) in a form the agent can consume.

The research challenge is that profiling has three hard constraints in agent settings:

1. **Budgeted runtime overhead.** Profiling consumes time and can perturb measurements.
2. **Budgeted output size.** Raw profiles (pprof dumps, full call trees) can overwhelm context windows.
3. **Decision usefulness.** The artifact must support actionable localization (what to change) and closure (did the change help, and why).

Our goal is to study: **which profiling information, delivered under which budgets, measurably improves agent performance outcomes** on repository-scale tasks.

We intentionally treat "API design" as latent: instead of proposing endpoints, we treat the profiling tool as producing observations under controlled knobs (budget, format, trigger). The core scientific object is the mapping:

> (profiling knobs, agent budget) → (speedup quality, correctness, tool efficiency, failure modes)

---

## 4. Research Questions

We propose five focused RQs, each tied to an experimental manipulation that can be implemented without redesigning the benchmark.

### RQ1: Does providing profiling capability improve agent outcomes under fixed total budget?

Compare **no profiling** vs **profiling-enabled** agents, holding constant: model, prompts, action limits, and total wall-clock budget per task.

Outcomes: SWE-fficiency `overall_score` (harmonic mean SR), `proportion_incorrect`, `proportion_correct_but_no_speedup`, `proportion_human_speedup_or_better` ([GitHub][4]).

### RQ2: How sensitive are agents to the *representation* of profiling evidence?

Holding the underlying profile data fixed, vary only the *presentation*:

* ultra-compressed top-K hot symbols
* hierarchical call-path summaries
* diff-style summaries (before vs after patch)

Measure changes in SR and correctness, and track token footprint.

### RQ3: What is the trade-off curve between profiling budget and performance gain?

Vary the allowed profiling budget (e.g., maximum number of profiling calls and/or maximum seconds per call) while keeping total task budget fixed. Identify diminishing returns or regressions (e.g., agent wastes time profiling).

### RQ4: For which bottleneck classes is profiling most/least useful?

Partition tasks by bottleneck characteristics (derived from expert patch or offline profiling) and measure per-class gains. This yields scientific insights beyond raw leaderboard numbers.

### RQ5 (optional but valuable): What are the dominant failure modes when agents use profiling tools?

Quantify "profiling spam," misuse, and whether the agent closes the loop with post-patch validation.

---

## 5. Experimental Protocol (Concrete, Reproducible)

We center the protocol on SWE-fficiency because it provides a stable CLI evaluation harness and standardized outputs ([GitHub][4]).

### 5.1 Benchmark selection and instance sampling

**Primary benchmark:** SWE-fficiency (498 tasks across 9 Python libraries) ([GitHub][4]).

**Subsampling (workshop-feasible, reproducible):**

1. Stratify by repository (9 repos).
2. Uniformly sample *n* tasks per repo (e.g., n=5 → 45 tasks).
3. Publish instance IDs and sampling seed in an appendix.

Rationale: full 498×multi-condition runs are expensive; stratified sampling reduces cherry-picking and preserves diversity.

### 5.2 Evaluation harness (official)

We use SWE-fficiency's official CLI:

1. **Gold baseline run**
   `swefficiency eval --run_id my_eval --num_workers 12`
   Produces gold reference performance under expert patches ([GitHub][4]).

2. **Run model predictions**
   `swefficiency eval --run_id my_eval --num_workers 12 --prediction_path predictions.jsonl` ([GitHub][4])

   Predictions file format (JSONL, one per instance):
   `{"instance_id":"<id>","model_patch":"<patch_text>","model_name_or_path":"<model_name>"}` ([GitHub][4])

3. **Generate report**
   `swefficiency report --gold_run ... --pred_run ...`
   Produces per-instance CSV and summary JSON with the key metrics listed above ([GitHub][4]).

**Reproducibility controls:** follow SWE-fficiency guidance on VM/container setup and CPU/memory pinning; they recommend allocating 4 vCPUs and 16GB RAM per worker and provide setup scripts ([GitHub][4]).

### 5.3 Agent setup (generation)

We separate **generation** from **evaluation**:

* Generation runs an agent scaffold (e.g., OpenHands/SWE-agent style) inside the same containerized repo snapshot used for evaluation.
* Output is a unified diff patch text, stored into `predictions.jsonl`.

**Budgets and fairness (must be enforced):**

* fixed maximum actions/steps per task (e.g., 100 actions, consistent with SWE-fficiency's emphasis on iterative workflow ([SWE-fficiency][10]))
* fixed wall-clock budget per task
* profiling time and profiling output tokens count against the same budgets

### 5.4 Experimental conditions

We propose the minimal set of conditions required to answer RQ1–RQ3:

**C0 (NoProfiling):** agent has standard repo tools (search, run tests/workload) but cannot request profiling evidence.

**C1 (ProfilingEnabled):** agent can request a profiling evidence artifact under a strict budget. The mechanism can be implemented as a wrapper executable or a tool hook; the paper does not standardize its API, only its budgets and outputs.

**C2/C3/C4 (Representation variants for RQ2):** same as C1, but profile evidence is presented differently (compressed/hierarchical/diff). Under the hood the same raw profile is parsed; only the agent-visible artifact differs.

**B0/B1/B2/B3 (Budget variants for RQ3):** vary profiling call limits and/or time caps per call; keep total task budget fixed.

A compact depiction (for the paper):

| Condition | Profiling available | Output form                | Profiling budget |
| --------- | ------------------: | -------------------------- | ---------------- |
| C0        |                  No | —                          | 0                |
| C1        |                 Yes | Fixed "baseline" summary   | Medium           |
| C2        |                 Yes | Top-K compressed           | Medium           |
| C3        |                 Yes | Hierarchical               | Medium           |
| C4        |                 Yes | Diff (before/after)        | Medium           |
| B1        |                 Yes | Best-performing from C2–C4 | Low              |
| B2        |                 Yes | Best-performing            | Medium           |
| B3        |                 Yes | Best-performing            | High             |

### 5.5 Instrumentation and logs (required for RQ4/RQ5)

For each task run, record:

* instance_id
* final patch
* SWE-fficiency outcome metrics from report (`overall_score`, correctness-related proportions) ([GitHub][4])
* agent trajectory events:

  * command/tool invoked
  * wall time per step
  * stdout size (chars) and estimated token count
* profiling-specific:

  * number of profiling calls
  * time spent profiling
  * size of profiling artifact delivered to agent

These logs enable both statistical evaluation and failure-mode analysis.

### 5.6 Statistical analysis plan

Because per-instance SR values are heavy-tailed, we emphasize robust aggregation:

* primary: report SWE-fficiency `overall_score` (harmonic mean SR) ([GitHub][4])
* paired comparisons: per-instance SR differences between conditions; bootstrap confidence intervals
* correctness deltas: `proportion_incorrect` changes (profiling might reduce "blind edits") ([GitHub][4])
* cost-effectiveness: SR vs total token+time consumption (a workshop-friendly derived metric)

---

## 6. Threats to Validity and Rigor Checklist

This is where most weak workshop papers fail. If you want to be taken seriously, you must write these explicitly.

### 6.1 Measurement noise and perturbation

Performance measurements vary with CPU scheduling, caches, background activity. SWE-fficiency mitigates this by containerization, pinning recommendations, and rejecting instances whose speedups are not statistically significant ([GitHub][4]). Still, profiling itself perturbs runtime; this is why we treat profiling budget as a first-class variable (RQ3).

### 6.2 Fairness of budgets

If profiling-enabled agents get more wall-clock or tokens, any "improvement" is meaningless. Profiling overhead and artifact size must be counted against the same budgets. This is non-negotiable.

### 6.3 Generality beyond Python

SWE-fficiency is Python-centric. To mitigate, we propose optional secondary validation:

* SWE-Perf (repo-level optimization tasks) ([GitHub][5])
* PerfBench (.NET performance bug fixing with agent-generated benchmarks) ([arXiv][6])

Even small-sample validation improves credibility.

### 6.4 Overfitting to benchmark-specific quirks

Agents can "learn the harness." To reduce this:

* report tool-usage patterns (RQ5)
* run representation/budget ablations
* publish instance list and scripts

---

## 7. Discussion: What outcomes would be meaningful?

This work is valuable even without a perfect solution if it produces **clear empirical claims**:

1. **Profiling helps only when represented correctly.** Raw profiles may not help; compressed/diff summaries might.
2. **Budget matters more than access.** Too little profiling yields no signal; too much encourages tool spam.
3. **Profiling helps specific bottleneck classes.** For example, compute hot paths vs data movement vs lock contention.
4. **Failure modes are systematic.** Agents may profile blindly, misattribute causality, or optimize micro-hotspots that do not move end-to-end runtime.

These are publishable workshop-level findings because they define a measurable design space and motivate future interface and benchmark tracks.

---

## 8. Conclusion

Profiling is becoming a standardized observability signal, with explicit correlation semantics to logs/metrics/traces in OpenTelemetry ([OpenTelemetry][3]) and real tooling ecosystems demonstrating trace-to-profile linking ([Do more with less. | Polar Signals][11]). Yet current agent benchmarks and AIOps/RCA evaluation frameworks largely operationalize telemetry as L/M/T ([arXiv][1]). This mismatch creates an actionable research gap: **budgeted, agent-facing profiling**.

We propose a focused research agenda and a concrete experimental protocol that can be executed today using SWE-fficiency's official evaluation harness and metrics ([GitHub][4]), optionally cross-validating on SWE-Perf and PerfBench ([GitHub][5]). The goal is to convert "profiling should help" into a rigorous, comparable experimental object—so that future work on profiling tool interfaces for agents can be measured, reproduced, and meaningfully cited.

---

## References (indicative)

*(You would format these in BibTeX for submission; below are the key cited sources in this draft.)*

* SWE-fficiency website and harness documentation ([SWE-fficiency][10])
* SWE-Perf benchmark site and repo ([SWE-Perf][7])
* PerfBench arXiv abstract ([arXiv][6])
* RCAEval arXiv HTML ([arXiv][1])
* AIOpsLab Microsoft Research blog ([Microsoft][2])
* OpenRCA GitHub ([GitHub][8])
* OpenTelemetry profiling announcement and profiling state update ([OpenTelemetry][3])
* Trace–profile correlation blog and traces-to-profiles docs ([Do more with less. | Polar Signals][11])

---

### (Appendix stub you can add for submission)

* A. Instance sampling seed + list
* B. Budget definitions (wall-clock, steps, token accounting)
* C. Log schema (JSONL fields)
* D. Minimal scripts to run SWE-fficiency evaluation (`eval`, `report`) ([GitHub][4])

---

## Next Steps (Author Notes)

如果你接下来要把它变成"能投出去的 workshop paper"，我建议你马上补两块内容（不需要发明 API）：

1. **一个小规模 pilot 实验结果**：哪怕只跑 20–30 个 SWE-fficiency 实例，对比 C0 vs C1 vs（C2/C4 任选一个），把 `overall_score` 和 `proportion_incorrect` 报出来 ([GitHub][4])。
2. **失败模式定量**：profiling 调用次数分布、profiling 输出 token 占比、以及"profiling spam"在低/高预算下的变化（RQ5）。这会让 paper 从"议论文"变成"实验论文"。

[1]: https://arxiv.org/html/2412.17015v1 "RCAEval: A Benchmark for Root Cause Analysis of Microservice Systems with Telemetry Data"
[2]: https://www.microsoft.com/en-us/research/blog/aiopslab-building-ai-agents-for-autonomous-clouds/ "AIOpsLab: Building AI agents for autonomous clouds - Microsoft Research"
[3]: https://opentelemetry.io/blog/2024/profiling/ "OpenTelemetry announces support for profiling | OpenTelemetry"
[4]: https://github.com/swefficiency/swefficiency "GitHub - swefficiency/swefficiency: Benchmark harness and code for \"SWE-fficiency: Can Language Models Optimize Real World Repositories on Real World Workloads?\""
[5]: https://github.com/SWE-Perf/SWE-Perf "GitHub - SWE-Perf/SWE-Perf"
[6]: https://arxiv.org/abs/2509.24091?utm_source=chatgpt.com "PerfBench: Can Agents Resolve Real-World Performance Bugs?"
[7]: https://swe-perf.github.io/ "SWE-Perf: Can Language Models Optimize Code Performance on Real-World Repositories?"
[8]: https://github.com/microsoft/OpenRCA "GitHub - microsoft/OpenRCA: [ICLR'25] OpenRCA: Can Large Language Models Locate the Root Cause of Software Failures?"
[9]: https://opentelemetry.io/blog/2024/state-profiling/ "The State of Profiling | OpenTelemetry"
[10]: https://swefficiency.com/ "SWE-fficiency"
[11]: https://www.polarsignals.com/blog/posts/2024/03/05/correlating-tracing-with-profiling-using-ebpf "Correlating Tracing with Profiling using eBPF"
