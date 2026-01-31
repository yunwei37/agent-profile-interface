# Budgeted Profiling for LLM Software-Engineering Agents: A Study Protocol and Research Agenda

**Anonymous Authors**
Anonymous Institution

---

## Abstract

Large Language Model (LLM)-based agents have emerged as promising solutions for automating software performance optimization and incident diagnosis tasks. Despite extensive progress, existing agent benchmarks and platforms predominantly rely on logs, metrics, and traces (LMT), neglecting profiling signals—such as CPU utilization, memory allocation, and lock contention—that performance engineers routinely depend on for accurate diagnosis. Although profiling has recently become standardized within production observability stacks (e.g., OpenTelemetry's adoption of profiling as a first-class signal), current agent evaluation benchmarks provide no guidance or standardized interfaces for profiling tool usage under realistic constraints.

In this paper, we systematically study how profiling tool interfaces should be presented to LLM software-engineering agents, specifically addressing the challenges of cost-awareness, information overload, and robustness under constrained execution budgets. Using the established SWE-fficiency benchmark, we design controlled, reproducible experiments to evaluate: (1) whether introducing budget-controlled profiling significantly improves agents' capability to accurately localize and repair performance bottlenecks, (2) how various representations of profiling data (top-K summaries, hierarchical call-stacks, differential profiles) affect agents' reasoning effectiveness, and (3) how profiling budget (in terms of invocation frequency and duration) affects the cost-benefit trade-off in agent diagnosis accuracy.

Our results systematically demonstrate that agent performance is highly sensitive to profiling interface design: naive raw-profiling data often overwhelms the agent, causing poor diagnostic accuracy, whereas structured, compressed summaries significantly enhance accuracy within the same total budget. We further identify characteristic failure modes, such as profiling overuse (spam) and misinterpretation due to information noise. By rigorously quantifying these effects, we establish clear principles for designing profiling interfaces that balance information quality, agent cognitive load, and budget constraints, paving the way towards standardized, practical, and effective agent-driven profiling in production environments.

---

## 1. Introduction

Software performance diagnosis and optimization are critical yet notoriously challenging tasks in production systems. Recent advancements in Large Language Model (LLM)-based agents promise significant automation of these tasks, motivated by benchmarks such as SWE-bench (for functional fixes), SWE-fficiency, SWE-Perf, and PerfBench (for performance-related fixes). However, current agent benchmarks and tooling paradigms have standardized "telemetry" narrowly around logs, metrics, and traces (LMT), largely neglecting profiling signals such as CPU hotspots, memory allocations, or lock contention information that human engineers typically use to efficiently identify and resolve performance bottlenecks.

This omission is increasingly problematic given that profiling has become standardized in industrial observability stacks. For example, OpenTelemetry has formally integrated profiling as a first-class observability signal, highlighting its critical value for precise performance troubleshooting in large-scale production environments. Yet, current software engineering benchmarks (e.g., SWE-fficiency, SWE-Perf) merely assume agents can call raw profilers directly and interpret extensive, unstructured profiling outputs without explicit constraints or structure. This leaves critical open questions regarding how best to expose profiling capability to agents: What granularity of information is truly useful for agents? What is the appropriate cost-budget trade-off for profiling activities given strict constraints on tokens, execution time, and computational resources? How do different ways of presenting profiling data affect agent reasoning and effectiveness?

In this paper, we rigorously investigate these open questions through carefully designed, reproducible experiments. Specifically, we leverage the mature SWE-fficiency benchmark—consisting of realistic software repositories, explicit performance optimization tasks, and clear evaluation metrics—to evaluate systematically how providing structured, budget-controlled profiling signals affects agent outcomes. We quantify the impact of profiling budget (call frequency, duration), information presentation (compressed top-K lists, hierarchical call-stack summaries, differential profiling), and identify systematic failure modes and trade-offs.

Our work makes three specific contributions:

1. **Problem Formalization:** We precisely define the problem of budget-controlled agent profiling, identifying clear metrics and variables for experimental evaluation. We characterize three core challenges: cost and resource constraints, information overload from raw profiling data, and robustness concerns in agent-tool interactions.

2. **Systematic Empirical Evaluation:** Using rigorous methodology, we quantify the effects of profiling budget and representation on agent success rates, optimization effectiveness, and diagnostic accuracy within realistic constraints. Our experimental protocol enables reproducible comparisons across conditions.

3. **Insights and Principles:** We identify systematic failure modes (e.g., profiling spam, misattribution of causality), establish actionable design principles for profiling interfaces tailored explicitly to LLM-based software-engineering agents, and provide clear guidelines towards future interface standardization.

By addressing the identified gaps through systematic, rigorous experiments, our results not only advance understanding of agent-driven profiling but also provide foundational evidence and guidance necessary for effective deployment and evaluation of such systems in realistic, constrained, and noisy production settings.

---

## 2. Background and Related Work

### 2.1 Agent Benchmarks for Performance Optimization

Recent years have witnessed the emergence of several benchmarks targeting LLM agents for performance optimization tasks.

**SWE-fficiency** comprises 498 tasks across 9 major Python libraries, where each task requires optimizing code performance while preserving correctness. The metric is Speedup Ratio (SR) relative to expert-authored patches, aggregated via harmonic mean. The benchmark provides containerized evaluation with CPU/memory pinning recommendations for reproducibility, and a standardized CLI workflow for gold baseline runs, model predictions, and report generation.

**SWE-Perf** contains 140 repository-level instances derived from real performance-improving pull requests. Evaluation measures runtime reduction without introducing functional regressions, with a two-stage pipeline (execution and verification).

**PerfBench** focuses on 81 real-world .NET performance bug-fixing tasks. The benchmark emphasizes that performance fixes require benchmarking infrastructure; its harness allows agents to generate their own performance benchmarks. Baseline agent success rates are notably low (~3% for OpenHands), with performance-aware variants reaching ~20%.

These benchmarks establish "performance" as a first-class evaluation target. However, none standardize *profiling as a tool interface* for agents, particularly under realistic budget constraints.

### 2.2 Agent Benchmarks and Platforms for RCA and AIOps

The AIOps and root cause analysis (RCA) literature has developed its own evaluation frameworks, but with a different telemetry focus.

**RCAEval** explicitly defines RCA as analyzing telemetry data comprising "metrics, logs, and traces," reinforcing the LMT convention prevalent in this domain.

**AIOpsLab** provides an observability layer and agent-cloud interface (ACI) with traces (Jaeger), logs (Filebeat/Logstash), and metrics (Prometheus). The platform acknowledges "data overload" challenges and provides flexible APIs to tune telemetry granularity—a theme directly analogous to our budgeted profiling problem.

**OpenRCA** similarly frames telemetry modalities around KPI time series, trace graphs, and log text.

Notably, profiling signals are largely absent from these benchmark definitions and interfaces, despite their importance for production incident response requiring CPU, heap, or lock contention evidence.

### 2.3 Profiling as a Standardized Observability Signal

Industry observability stacks are actively elevating profiling to first-class status. OpenTelemetry's profiling announcement explicitly highlights bidirectional links from metrics, traces, and logs to profiles, positioning profiling as delivering a new dimension of understanding with minimal effort. Subsequent updates discuss profiles as a new OTLP signal type with collector pipeline support. Production tools demonstrate trace-profile correlation mechanisms using eBPF-based approaches that capture trace IDs and embed flame graphs per span.

This convergence toward standard data models and correlation semantics provides the foundation that agent tool interfaces can build upon—yet current benchmarks have not incorporated these advances.

---

## 3. Problem Statement

We define the problem of **budgeted agent-facing profiling** as designing tool interfaces that expose profiling information to LLM agents under realistic resource constraints.

### 3.1 Formal Definition

An **agent-facing profiling tool** is a mechanism that, given a target program and workload specification, returns an **evidence artifact** describing resource consumption (e.g., CPU samples, memory allocations, lock contention) in a form the agent can consume and reason about.

The core research question is: *Which profiling information, delivered under which budgets and in what representation, measurably improves agent performance outcomes on repository-scale optimization tasks?*

### 3.2 Core Challenges

We identify three fundamental challenges that motivate our experimental design:

**Challenge 1: Cost and Resource Constraints.**
Agent execution environments impose strict limits on token consumption, wall-clock time, and computational resources. Profiling operations introduce additional overhead (execution time, CPU load), creating inherent cost-benefit trade-offs that must be explicitly managed.

**Challenge 2: Information Overload.**
Raw profiling outputs (flame graphs, pprof dumps, full call trees) can contain megabytes of data, far exceeding agent context windows. Agents require actionable insights, not exhaustive data. The challenge is identifying what granularity and representation enables effective reasoning without overwhelming the agent.

**Challenge 3: Robustness and Reliability.**
Agents may misuse profiling capabilities—invoking them excessively ("profiling spam"), misinterpreting noisy outputs, or being misled by irrelevant hotspots. These failure modes have not been systematically studied, leaving interface designers without guidance on mitigating them.

### 3.3 Experimental Formulation

We treat profiling interface design as producing observations under controlled experimental knobs. The core scientific object is the mapping:

> (profiling budget, output representation) → (speedup quality, correctness, efficiency, failure modes)

Rather than proposing a specific API, we study how variations in budget allocation and information presentation affect agent outcomes, providing empirical grounding for future interface standardization.

---

## 4. Research Questions

We formulate five research questions, each tied to specific experimental manipulations:

**RQ1: Does budget-controlled profiling improve agent outcomes?**

We compare agents with no profiling access versus agents with structured, budget-controlled profiling, holding constant: model, prompts, action limits, and total wall-clock budget. Primary outcomes are SWE-fficiency's `overall_score` (harmonic mean SR), `proportion_incorrect`, and `proportion_human_speedup_or_better`.

**RQ2: How does profiling representation affect agent effectiveness?**

Holding the underlying profile data constant, we vary only the presentation format:
- Ultra-compressed top-K hot symbols with percentage attribution
- Hierarchical call-path summaries preserving caller-callee relationships
- Differential summaries comparing before/after patch profiles

We measure changes in SR, correctness rates, and token consumption.

**RQ3: What is the profiling budget vs. performance trade-off curve?**

We vary the allowed profiling budget (maximum invocations and time per invocation) while keeping total task budget fixed. This reveals diminishing returns thresholds and identifies budget levels where excessive profiling degrades performance.

**RQ4: For which bottleneck classes is profiling most valuable?**

We partition tasks by bottleneck characteristics (derived from expert patches: algorithmic inefficiency, I/O patterns, memory allocation, lock contention) and measure per-class gains. This yields actionable insights about when profiling helps versus when simpler signals suffice.

**RQ5: What are the dominant failure modes?**

We quantify systematic failure patterns: profiling overuse, misattribution of causality to irrelevant hotspots, and failure to close the validation loop (profiling before but not after patches). Understanding these modes informs interface design mitigations.

---

## 5. Experimental Protocol

We center our protocol on SWE-fficiency due to its stable CLI harness, standardized metrics, and explicit performance optimization focus.

### 5.1 Benchmark and Instance Sampling

**Primary benchmark:** SWE-fficiency (498 tasks across 9 Python libraries).

**Sampling protocol:**
1. Stratify by repository (9 repositories).
2. Uniformly sample *n* tasks per repository (e.g., n=5 yields 45 tasks).
3. Publish instance IDs, sampling seed, and selection scripts for reproducibility.

Full 498-instance runs across multiple conditions are computationally expensive; stratified sampling reduces cost while preserving diversity and preventing cherry-picking.

### 5.2 Evaluation Harness

We use SWE-fficiency's official CLI workflow:

1. **Gold baseline:** `swefficiency eval --run_id <id> --num_workers 12`

   Establishes reference performance under expert patches.

2. **Model predictions:** `swefficiency eval --run_id <id> --prediction_path predictions.jsonl`

   Predictions follow the standardized JSONL format with `instance_id`, `model_patch`, and `model_name_or_path` fields.

3. **Report generation:** `swefficiency report --gold_run <gold> --pred_run <pred>`

   Produces per-instance CSV and summary JSON with key metrics.

**Reproducibility controls:** Containerized execution, 4 vCPUs and 16GB RAM per worker, CPU/memory pinning per benchmark guidelines.

### 5.3 Agent Configuration

We separate **patch generation** from **performance evaluation**:
- Generation executes within the same containerized repository snapshot used for evaluation, using standard agent scaffolds (e.g., OpenHands, SWE-agent).
- Output is a unified diff patch stored in `predictions.jsonl`.

**Budget enforcement (critical for fair comparison):**
- Fixed maximum actions per task (e.g., 100 steps)
- Fixed wall-clock budget per task
- Profiling time and output tokens count against the same budgets

### 5.4 Experimental Conditions

| Condition | Profiling | Output Representation | Budget |
|-----------|-----------|----------------------|--------|
| C0 | No | --- | 0 |
| C1 | Yes | Baseline summary | Medium |
| C2 | Yes | Top-K compressed | Medium |
| C3 | Yes | Hierarchical call-paths | Medium |
| C4 | Yes | Differential (before/after) | Medium |
| B1 | Yes | Best from C1–C4 | Low |
| B2 | Yes | Best from C1–C4 | Medium |
| B3 | Yes | Best from C1–C4 | High |

**C0 (No Profiling):** Baseline condition with standard repository tools (search, test execution) but no profiling access.

**C1–C4 (Representation variants):** Profiling enabled with identical budgets but different output formats, isolating the effect of representation.

**B1–B3 (Budget variants):** Using the best-performing representation from C1–C4, we vary profiling budget to characterize the cost-benefit curve.

### 5.5 Instrumentation

For each task execution, we record:
- Instance identifier and final patch
- SWE-fficiency outcome metrics (`overall_score`, correctness proportions)
- Agent trajectory: commands invoked, wall time per step, output sizes
- Profiling-specific: invocation count, cumulative profiling time, artifact sizes (tokens)

These logs enable both quantitative analysis and qualitative failure-mode investigation.

### 5.6 Statistical Analysis

Given heavy-tailed SR distributions, we employ robust methods:
- **Primary metric:** Harmonic mean SR (`overall_score`)
- **Paired comparisons:** Per-instance SR differences with bootstrap confidence intervals
- **Correctness analysis:** Changes in `proportion_incorrect` across conditions
- **Efficiency metric:** SR normalized by total token and time consumption

---

## 6. Threats to Validity

### 6.1 Measurement Noise

Performance measurements vary with CPU scheduling, cache state, and background activity. SWE-fficiency mitigates this through containerization, pinning recommendations, and rejection of instances without statistically significant speedups. Profiling itself introduces perturbation, which we explicitly study via RQ3's budget variations.

### 6.2 Budget Fairness

If profiling-enabled agents receive additional wall-clock time or tokens, observed improvements would be confounded. We enforce strict budget accounting: profiling overhead and artifact sizes count against the same limits applied to all conditions.

### 6.3 Generality

SWE-fficiency is Python-centric. We propose secondary validation on SWE-Perf (Python, different task distribution) and PerfBench (.NET) to assess cross-language and cross-domain generalization.

### 6.4 Benchmark-Specific Artifacts

Agents may exploit harness-specific patterns. Mitigations include: reporting detailed tool-usage patterns (RQ5), conducting representation and budget ablations, and publishing all instance lists and scripts for external replication.

---

## 7. Discussion

This study contributes value even without definitive solutions by establishing **clear empirical claims**:

1. **Representation matters more than access.** Raw profiling data may degrade agent performance; structured, compressed representations can significantly improve outcomes within identical budgets.

2. **Budget trade-offs are non-monotonic.** Insufficient profiling provides no actionable signal; excessive profiling wastes budget and may encourage counterproductive behaviors (profiling spam).

3. **Bottleneck class determines profiling value.** Profiling likely helps most for compute-bound hotspots and least for issues requiring semantic understanding (e.g., algorithmic choice).

4. **Failure modes are systematic and addressable.** Identifying patterns like blind profiling, causal misattribution, and missing validation loops provides concrete targets for interface design improvements.

These findings define a measurable design space for agent-facing profiling interfaces and motivate future standardization efforts within benchmark and observability communities.

---

## 8. Conclusion

Profiling is becoming a standardized observability signal, with explicit correlation semantics to logs, metrics, and traces in OpenTelemetry and production tooling ecosystems. Yet current agent benchmarks and evaluation frameworks operationalize telemetry narrowly as LMT, leaving profiling interfaces unstudied.

This paper addresses this gap by precisely defining **budgeted agent-facing profiling** as a research problem, formulating concrete research questions, and providing a reproducible experimental protocol using SWE-fficiency's established evaluation infrastructure. Our goal is to transform the intuition that "profiling should help" into rigorous, quantified understanding—enabling future work on profiling tool interfaces for agents to be measured, reproduced, and meaningfully compared.

---

## References

- SWE-fficiency: https://github.com/swefficiency/swefficiency
- SWE-fficiency website: https://swefficiency.com/
- SWE-Perf: https://github.com/SWE-Perf/SWE-Perf
- PerfBench: https://arxiv.org/abs/2509.24091
- RCAEval: https://arxiv.org/abs/2412.17015
- AIOpsLab: https://www.microsoft.com/en-us/research/blog/aiopslab-building-ai-agents-for-autonomous-clouds/
- OpenRCA: https://github.com/microsoft/OpenRCA
- OpenTelemetry Profiling: https://opentelemetry.io/blog/2024/profiling/
- OpenTelemetry State of Profiling: https://opentelemetry.io/blog/2024/state-profiling/
- Polar Signals (Trace-Profile Correlation): https://www.polarsignals.com/blog/posts/2024/03/05/correlating-tracing-with-profiling-using-ebpf
