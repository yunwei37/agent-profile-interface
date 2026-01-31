下面我按“**survey paper**（可直接扩写成论文结构）”的方式，把你提到的 **AIOps/LLM-RCA（logs/metrics/traces）**、以及“让 AI/agent 去定位/修复性能问题”的 **benchmarks/datasets** 全部拉通：它们各自的 **research question（RQ）** 是什么、**评测怎么做**、以及它们在 **profiling（CPU/lock/heap）/profiling tool interface** 维度到底缺了什么。最后给你一条**workshop 级别**最小切口：不做大而全框架、也不一定新建 benchmark，只在现有 bench 上提出新的 RQ + 一个 agent-friendly、预算可控的 profiling tool API，就足够写一篇“被 cite 的论文”。

---

## 0. 一句话结论

* **主流 AIOps / LLM-RCA benchmark** 明确把“telemetry”定义成 **logs/metrics/traces**，并围绕这三类做任务与评测（OpenRCA、RCAEval、AIOpsLab 都是这个范式）。([OpenReview][1])
* **性能优化/性能 bug 修复 benchmark**（SWE-fficiency / SWE-Perf / GSO / PerfBench）虽然在任务上逼你做性能工程，但评测信号通常只落在“**性能测试/运行时** + correctness tests”，并没有把 **profiling 作为一等公民**的 agent 工具接口（最多给你 shell，让你自己跑 profiler）。([SWE-fficiency][2])
* **OpenTelemetry 已经把 profiles 作为第四种信号**并强调与 logs/metrics/traces 的双向关联，这是你做“agent-friendly profiling API / tool IR”的天然落点：你可以站在正在成形的标准数据模型（pprof 可映射）之上做接口与评测。([OpenTelemetry][3])

---

## 1. 你要写的 survey 的“问题定义”

你要研究的不是“profile agent”，而是：

> **Agent-driven Profiling for Debugging/Optimization**：让 LLM agent 在预算约束下，主动选择**何时、对哪里、以什么粒度**采集 CPU/lock/heap profiles，并把 profile 与其它信号（tests、traces、metrics、logs）关联起来，来完成**定位 + 修复/优化**。

这里的关键是：profiling 不只是“多一个信号”，而是 **一个交互式工具**，要解决 3 个工程现实：

1. **预算可控**：profiling 有开销（运行时、采样频率、数据体量），而 agent 的 token/context 也有限。
2. **输出可消费**：profiling 原始输出（pprof/flamegraph/诊断日志）通常“>10k tokens 级别”，agent会被淹没（PerfBench 就明确遇到这个问题并专门做了输出裁剪）。([arXiv][4])
3. **评测可对齐**：性能评测易受噪声/环境影响，且存在 reward hacking（GSO 发现大量模型会“骗测评”并引入 HackDetector）。([gso-blog on Notion][5])

---

## 2. Benchmark 地图（按任务类型分类）

下面这张“地图”基本覆盖你现在需要 survey 的范围（如果你写 workshop/short paper，用这套 taxonomy 就够了）。

### A. Incident RCA（以 logs/metrics/traces 为核心信号）

* **AIOpsLab**：AgentOps 设想 + Orchestrator/Agent-Cloud-Interface，系统提供 observability 的 telemetry（metrics/traces/logs），任务覆盖 incident 生命周期。([Microsoft GitHub][6])
* **OpenRCA**：335 个 failure cases，>68GB telemetry data（logs/metrics/traces），评估 LLM 做 RCA。([OpenReview][1])
* **RCAEval**：735 个 failure cases，收集 metrics/logs/traces，提供评测环境与 baselines。([arXiv][7])

> 共性：benchmark 目标是“从三大信号中推 root cause”，**profiling 信号缺位**（至少不是 benchmark 的任务接口或评测核心）。([OpenReview][1])

### B. Repo-scale 性能优化（真实 repo + 性能测试/工作负载）

* **SWE-fficiency**：498 个任务、9 个 Python 大库；给完整代码库 + 真实 workload + unit tests；指标是 **Speedup Ratio (SR) = (model speedup)/(expert speedup)**，并给出容器化、CPU/mem pinning、3 小时/任务、100 actions 的评测设定。([SWE-fficiency][2])
* **SWE-Perf**：从 10 万+ PR 过滤出 140 个性能优化实例；每个实例包含 repo、目标函数、performance-related tests、expert patch、Docker 环境与 runtime metrics；评测设计成 Apply / Correctness / Performance 三层，并强调对原始与打补丁后的 runtime 做可比对重测。([OpenReview][8])
* **GSO**：以优化任务衡量 SWE-agent 的性能工程能力；指标 **Opt@1**（达到 ≥95% human speedup 且通过 correctness tests），并引入 **HackDetector** 来惩罚 reward hacking。([GSO Bench][9])

> 共性：任务逼你做性能工程，但“profiling”更多只是隐含在“你可以自己跑”的工作流里，没有一个**标准化、agent-friendly 的 profiling 工具接口**。

### C. 性能 Bug 修复（issue/bugfix 视角，而不是 PR 优化）

* **PerfBench**：专门评估 agent 修“真实世界性能 bugs”；他们把 agent 工作流改成“先建 BenchmarkDotNet + MemoryDiagnoser，再优化，再复测”，并且因为 benchmark 输出过长，做了**输出解析与裁剪**：成功时只保留 summary table，声称 token 使用量降 >90%。([arXiv][4])

> 这是你写“agent-friendly profiling tool API”的一个非常强的对照案例：PerfBench 已经证明“**工具输出形态**”本身就是 agent 成败关键（不是模型大小 alone）。

### D. 代码级/算法级效率 benchmark（多为 snippets，不是复杂 repo）

* **PIE（Learning Performance-Improving Code Edits）**：从大量 C++ 提交对里抽取“慢→快”的 human edits；核心贡献之一是用 **gem5 全系统模拟**减少性能测量噪声，从而能可靠评估 speedup。([arXiv][10])
* **EFFIBENCH**：1000 个 LeetCode“效率敏感题”，给 canonical 最优解；提供执行时间、峰值内存、总内存使用等多指标，并报告 LLM 代码相对人类最优解的差距。([NeurIPS Proceedings][11])
* **Mercury**：强调“只看 Pass（正确性）不足以衡量效率”，提出 **Beyond** 之类综合指标来同时评价正确性与效率，并讨论训练/对齐。([NeurIPS Proceedings][12])
* **ECCO**：关注“提高效率不牺牲正确性”的困难，提出可复现实验平台（固定云实例 + Judge0 沙箱执行）来稳定评估 runtime/memory，并系统比较 ICL/迭代反馈/finetune；结论是很多方法会牺牲正确性，execution 信息有助于保正确性。([OpenReview][13])

> 这些工作在“稳定评测效率”上做了很多方法学，但它们的对象多是 snippets；你做 repo-scale / 系统 profiling 工具接口，切口不同。

### E. 性能数据集（挖掘/推荐/训练，而非在线 agent 诊断）

* **PerfCurator**：从 GitHub 大规模挖 performance bug-fix commits，给出多语言 commit 规模（Python/C++/Java 都是十万量级），用于训练/检测。([arXiv][14])
* **PerfLens（SOAP’21）**：做“性能改进推荐”，并明确指出加入 profiler data 后推荐准确率显著提高（90% vs 55%）。([Microsoft][15])

> PerfLens 这条证据对你特别有用：它说明 **profiling 信息是提升诊断/优化质量的关键特征**，但现在的 LLM agent benchmarks 却普遍没把 profiling 做成可消费的任务接口。

---

## 3. 这些 benchmark 的核心 RQ 与评测方式（逐个点名）

我把“论文在问什么 + 怎么测”的最关键部分抽出来（适合你直接写进 survey 的表格或 related work）。

### 3.1 AIOpsLab

* **RQ（隐含）**：如何定义/评估 AgentOps（自治云运维）任务全流程？如何把 agent 接入可部署系统，并用可重复的方式评估？([Microsoft GitHub][6])
* **评测**：Orchestrator 提供任务与 APIs；系统 observability 给 metrics/traces/logs；最终用预定义任务指标评估 agent 解决情况。([Microsoft GitHub][6])
* **与 profiling 的关系**：profiling 不在“telemetry 定义”的核心范式里。

### 3.2 OpenRCA

* **RQ**：LLM 能否在“真实软件运行场景”里，从大体量 telemetry（logs/metrics/traces）里定位 root cause？([OpenReview][16])
* **评测**：给自然语言 query + 大体量 telemetry，评估是否定位到正确 root cause 元素（具体指标你写 survey 时可再展开，但核心仍是 L/M/T）。([OpenReview][1])

### 3.3 RCAEval

* **RQ**：如何为微服务 RCA 提供统一 benchmark + 多源 telemetry（metrics/logs/traces）+ 可复现实验环境和 baselines？([arXiv][7])
* **评测**：数据集带标注 root cause service / indicators，框架复现多种 RCA baselines。([arXiv][7])

---

### 3.4 SWE-fficiency

* **RQ**：LLM/agent 能否在真实 repo + 真实 workload 上，完成接近人类性能工程师的优化？([SWE-fficiency][2])
* **评测**：

  * 指标：**SR = (model speedup)/(expert speedup)**，harmonic mean 聚合；SR=1 表示达人类。([SWE-fficiency][2])
  * 任务：完整 repo snapshot + workload script + unit tests；强调“profiling/定位/相关 tests/迭代优化”的真实工作流。([SWE-fficiency][2])
  * 设定：容器化 + CPU/mem pinning（4 vCPU/16GB）、3 小时/任务、100 actions。([SWE-fficiency][2])
* **关键点**：它非常适合你做“agent-friendly profiling tool API”论文，因为它的任务描述就把 profiling 当成必经步骤，但**工具接口层面仍是 shell + 人类式工作流**。

### 3.5 SWE-Perf

* **RQ**：LLM 在“真实仓库尺度”的性能优化能力到底如何？如何构建高质量、稳定、可重复的性能优化 benchmark？([SWE-Perf][17])
* **评测**：

  * 数据：140 实例；包含 repo、target functions、performance-related tests、expert patch、Docker env、runtime metrics。([OpenReview][8])
  * 评测框架：Apply / Correctness / Performance 三层；并强调测试时要**重测原始 codebase runtime**确保可比性。([OpenReview][8])
  * 稳定性：数据构建阶段强调 warm-up、重复执行、过滤 outliers、统计验证稳定收益（你写方法学分析时，这段非常能用）。([OpenReview][8])

### 3.6 GSO

* **RQ**：SWE-agent 在“性能优化”这种需要工程推理与实测的任务上能力如何？如何构建任务/评测，避免模型钻空子？([OpenReview][18])
* **评测**：

  * 指标：Opt@1 = 单次尝试达到 ≥95% human speedup 且通过 correctness tests。([GSO Bench][9])
  * 额外：HackDetector（rubric + LLM 判别）惩罚 reward hacking，声称模型存在大量“通过测试但作弊”的情况。([gso-blog on Notion][5])
* **与你的主题强相关的点**：GSO 证明了“只靠 tests + speedup”会被攻击，你的 profiling tool API 设计如果不考虑**anti-hacking / provenance / sandbox**，评测也可能被玩坏。

### 3.7 PerfBench

* **RQ**：agent 能不能修复真实的性能 bug？这种任务与“功能 bug 修复”相比到底难在哪？([arXiv][4])
* **评测**：

  * 工作流：指导 agent 建 benchmark（BenchmarkDotNet + MemoryDiagnoser），复现并测 CPU/memory，再优化再复测。([arXiv][4])
  * 工具输出：对 benchmark 输出做解析裁剪；成功时只保留 summary table，报告 token 降 >90%。([arXiv][4])
* **与你的主题直接对齐**：这几乎就是“agent-friendly performance tooling”的先例，只是它聚焦在 benchmark 输出，而不是 profiles。

---

## 4. Profiling 标准与生态：为什么你现在做接口/工具有“顺势 novelty”

### 4.1 OpenTelemetry 已明确把 Profiles 作为核心信号，并强调与其它信号的关联

* OTel 宣布支持 profiling，并明确 **profiles 将与 logs/metrics/traces 建立双向链接**（能从一个信号跳转到另一个）。([OpenTelemetry][3])
* OTel 的 profiles 数据模型强调与 **pprof 可双向映射**（可把现有生态的 profiler 输出纳入统一模型）。([OpenTelemetry][19])
* OTel 在 2025 的稳定性讨论里已经把 signals 叙述成“四件套：tracing/metrics/logs/profiles”。([OpenTelemetry][20])

### 4.2 工业界/开源生态正在把“trace ↔ profile”打通（你可以把它当作 tool API 的现实基础）

* Parca/Polar Signals 展示过 **用 eBPF 做 tracing 与 profiling 的关联**（按 trace ID 查询跨服务 CPU profile）。([Do more with less. | Polar Signals][21])
* Grafana/Pyroscope 把 “Span Profiles / traces-to-profiles” 作为产品级功能与文档。([Grafana Labs][22])
* Elastic 也在讲“profiling + tracing correlation”帮助从慢 trace 直接定位到代码行级瓶颈。([Elastic][23])

**对你的论文写法建议**：把这些当成“**profiling 已成为 observability 的标准信号之一**”的外部趋势证据，你做 agent-friendly API 属于“把标准能力转成 agent 可用的工具形态”，不是空想。

---

## 5. 现在的 research gap 到底在哪里（把话说窄、说清楚）

你前面说的 Gap A（profiling 在 agent 评测与接口中缺位）基本成立；survey 写法上，我建议你把 gap 拆成 3 个更“可发 workshop”的小 gap（每个都能形成独立 RQ）：

### Gap-1：Bench 里“profiling=隐含步骤”，但**没有 agent-consumable 的 profile 输出协议**

* SWE-fficiency 明确要求 agent “profile workloads, localize bottlenecks…”，但评测接口给的还是 bash/文件编辑这类“人类工具”。([SWE-fficiency][2])
* PerfBench 证明原始性能输出会淹没 LLM，于是他们做了输出裁剪；profiling 输出更复杂，问题更严重。([arXiv][4])

**可形成 RQ**：

> 在相同任务预算下（actions/time/tokens），哪种 profiling 输出摘要形态最能提升 agent 的“定位质量/优化收益”？

### Gap-2：缺乏“预算可控”的 profiling 工具语义（不仅是开关 profiler）

* profiling 的预算维度很多：采样频率/持续时间/是否分层（进程→线程→函数→行）/是否采集 heap/lock/alloc/是否跨服务。现有 benchmark 很少把它做成可调 knob。
* SWE-Perf/PIE/ECCO 等强调“稳定评测”的方法学，但更多在评测 pipeline，而不是 agent 的交互工具预算。([OpenReview][8])

**可形成 RQ**：

> agent 应该如何分配 profiling budget（少量多次 vs 一次重采样）才能在噪声与开销中取得最优信息增益？

### Gap-3：缺乏“抗 reward hacking”的性能诊断工具链设计

* GSO 发现大量 reward hacking，引入 HackDetector；说明“性能评测基础设施”是会被模型攻击的。([gso-blog on Notion][5])

**可形成 RQ**：

> profiling tool API 的最小安全/可信约束是什么（只读、隔离、签名、重放、不可篡改元数据）才能支持可靠评测？

---

## 6. 你能做什么：workshop paper 不需要新 benchmark 的三种可行路线

你完全可以**不造新 benchmark**，只做“benchmark augmentation + 新 RQ + 实证分析”，就能写一篇有价值的 workshop/short paper。

### 路线 A：选 SWE-fficiency（最对题）

**为什么**：任务描述天然包含 profiling，指标 SR 与人类 expert 对齐，且 repo/workload 很贴近真实性能工程。([SWE-fficiency][2])

**你做的贡献（最小化）**：

1. 设计一个 **agent-friendly profiling tool API**（只做 CPU + alloc 两种也行），核心是“结构化摘要输出 + budget knobs”。
2. 把它接到 SWE-fficiency 的 agent scaffolds（他们提到用 OpenHands/SWE-agent 框架评测）。([SWE-fficiency][2])
3. 做 2~3 个非常清晰的 RQ 实验（见下一节），输出结论 + failure analysis。

### 路线 B：选 PerfBench（最能写清“输出可消费”）

**为什么**：PerfBench 已经证明“性能工具输出裁剪”是关键变量（>90% token reduction）；你做 profiling API 属于同一类问题的自然延伸。([arXiv][4])

**你做的贡献**：

* 把 profiling 的原始输出（例如 pprof/ETW/热点列表）变成“summary table + top-k hot path + diff(before/after)”；并系统评估“裁剪策略 vs 诊断成功率”的 trade-off。

### 路线 C：选 GSO（能讲清 anti-hacking）

**为什么**：它把“作弊/环境劫持”摆到台面上，还公开 HackDetector 思路。([gso-blog on Notion][5])

**你做的贡献**：

* 提出“profiling API 的可信执行/不可篡改输出”设计，并用 GSO 的 hacking 案例做安全性动机（哪怕你只做概念性 prototype + 小规模实验，也能成文）。

---

## 7. Workshop 论文的推荐切入点：3 个“窄 RQ”，不做大而全框架

下面这 3 个问题，我认为最适合 workshop：每个都能用现有 benchmark 做实验，不需要造新数据集。

### RQ1：**Profile 摘要的“接口形态”如何影响 agent 成功率？**

对比几种 API 输出：

* A) 原始 profiler 输出（baseline，通常爆 token）
* B) top-k functions + self-time/cum-time 表格（最经典）
* C) hot path（top-k stacks）+ 与 workload phase 的对齐（更像 flamegraph 的结构化版）
* D) diff profiles（before vs after）只输出“增长/下降最大”的 top-k（非常 agent-friendly）

**评估指标**：

* SWE-fficiency：SR 的提升幅度（或 SR@budget），以及“是否命中 expert 修改文件/函数”的定位质量。([SWE-fficiency][2])
* PerfBench：success rate + CPU/memory improvement，并统计 token/step。([arXiv][4])

### RQ2：**预算控制策略：固定预算 vs 自适应预算，哪个更划算？**

预算 knobs 示例：

* 最大 profiling 次数 N
* 每次持续时间 T、采样频率 f
* 允许的 overhead 上限（比如额外 runtime ≤ X%）
* 输出 token 上限（比如 ≤ 2k tokens）

自适应策略例子：

* 先低成本采样（短 T、低 f）定位 top suspects → 再对 suspects 做高分辨率 profile。
* 只有当“性能测试显示改善/退化”超过阈值才做 diff profile（避免浪费预算）。

**评估指标**：在固定 wall-clock / actions / tokens 下，诊断/优化收益最大化。

### RQ3：**Profiling 工具链如何抵御 reward hacking / harness hijacking？**

把 GSO HackDetector 的观察转成你的设计约束：

* profiling 工具是否只读（agent 不能改 profiler/harness）
* profile 输出是否可验证（hash/signature/recorded provenance）
* 是否做多次重跑/随机种子 workload（降低硬编码/记忆化作弊）
  这类设计非常“系统味”，即便 workshop 也很容易被认可。([gso-blog on Notion][5])

---

## 8. 你可以写的一条“完整 research story”（偏 workshop，但可扩展成正式 paper）

下面是一条我认为**最有机会落地**、且“讲得清楚”的故事线：

### 标题候选

**Budgeted Profiling Tools for LLM Agents: An Agent-Friendly API atop OpenTelemetry Profiles**

### 动机（Problem）

1. 现有 agent 评测在 RCA 侧基本停留在 logs/metrics/traces（OpenRCA、RCAEval、AIOpsLab）。([OpenReview][1])
2. 但性能工程的“决定性证据”常来自 CPU/alloc/lock profiles；PerfLens 甚至定量显示 profiler data 能显著提高性能改进推荐准确率。([Microsoft][15])
3. 新问题：就算 benchmark 需要性能优化（SWE-fficiency / SWE-Perf / GSO），agent 仍缺一个“**可控预算、可消费输出**”的 profiling 工具接口；PerfBench 已经证明“输出裁剪”直接影响 agent 表现。([SWE-fficiency][2])

### 关键观察（Opportunity）

OpenTelemetry 已经把 profiles 作为核心信号，并强调与 traces/logs/metrics 的关联；pprof 也可映射进 OTel profiles 数据模型。([OpenTelemetry][3])
产业界也在做 trace↔profile correlation（Parca、Pyroscope、Elastic）。([Do more with less. | Polar Signals][21])

### 你的主张（Thesis）

> **profiling 的关键 research 不是“能不能采样”，而是“agent 该如何在预算下问问题 + 消费答案”。**
> 因此需要一个 **agent-friendly、预算可控的 profiling tool API/IR**，把 profiles 变成类似“结构化查询结果”，而不是 raw flamegraph。

### 方案（Approach）——不用大而全，做最小原语集合

API 只需覆盖 4 类原语（你在 workshop 里把它当“design proposal + 实证”即可）：

1. `profile.capture(workload, budget)`：一次性采集（预算=时间/频率/开销上限）。
2. `profile.summary(profile, k, schema=table|hotpaths)`：输出 top-k hotspots（严格 token 上限）。
3. `profile.diff(profileA, profileB, k)`：输出“变化最大”的 hotspots（对优化迭代尤其关键）。
4. `profile.link(context)`：可选，把 profile 与 trace/span/resource 关联（站在 OTel profiles 的语义上写）。([OpenTelemetry][3])

### 挑战与解决（Challenges → Solutions）

* **输出爆炸**：借鉴 PerfBench 的思路，默认只返回 summary/diff；失败时才允许返回更完整日志。([arXiv][4])
* **噪声与可比性**：参考 SWE-Perf 强调“重测原始 runtime”与多次重复/统计稳定性的方法学，profiling 也需要“可比对重跑”的协议。([OpenReview][8])
* **reward hacking**：借鉴 GSO HackDetector 的动机，设计只读/隔离/可验证输出的 tool contract，并在实验中报告是否出现“可疑优化”（例如修改测试/缓存作弊）。([gso-blog on Notion][5])

### 评估（Evaluation Plan）

选 **SWE-fficiency**（主）+ **PerfBench**（辅）就能写出非常完整的 workshop 实证：

* **主指标**：

  * SWE-fficiency：SR（以及 SR@fixed tokens/actions/time）。([SWE-fficiency][2])
  * PerfBench：success rate、CPU/memory improvements、token/step 成本。([arXiv][4])
* **消融**：summary vs diff、top-k 的 k 值、预算分配策略。
* **诊断质量**：agent 选择的 hot functions 是否更接近 expert patch 的修改位置（SWE-fficiency 明确讨论模型经常选错函数/文件，这正好做量化分析）。([SWE-fficiency][2])

---

## 9. 你写 survey + workshop paper 的落笔建议

### survey 里你要“点明的结论”（可以作为结论段落）

1. RCA 类 benchmark 的主流范式仍是 L/M/T；profiling 在任务接口与评测中缺位。([OpenReview][1])
2. 性能优化类 benchmark 已经把“真实 repo + 性能收益”做成可重复评测（SWE-fficiency、SWE-Perf、GSO），并逐步面对“输出可消费/作弊”这类 agent 工具链问题（PerfBench 的 token 裁剪、GSO 的 HackDetector）。([SWE-fficiency][2])
3. OTel profiles 作为新标准信号提供了“统一接口语义”的外部条件，你可以把“agent-friendly profiling API”写成一个**标准化落地**而不是“另起炉灶”。([OpenTelemetry][3])

### workshop paper 最好避免的坑

* 不要承诺做“统一 AIOps 框架 + 新 benchmark + 新 agent”三件套；那会过大。
* 只做 “我们做了个 profiler wrapper”也不够研究：要用 **RQ + 对照实验**把“接口形态/预算策略/抗作弊约束”这些问题讲清楚。

---

如果你愿意把这进一步变成可投稿的文稿，我建议你下一步就做两件事（都不需要新 benchmark）：

1. 选定主战场：**SWE-fficiency**（最贴“profiling-driven repo optimization”）或 **PerfBench**（最贴“agent 工具输出设计”）。([SWE-fficiency][2])
2. 把上面 RQ1/RQ2/RQ3 里挑 **两个**做扎实（实验 + failure analysis + 设计建议），一篇 workshop 基本就成型了。

[1]: https://openreview.net/forum?id=M4qNIzQYpd "https://openreview.net/forum?id=M4qNIzQYpd"
[2]: https://swefficiency.com/ "https://swefficiency.com/"
[3]: https://opentelemetry.io/blog/2024/profiling/ "https://opentelemetry.io/blog/2024/profiling/"
[4]: https://arxiv.org/html/2509.24091v2 "https://arxiv.org/html/2509.24091v2"
[5]: https://gso-blog.notion.site/gso-hackdetector "https://gso-blog.notion.site/gso-hackdetector"
[6]: https://microsoft.github.io/AIOpsLab/ "https://microsoft.github.io/AIOpsLab/"
[7]: https://arxiv.org/html/2412.17015v5 "https://arxiv.org/html/2412.17015v5"
[8]: https://openreview.net/pdf?id=KxFaKvtBiG "https://openreview.net/pdf?id=KxFaKvtBiG"
[9]: https://gso-bench.github.io/leaderboard.html "https://gso-bench.github.io/leaderboard.html"
[10]: https://arxiv.org/abs/2302.07867 "https://arxiv.org/abs/2302.07867"
[11]: https://papers.nips.cc/paper_files/paper/2024/file/15807b6e09d691fe5e96cdecde6d7b80-Paper-Datasets_and_Benchmarks_Track.pdf "https://papers.nips.cc/paper_files/paper/2024/file/15807b6e09d691fe5e96cdecde6d7b80-Paper-Datasets_and_Benchmarks_Track.pdf"
[12]: https://proceedings.neurips.cc/paper_files/paper/2024/file/1df1df43b58845650b8dada00fca9772-Paper-Datasets_and_Benchmarks_Track.pdf "https://proceedings.neurips.cc/paper_files/paper/2024/file/1df1df43b58845650b8dada00fca9772-Paper-Datasets_and_Benchmarks_Track.pdf"
[13]: https://openreview.net/pdf?id=YP8QNMaAhq "https://openreview.net/pdf?id=YP8QNMaAhq"
[14]: https://arxiv.org/abs/2406.11731 "https://arxiv.org/abs/2406.11731"
[15]: https://www.microsoft.com/en-us/research/publication/perflens-a-data-driven-performance-bug-detection-and-fix-platform/ "https://www.microsoft.com/en-us/research/publication/perflens-a-data-driven-performance-bug-detection-and-fix-platform/"
[16]: https://openreview.net/pdf?id=M4qNIzQYpd "https://openreview.net/pdf?id=M4qNIzQYpd"
[17]: https://swe-perf.github.io/ "https://swe-perf.github.io/"
[18]: https://openreview.net/forum?id=I5qDL315bQ&referrer=%5Bthe+profile+of+Naman+Jain%5D%28%2Fprofile%3Fid%3D~Naman_Jain2%29 "https://openreview.net/forum?id=I5qDL315bQ&referrer=%5Bthe+profile+of+Naman+Jain%5D%28%2Fprofile%3Fid%3D~Naman_Jain2%29"
[19]: https://opentelemetry.io/docs/specs/semconv/general/profiles/ "https://opentelemetry.io/docs/specs/semconv/general/profiles/"
[20]: https://opentelemetry.io/blog/2025/stability-proposal-announcement/ "https://opentelemetry.io/blog/2025/stability-proposal-announcement/"
[21]: https://www.polarsignals.com/blog/posts/2024/03/05/correlating-tracing-with-profiling-using-ebpf "https://www.polarsignals.com/blog/posts/2024/03/05/correlating-tracing-with-profiling-using-ebpf"
[22]: https://grafana.com/docs/pyroscope/latest/configure-client/trace-span-profiles/ "https://grafana.com/docs/pyroscope/latest/configure-client/trace-span-profiles/"
[23]: https://www.elastic.co/observability-labs/blog/continuous-profiling-distributed-tracing-correlation "https://www.elastic.co/observability-labs/blog/continuous-profiling-distributed-tracing-correlation"
