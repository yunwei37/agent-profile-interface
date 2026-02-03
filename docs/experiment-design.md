# 实验设计文档：Budgeted Profiling for LLM Software-Engineering Agents

本文档详细描述实验的具体设计、实现方案和执行计划。

---

## 1. 实验目标

验证以下核心假设：

1. **H1**: 在固定预算下，提供结构化 profiling 信息能显著提升 agent 的性能优化成功率
2. **H2**: Profiling 输出的表示形式（raw vs compressed vs differential）对 agent 效果有显著影响
3. **H3**: 存在最优的 profiling 预算阈值，超过该阈值会产生递减收益或负面效果
4. **H4**: 不同类型的性能瓶颈对 profiling 的敏感度不同

---

## 2. 实验平台选择

### 2.1 主实验平台：SWE-fficiency

**选择理由**：
- 498 个任务覆盖 9 个主流 Python 库（numpy, pandas, scikit-learn 等）
- 明确的性能优化任务定义（不是 bug 修复）
- 标准化的评测指标（Speedup Ratio）
- 完整的 CLI 工具链和容器化支持
- 任务描述明确要求 "profile or inspect execution, localize bottlenecks"

**评测指标**：
- `overall_score`: Harmonic mean of Speedup Ratio (SR)
- `proportion_incorrect`: 引入功能错误的比例
- `proportion_correct_but_no_speedup`: 正确但无加速的比例
- `proportion_human_speedup_or_better`: 达到或超过人类专家的比例

### 2.2 辅助验证平台

| 平台 | 语言 | 任务数 | 用途 |
|------|------|--------|------|
| SWE-Perf | Python | 140 | 验证结论泛化性 |
| PerfBench | .NET | 81 | 跨语言验证 |

---

## 3. 实例采样策略

### 3.1 采样方法：分层随机采样

```
总任务数: 498
仓库数: 9
每仓库采样: 10 个任务
总采样: 90 个任务
```

**分层依据**：
1. 按仓库分层（保证覆盖度）
2. 在每层内随机采样（避免选择偏差）

### 3.2 采样实现

```python
import random
from collections import defaultdict

def stratified_sample(instances, n_per_repo=10, seed=42):
    """分层随机采样"""
    random.seed(seed)

    # 按仓库分组
    by_repo = defaultdict(list)
    for inst in instances:
        repo = inst['repo']
        by_repo[repo].append(inst)

    # 每仓库采样
    sampled = []
    for repo, repo_instances in by_repo.items():
        if len(repo_instances) >= n_per_repo:
            sampled.extend(random.sample(repo_instances, n_per_repo))
        else:
            sampled.extend(repo_instances)  # 不足则全取

    return sampled
```

### 3.3 采样种子与可复现性

- **随机种子**: 42（固定）
- **采样列表**: 保存到 `data/sampled_instances.json`
- **版本控制**: 记录 SWE-fficiency 的 commit hash

---

## 4. Profiling 工具实现

### 4.1 Profiler 选择

| Profiler | 类型 | 开销 | 优势 |
|----------|------|------|------|
| **py-spy** | CPU sampling | <5% | 无需代码修改，可 attach |
| **memray** | Memory | ~10% | 详细的分配追踪 |
| **cProfile** | CPU deterministic | 10-30% | Python 内置 |
| **line_profiler** | Line-level | 高 | 行级精度 |

**主选方案**: py-spy（CPU）+ memray（Memory）

### 4.2 Profiling Tool Wrapper API

```python
class ProfilingTool:
    """Agent 可调用的 Profiling 工具接口"""

    def __init__(self, budget_config):
        self.max_invocations = budget_config.get('max_invocations', 5)
        self.max_duration_per_call = budget_config.get('max_duration', 30)  # seconds
        self.max_output_tokens = budget_config.get('max_tokens', 500)
        self.invocation_count = 0
        self.total_time = 0

    def profile(self, workload_cmd: str, duration: int = 10) -> dict:
        """
        执行 profiling 并返回结构化结果

        Args:
            workload_cmd: 要 profile 的命令
            duration: 采样持续时间（秒）

        Returns:
            {
                'success': bool,
                'summary': str,  # 根据 output_format 格式化
                'tokens_used': int,
                'time_used': float,
                'error': str or None
            }
        """
        # 检查预算
        if self.invocation_count >= self.max_invocations:
            return {'success': False, 'error': 'Budget exhausted: max invocations reached'}

        duration = min(duration, self.max_duration_per_call)

        # 执行 profiling
        raw_profile = self._run_pyspy(workload_cmd, duration)

        # 格式化输出
        summary = self._format_output(raw_profile)

        # 更新预算
        self.invocation_count += 1
        self.total_time += duration

        return {
            'success': True,
            'summary': summary,
            'tokens_used': len(summary.split()),
            'time_used': duration
        }

    def _run_pyspy(self, cmd, duration):
        """执行 py-spy 采样"""
        # 实际实现...
        pass

    def _format_output(self, raw_profile):
        """根据配置格式化输出"""
        # 实际实现见 4.3 节
        pass
```

### 4.3 输出格式设计

#### 格式 C1: Baseline（Raw 截断）

```
py-spy output (truncated to 4000 tokens):
Total samples: 10000
  %Own   %Total  Function (filename:line)
  25.3%  45.2%   inner_loop (module.py:123)
  18.7%  32.1%   compute_distance (utils.py:45)
  12.4%  28.9%   matrix_multiply (linalg.py:89)
  ...
  [truncated]
```

**Token 估计**: ~4000

#### 格式 C2: Top-K Compressed

```json
{
  "total_samples": 10000,
  "duration_seconds": 10.0,
  "hotspots": [
    {
      "rank": 1,
      "function": "inner_loop",
      "file": "module.py",
      "line": 123,
      "self_percent": 25.3,
      "cumulative_percent": 45.2
    },
    {
      "rank": 2,
      "function": "compute_distance",
      "file": "utils.py",
      "line": 45,
      "self_percent": 18.7,
      "cumulative_percent": 32.1
    }
    // ... top 10
  ]
}
```

**Token 估计**: ~500

#### 格式 C3: Hierarchical Call-paths

```json
{
  "hot_paths": [
    {
      "path": ["main", "process_data", "inner_loop"],
      "percent": 45.2,
      "leaf_file": "module.py:123"
    },
    {
      "path": ["main", "compute_metrics", "compute_distance"],
      "percent": 32.1,
      "leaf_file": "utils.py:45"
    },
    {
      "path": ["main", "finalize", "matrix_multiply"],
      "percent": 28.9,
      "leaf_file": "linalg.py:89"
    }
  ]
}
```

**Token 估计**: ~800

#### 格式 C4: Differential（Before/After）

```json
{
  "comparison": "before_patch vs after_patch",
  "changes": [
    {
      "function": "inner_loop",
      "file": "module.py:123",
      "before_percent": 25.3,
      "after_percent": 8.1,
      "delta": -17.2,
      "interpretation": "significantly reduced"
    },
    {
      "function": "new_optimized_func",
      "file": "module.py:150",
      "before_percent": 0.0,
      "after_percent": 12.4,
      "delta": +12.4,
      "interpretation": "new hotspot"
    }
  ],
  "overall_speedup": 2.1
}
```

**Token 估计**: ~600

---

## 5. Agent 配置

### 5.1 Base Agent 选择

**候选方案**：

| Agent | 优势 | 劣势 |
|-------|------|------|
| OpenHands | 开源，SWE-bench 主流 | 配置复杂 |
| SWE-agent | 专为代码任务设计 | 定制性有限 |
| Claude + Tool Use | 强推理能力 | API 成本 |
| Custom (LangChain) | 完全可控 | 开发成本 |

**建议方案**: OpenHands + 自定义 Profiling Tool

### 5.2 Agent 预算配置

```yaml
agent_budget:
  max_actions: 100          # 最大动作数
  max_wall_clock: 30        # 分钟
  max_tokens_input: 128000  # 输入 token 上限
  max_tokens_output: 8000   # 单次输出上限

profiling_budget:
  max_invocations: 5        # 最大 profiling 调用次数
  max_duration_per_call: 30 # 单次最大秒数
  max_output_tokens: 500    # 输出 token 上限
```

### 5.3 Tool 定义（OpenHands 格式）

```python
profiling_tool = {
    "name": "profile_workload",
    "description": """
    Profile the performance of a workload to identify CPU hotspots.
    Returns the top-K functions by CPU time with file locations.
    Use this to identify optimization targets before making changes.
    Budget: You can call this at most 5 times per task.
    """,
    "parameters": {
        "workload_command": {
            "type": "string",
            "description": "The command to run for profiling (e.g., 'python benchmark.py')"
        },
        "duration": {
            "type": "integer",
            "description": "Profiling duration in seconds (default: 10, max: 30)",
            "default": 10
        }
    }
}
```

---

## 6. 实验条件矩阵

### 6.1 主实验条件

| 条件 | Profiling | 输出格式 | 预算 | 目的 |
|------|-----------|----------|------|------|
| C0 | No | N/A | 0 | Baseline |
| C1 | Yes | Raw (truncated) | Medium | Raw 效果 |
| C2 | Yes | Top-K compressed | Medium | Compressed 效果 |
| C3 | Yes | Hierarchical | Medium | 结构化效果 |
| C4 | Yes | Differential | Medium | Diff 效果 |

### 6.2 预算消融实验

| 条件 | 预算设置 | 目的 |
|------|----------|------|
| B1 | max_invocations=2 | Low budget |
| B2 | max_invocations=5 | Medium budget |
| B3 | max_invocations=10 | High budget |

### 6.3 总实验运行数

```
主实验: 5 conditions × 90 instances = 450 runs
预算消融: 3 conditions × 90 instances = 270 runs (使用最佳格式)
重复运行: 3 repeats × 关键条件
---
总计: ~900 runs（估计）
```

---

## 7. 评测流程

### 7.1 单次实验流程

```
1. 初始化环境
   └── 启动 Docker 容器
   └── 加载 repo snapshot
   └── 配置 profiling tool

2. Agent 执行
   └── 加载任务描述
   └── Agent 迭代（最多 100 actions）
       ├── 读代码
       ├── 运行测试
       ├── [可选] 调用 profiling
       ├── 编辑代码
       └── 验证修改
   └── 输出 patch

3. 评测
   └── 应用 patch
   └── 运行正确性测试
   └── 运行性能测试（3 次取中值）
   └── 计算 Speedup Ratio

4. 记录
   └── 保存 trajectory
   └── 保存 metrics
   └── 保存 profiling 使用日志
```

### 7.2 SWE-fficiency CLI 集成

```bash
# 1. 生成 predictions
python run_agent.py \
    --instances data/sampled_instances.json \
    --condition C2 \
    --output predictions/C2.jsonl

# 2. 运行评测
swefficiency eval \
    --run_id eval_C2 \
    --prediction_path predictions/C2.jsonl \
    --num_workers 12

# 3. 生成报告
swefficiency report \
    --gold_run gold_baseline \
    --pred_run eval_C2 \
    --output_dir results/C2/
```

### 7.3 评测环境配置

```yaml
# docker-compose.yml
services:
  evaluator:
    image: swefficiency/evaluator:latest
    cpus: 4
    mem_limit: 16g
    volumes:
      - ./predictions:/predictions
      - ./results:/results
    environment:
      - CPU_PINNING=true
```

---

## 8. 数据收集与分析

### 8.1 收集的数据

```json
// trajectory_log.jsonl (每个 action 一行)
{
  "instance_id": "numpy__numpy-12345",
  "action_idx": 5,
  "action_type": "profile_workload",
  "timestamp": "2025-01-15T10:23:45Z",
  "input": {"workload_command": "python bench.py", "duration": 10},
  "output_tokens": 487,
  "wall_time_seconds": 12.3,
  "profiling_invocation_count": 2
}
```

```json
// result_summary.json (每个实例一个)
{
  "instance_id": "numpy__numpy-12345",
  "condition": "C2",
  "success": true,
  "speedup_ratio": 1.85,
  "expert_speedup": 2.10,
  "relative_sr": 0.88,
  "correctness": "pass",
  "total_actions": 47,
  "profiling_calls": 3,
  "profiling_tokens": 1461,
  "total_wall_time": 1234.5
}
```

### 8.2 统计分析计划

#### 主要比较（RQ1: Profiling 是否有效）

```python
from scipy import stats

# C0 vs C2 (best profiling format)
sr_c0 = results[results['condition'] == 'C0']['speedup_ratio']
sr_c2 = results[results['condition'] == 'C2']['speedup_ratio']

# Paired Wilcoxon test (non-parametric)
stat, p_value = stats.wilcoxon(sr_c2, sr_c0)

# Bootstrap CI for difference
def bootstrap_ci(a, b, n_boot=10000):
    diffs = []
    for _ in range(n_boot):
        idx = np.random.choice(len(a), len(a), replace=True)
        diffs.append(np.mean(a[idx]) - np.mean(b[idx]))
    return np.percentile(diffs, [2.5, 97.5])
```

#### 格式比较（RQ2: 哪种格式最好）

```python
# Friedman test (multiple conditions)
from scipy.stats import friedmanchisquare

sr_by_condition = [
    results[results['condition'] == c]['speedup_ratio'].values
    for c in ['C1', 'C2', 'C3', 'C4']
]
stat, p_value = friedmanchisquare(*sr_by_condition)

# Post-hoc: Nemenyi test
```

#### 预算曲线（RQ3: 最优预算）

```python
# Plot budget vs SR
import matplotlib.pyplot as plt

budgets = [2, 5, 10]
mean_sr = [results[results['budget'] == b]['speedup_ratio'].mean() for b in budgets]
std_sr = [results[results['budget'] == b]['speedup_ratio'].std() for b in budgets]

plt.errorbar(budgets, mean_sr, yerr=std_sr, marker='o')
plt.xlabel('Max Profiling Invocations')
plt.ylabel('Speedup Ratio')
```

### 8.3 失败模式分析（RQ5）

```python
def analyze_failure_modes(trajectories):
    """分析 profiling 相关的失败模式"""

    failure_modes = {
        'profiling_spam': 0,      # 过度调用 profiling
        'no_action_after': 0,     # profiling 后不行动
        'misattribution': 0,      # 优化了非关键代码
        'no_validation': 0        # 修改后未验证
    }

    for traj in trajectories:
        profile_calls = [a for a in traj if a['type'] == 'profile']
        edits = [a for a in traj if a['type'] == 'edit']

        # Profiling spam: >80% actions are profiling
        if len(profile_calls) / len(traj) > 0.8:
            failure_modes['profiling_spam'] += 1

        # No action after profiling
        for i, action in enumerate(traj):
            if action['type'] == 'profile':
                if i == len(traj) - 1 or traj[i+1]['type'] == 'profile':
                    failure_modes['no_action_after'] += 1
                    break

        # 其他分析...

    return failure_modes
```

---

## 9. 资源估算

### 9.1 计算资源

| 资源 | 单次运行 | 总需求 |
|------|----------|--------|
| CPU | 4 cores | 持续 |
| Memory | 16 GB | 持续 |
| GPU | 无 | - |
| 存储 | 10 GB/instance | ~1 TB |

### 9.2 API 成本估算

假设使用 Claude API:

```
每个 instance:
- 输入 tokens: ~50k (代码 + 任务描述 + 历史)
- 输出 tokens: ~5k
- Profiling 输出: ~1.5k × 3 calls = 4.5k

总 tokens/instance: ~60k
成本/instance: ~$0.90 (Claude Sonnet)

总实验:
- 900 runs × $0.90 = ~$810
- 加上重试和调试: ~$1,200
```

### 9.3 时间估算

```
单个 instance:
- Agent 执行: ~20 min
- 评测: ~10 min
- 总计: ~30 min

并行度: 12 workers
总 instances: 900
---
总时间: 900 × 30 / 12 / 60 = ~37.5 小时
加上配置和调试: ~50 小时
```

---

## 10. 风险与缓解

### 10.1 技术风险

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| Docker 环境不稳定 | 评测失败 | 使用 SWE-fficiency 官方镜像 |
| Profiling 开销过大 | 影响测量 | 使用 sampling profiler (py-spy) |
| Agent 行为不稳定 | 结果噪声 | 多次运行取中值 |
| API 限流 | 实验中断 | 实现 retry + backoff |

### 10.2 实验设计风险

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 预算不公平 | 结论偏差 | Profiling 开销计入总预算 |
| 采样偏差 | 不可泛化 | 分层随机采样 + 公开实例列表 |
| Reward hacking | 虚假结果 | 检查 patch 合理性 |
| 过拟合 benchmark | 不可泛化 | 辅助平台验证 |

### 10.3 缓解：Reward Hacking 检测

```python
def check_suspicious_patch(patch, test_results):
    """检测可疑的 patch"""

    suspicious = []

    # 检查是否修改了测试
    if 'test_' in patch or 'assert' in patch:
        suspicious.append('modified_tests')

    # 检查是否添加了缓存/记忆化
    if 'cache' in patch.lower() or '@lru_cache' in patch:
        suspicious.append('added_caching')

    # 检查是否删除了大量代码
    deletions = patch.count('-')
    additions = patch.count('+')
    if deletions > additions * 3:
        suspicious.append('excessive_deletion')

    return suspicious
```

---

## 11. 时间线

### Phase 1: 准备（2 周）

- [ ] 搭建实验环境
- [ ] 实现 Profiling Tool Wrapper
- [ ] 集成 Agent 框架
- [ ] 采样实例列表
- [ ] Pilot 测试（5 instances）

### Phase 2: 主实验（3 周）

- [ ] 运行 C0 baseline
- [ ] 运行 C1-C4 格式实验
- [ ] 运行 B1-B3 预算实验
- [ ] 收集 trajectories

### Phase 3: 分析（2 周）

- [ ] 统计分析
- [ ] 失败模式分析
- [ ] 可视化
- [ ] 撰写结果

### Phase 4: 验证（1 周）

- [ ] SWE-Perf 验证（子集）
- [ ] PerfBench 验证（子集）
- [ ] 敏感性分析

---

## 12. 可交付成果

1. **代码**
   - Profiling Tool Wrapper 实现
   - Agent 集成代码
   - 评测脚本
   - 分析脚本

2. **数据**
   - 采样实例列表（`data/sampled_instances.json`）
   - 原始结果（`results/raw/`）
   - 处理后数据（`results/processed/`）
   - Trajectories（`logs/trajectories/`）

3. **文档**
   - 本实验设计文档
   - 结果报告
   - 复现指南

4. **论文素材**
   - Figure 1: Budget vs Performance 曲线
   - Figure 2: Format comparison
   - Table 1: Main results
   - Table 2: Failure mode analysis
   - Table 3: Design principles

---

## 附录 A: 实例列表模板

```json
{
  "sampling_seed": 42,
  "sampling_date": "2025-01-15",
  "swefficiency_commit": "abc123",
  "instances": [
    {"instance_id": "numpy__numpy-12345", "repo": "numpy"},
    {"instance_id": "pandas__pandas-67890", "repo": "pandas"},
    // ...
  ]
}
```

## 附录 B: 配置文件模板

```yaml
# config/experiment.yaml
experiment:
  name: "budgeted_profiling_v1"
  seed: 42

sampling:
  n_per_repo: 10

agent:
  type: "openhands"
  model: "claude-3-sonnet"
  max_actions: 100
  max_wall_clock_minutes: 30

profiling:
  enabled: true
  format: "top_k"  # raw, top_k, hierarchical, differential
  budget:
    max_invocations: 5
    max_duration_per_call: 30
    max_output_tokens: 500

evaluation:
  workers: 12
  cpu_pinning: true
  memory_limit_gb: 16
  repeats: 3
```

## 附录 C: 检查清单

### 实验前

- [ ] SWE-fficiency 环境测试通过
- [ ] Profiling tool 单元测试通过
- [ ] Agent 能完成简单任务
- [ ] 日志记录正常工作
- [ ] 预算控制正常工作

### 实验中

- [ ] 每 10 个实例检查一次结果
- [ ] 监控 API 使用量
- [ ] 备份中间结果
- [ ] 记录异常情况

### 实验后

- [ ] 所有结果文件完整
- [ ] 统计分析可复现
- [ ] 可视化清晰准确
- [ ] 文档更新完成
