# Deep Research Agent Harness 技术说明

## 1. 系统边界

本项目以 Deep Research Agent Harness 为核心，本地 Hybrid RAG 是可插拔检索后端之一：

- **Deep Research Harness**：DAG 规划、动态角色编组、并发和预算控制、共享记忆、证据验证、Red/Blue 修复、失败恢复和评测。
- **本地 Hybrid RAG**：文档解析、Chunk、Embedding、FAISS、BM25、RRF、相关性过滤、Context 和来源输出；既可独立运行，也可注册为 Harness 的 Retrieval Backend。

Harness 通过 `LocalHybridSearchBackend` 复用现有本地检索，也可以使用 `TavilySearchBackend`，或通过 `CompositeSearchBackend` 并发执行本地、Web、论文类检索工具并用加权 RRF 融合。编排器只依赖统一的 `RetrievalTool` 契约，不与具体检索实现绑定。

## 2. Harness 契约

`src/research_engine/harness.py` 定义四个核心对象：

- `AgentSpec`：声明角色、工具、依赖和元数据。
- `RunContext`：隔离单次运行的配置、产物和生命周期事件。
- `ToolRegistry`：注册具名检索或执行能力。
- `AgentRuntime`：负责调用 Agent；默认实现为当前进程内的 `LocalAgentRuntime`。

旧 Worker 通过兼容适配器接入；没有实现角色专用入口时，Runtime 自动回退到原有 `run(payload)`。因此替换远程 Worker、容器执行环境或检索后端时，不需要改写 Planner、Orchestrator 和评测层。

## 3. 规划与生命周期

Planner 输出结构化 `ResearchPlan`：

1. 模型优先返回 JSON。
2. 解析失败时依次尝试代码块提取、对象片段提取和启发式计划。
3. 校验任务 ID、依赖存在性和 DAG 无环性。
4. 批量失败触发 replan 时，新节点不得与已执行节点冲突。

每个任务使用九状态生命周期：

```text
pending → ready → running → succeeded
                         ├→ failed → degraded
                         ├→ timed_out → degraded
                         ├→ skipped
                         └→ cancelled
```

`orchestrator.py` 使用 `asyncio + Semaphore` 按拓扑并发执行。状态转换、计划版本、失败原因和降级动作全部写入 `ResearchResult.trace`。

## 4. Dynamic Swarm 与预算控制

`HeuristicSwarmPolicy` 根据问题长度、DAG 规模、比较/时效/风险关键词和约束数量计算复杂度：

| 档位 | 角色 | Agent 数量 |
|---|---|---:|
| simple | Researcher | 1 |
| standard | Scope、Evidence、Counter | 3 |
| complex | Scope、Evidence、Counter、Impact、Evidence Verifier | 5 |

当前 `heuristic-v4` 阈值为：

- `simple < 0.23`
- `complex >= 0.72`
- 中间区间为 standard

`BudgetController` 同时跟踪：

- Worker 调用数
- replan 次数
- Claim 验证查询数
- Red/Blue 轮次
- 已用时间与 deadline
- 质量效用和代理成本单位
- 连续低收益与证据停滞

停止原因包括 `review_passed`、`evidence_saturated`、`diminishing_returns`、预算耗尽和全局 deadline。`ResearchConfig` 是策略不可突破的硬上限。

## 5. 角色差异化检索

`ResearchWorker.run_for_agent` 为不同角色生成不同检索意图：

- Scope：定义、适用范围和边界。
- Evidence：官方文档、标准、论文和直接证据。
- Counter：反例、限制、争议和相反证据。
- Impact：风险、成本、影响和落地权衡。
- Evidence Verifier：针对具体 Claim 的直接支持或反驳证据。

候选结果按以下信号评分：

1. 查询关键词覆盖
2. 实体覆盖
3. 搜索后端分数
4. 来源权威度
5. 角色内容匹配度

随后执行低相关页面淘汰、规范 URL 去重、全文去重和同域近重复去重。政府/法规、标准、论文和官方文档优先于普通网页。

每次过滤都产生 `retrieval_filter_applied` Trace，记录过滤前后数量、淘汰原因、URL、命中关键词/实体和实际阈值，不把页面正文复制到 Trace。

### 已完成的角色阈值校准

ResearchBench-Live v1.0 包含 15 题，覆盖 simple / standard / complex 三档。正式检索校准使用 45 个角色检索单元，每单元最多 10 个候选。

对 33 条弱标签误杀候选完成人工审核后，Retrieval-Gold v1.0 确认 18 条真正误杀、15 条正确淘汰。Gold 阈值如下：

| 角色 | 阈值来源 | 最低相关性阈值 |
|---|---|---:|
| Researcher | Proxy | 0.14 |
| Scope Researcher | Gold | 0.44 |
| Evidence Researcher | Gold | 0.48 |
| Counter Researcher | Gold | 0.28 |
| Impact Analyst | Gold | 0.42 |
| Evidence Verifier | Proxy | 0.42 |

Gold 切片仅来自“被过滤候选”，因此它适合校准误杀，不代表总体检索 Precision/Recall 的无偏估计。

OpenAlex 已纳入 Frozen/Live 学术检索问题、官方资料检索和后端对比评测；当前运行时通过统一 Web/Composite 契约检索 OpenAlex 资料，不把学术来源逻辑耦合到编排器。

## 6. 共享记忆与上下文压缩

`SharedMemory` 使用 SQLite 持久化跨 Agent 证据：

- 写前进行语义去重。
- 结合启发式反义词和语义对立检测矛盾。
- 使用多策略选择保留、替换或并存。
- 保存来源、角色、时间和检索元数据。

上下文采用三级压缩：

1. L1 Embedding 粗过滤
2. L2 TextRank 细筛选
3. L3 对最高价值证据保留原文

Red/Blue 证据目录使用自适应长度，避免新增 VERIFY 证据因固定字符上限被截断。

## 7. Claim–Evidence Verifier

报告先拆成原子 Claim，再建立 `ClaimEvidenceLedger`。每个 Claim 的结果为：

- `supported`
- `partially_supported`
- `contradicted`
- `insufficient`
- `uncited`
- `unknown`

Verifier 同时检查时间、数值和实体冲突，并记录证据支持与未覆盖的具体方面。

成本控制策略：

- 只验证新增或被修改的 Claim。
- 使用 Claim 文本与 Evidence 内容的 SHA256 组合作为缓存键。
- 已支持且内容未变化的 Claim 不重复调用模型。
- 合并多个语义相似的待验证问题。
- 只对证据缺口定向检索，不重跑整份报告。

语义验证不可用或返回结构不完整时采用 fail-closed，不仅凭引用格式放行事实 Claim。

## 8. Red / Blue 对抗修复

Red 将问题分成：

- `factual_error`
- `inference_overreach`
- `citation_error`
- `evidence_gap`
- `writing_advice`

正常的证据不足声明和写作建议不会被当作阻塞性事实错误；重复问题通过 fingerprint 与轮次计数抑制，避免同一问题连续攻击。

Blue 返回结构化 JSON Patch，而不是自由重写全文。支持动作：

- `ADD`
- `DELETE`
- `MODIFY`
- `VERIFY`

程序按唯一 target 确定性应用 Patch，并拒绝：

- target 缺失或不唯一
- 未知 Evidence ID
- 无效 DELETE 或空替换
- 只删除引用标记
- 超过最大增长限制
- replacement 的 Claim 无法被证据支持

若候选证据只支持 Claim 的一部分，Blue 只能保留已支持部分；存在冲突则删除或降级；定向检索后仍无证据时改为明确的待验证问题。所有失败补丁均回滚。

## 9. 最终质量护栏与完成状态

默认硬门槛：

- 引用覆盖率不低于 80%
- Claim 支持率不低于 80%
- 不允许 severity 大于 1 的阻塞问题
- 不允许时间、数值、实体冲突残留

Dynamic 结果低于门槛或明显弱于 Fixed 时，系统生成 Fixed Harness 候选并比较两份报告。成本节省不能覆盖质量下降。

完成状态：

| 状态 | 含义 |
|---|---|
| `completed` | 质量门禁通过且没有显式证据缺口 |
| `completed_with_evidence_gaps` | 已支持事实通过门禁，但诚实保留无法回答的问题 |
| `completed_with_review_issues` | 仍有事实、推断、引用、冲突或验证问题 |
| `partial_timeout` | 全局超时后使用已有证据强制合成 |

## 10. 失败恢复

三级降级：

1. 单任务超时：`timed_out → degraded`，下游继续使用部分证据。
2. 批量失败：Planner 生成无冲突 replan，计划版本递增。
3. 全局超时：取消未完成节点并强制合成，结果标记为 `partial_timeout`。

评测支持 `--resume-from`：按 `pair_key` 复用成功行，只补跑失败项，并校验数据集 SHA256、Provider、Model、重复次数、Policy 版本、复杂度档位和预算兼容性。

## 11. 安全边界

Web 和知识库内容一律视为不可信数据。进入 Prompt 前检测并脱敏：

- 指令覆盖
- 密钥窃取
- 工具调用诱导
- 身份或系统提示伪造

Evidence metadata 保存检测信号、替换次数和清洗后内容哈希，不记录凭据。运行元数据只保存 Provider、Model、温度、超时、重试和 Prompt Schema 版本，不保存 API Key。

`.env`、SQLite 记忆库、索引、缓存和本地实验中间产物均由 `.gitignore` 排除。

## 12. 后端与配置

支持 DeepSeek、MiMo、vLLM 和 OpenAI-compatible 后端。配置从环境变量读取：

```dotenv
DEEPSEEK_API_KEY=your_key
DEEPSEEK_MODEL=deepseek-v4-flash
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_TIMEOUT_SECONDS=45
DEEPSEEK_MAX_RETRIES=1
TAVILY_API_KEY=your_key
```

复制模板：

```bash
cp .env.example .env
```

也可以安全输入 Tavily Key：

```bash
python3 evaluation/configure_tavily.py
```

该工具把 Key 写入被忽略的 `evaluation/.env` 并设置权限 `600`。

`fixture` 检索后端提供少量固定证据，只用于 CI 与安装 smoke。它不访问网络、不读取本地索引，也不能用于质量评测。

## 13. CLI 示例

安装：

```bash
python3 -m pip install -e .
python3 -m pip install -e ".[local-rag]"  # 需要本地 RAG 时
```

离线流水线：

```bash
python3 -m research_engine "研究问题" \
  --offline \
  --search-provider fixture
```

DeepSeek + Tavily：

```bash
python3 -m research_engine "研究问题" \
  --provider deepseek \
  --model deepseek-v4-flash \
  --search-provider tavily \
  --output evaluation/research_result.json
```

本地 + Web：

```bash
python3 -m research_engine "研究问题" \
  --provider deepseek \
  --search-provider hybrid-web \
  --local-weight 1.2 \
  --web-weight 1.0
```

`research_engine` 子模块使用包内相对导入。入口应为 `python3 -m research_engine` 或安装后的 `deep-research`，不要逐文件执行 `src/research_engine/*.py`。

## 14. 评测体系

### 数据集

| Track | 内容 | 目的 |
|---|---|---|
| Frozen v1.0 | 11 领域 / 35 题 / 冻结证据 | 可复现回归和成对消融 |
| Adversarial v0.1 | 35 Case / 140 Gold Fault | Red 检出与 Blue 修复 |
| Live v1.0 | 15 题 / 三档复杂度 | 真实 Web、角色变化与漂移 |
| Retrieval-Gold v1.0 | 33 条人工复核候选 | 检索阈值校准 |

### 指标

- 规则：事实准确率、幻觉率、引用覆盖率、引用有效性、来源多样性、关键事实召回率。
- Judge：事实性、逻辑性、引用质量、完整性、可执行性。
- 统计：题目级配对 Bootstrap 95% CI、Cohen's d、paired Cohen's dz。
- 系统：Worker/LLM 调用、Token、成本、延迟、角色分布、停止原因和回退次数。

### 四组 Blue 基线

1. `single_agent`
2. `no_red_blue`
3. `rewrite_blue`
4. `structured_patch_blue`

另以 `fixed_harness` 和 `dynamic_swarm` 隔离动态角色、并发、预算和停止条件的贡献。

```bash
PYTHONPATH=src python3 evaluation/run_ablation_experiments.py \
  --provider deepseek \
  --model deepseek-v4-flash \
  --variants single_agent no_red_blue rewrite_blue structured_patch_blue \
  --repeats 3 \
  --experiment-concurrency 1 \
  --bootstrap-samples 10000
```

`--provider offline` 只验证协议和流水线，不能用于声称模型质量提升。

## 15. 正式结果

### Fixed Harness vs Dynamic Swarm

ResearchBench-Frozen v1.0，35 题 × 3 次：

| 指标 | Fixed | Dynamic | Dynamic 相对变化 |
|---|---:|---:|---:|
| 综合质量 | 99.68% | 99.55% | -0.13 pp |
| 事实准确率 | 98.40% | 98.07% | -0.33 pp |
| 引用覆盖率 | 100.00% | 100.00% | 持平 |
| Worker 调用 | 11.56 | 9.67 | -16.39% |
| LLM 调用 | 7.60 | 7.02 | -7.64% |
| Token | 14,799 | 13,596 | -8.12% |
| 成本 | $0.009098 | $0.008453 | -7.09% |
| 延迟 | 26.15s | 24.72s | -5.45% |

质量差异未达显著：Δ=-0.13 pp，95% CI [-0.34, +0.07] pp，paired Cohen's dz=-0.20。Worker 调用降低 16.39%，95% CI 10.39%–22.13%；其他效率指标的区间跨 0。

正式产物：

- `evaluation/results/formal_frozen_v1_35x3/ablation-20260922T135223799454Z-summary.md`
- `evaluation/results/formal_frozen_v1_35x3/ablation-20260922T135223799454Z-quality-gate.md`
- `evaluation/results/frozen_release_20260921_v1/formal-run-launch.json`

### Structured Patch Blue

35 题 × 3 次正式对抗实验中，安全增强版本的保守字符串 Fault 修复率由 86.43% 提升到 99.29%，事实准确率达到 100%，幻觉率降为 0，干净事实和引用保留率均为 100%。

## 16. 可复现检查

参考执行环境：

- CPython 3.12.7：`.python-version`
- 锁定依赖：`requirements.lock`
- 自动化入口：`.github/workflows/ci.yml`

全新虚拟环境中的最小流程：

```bash
python3 -m pip install -r requirements.lock
python3 -m pip install --no-deps -e .
python3 -m research_engine --help
python3 -m pytest -q tests
python3 scripts/offline_smoke.py

cd evaluation/datasets
shasum -a 256 -c SHA256SUMS
cd ../..

python3 evaluation/validate_formal_ablation.py \
  evaluation/results/formal_frozen_v1_35x3/ablation-20260922T135223799454Z.json \
  --dataset evaluation/datasets/researchbench_frozen_v1.0.json \
  --output-prefix /tmp/researchbench-frozen-v1.0-quality-gate
```

GitHub Actions 在 Pull Request、`main` 分支推送和手动触发时运行 Ruff、覆盖率、Python 3.10/3.11/3.12 兼容测试、wheel/sdist 构建、全新环境 wheel 安装、模块与命令入口检查和零 Key smoke。在线 DeepSeek + Tavily 示例只在用户自行配置 `.env` 后手动运行，避免 CI 消耗额度或暴露凭据。

正式实验固定数据集、模型、费率、并发、超时和重复次数。三次重复先在题内聚合，再以 35 道题为独立单位做配对 Bootstrap。

## 17. 已知限制

- Frozen 单题证据量有限，主要用于隔离编排和修复差异。
- Live 会随搜索索引和网页内容变化，不能与 Frozen 合并统计。
- Retrieval-Gold 是被过滤候选切片，不代表总体检索分布。
- 成本是按固定费率和估算 Token 计算，不等于实际账单。
- OpenAlex 已覆盖真实检索与评测，但跨来源论文版本合并、引用图快照和撤稿状态仍依赖上游元数据质量。
- 本地 RAG 运行前需要构建可用的 FAISS、BM25 和 Embedding 资产。

系统架构图见 README；可编辑源文件为 `evaluation/deep_research_architecture.drawio`。
