# Deep Research Agent

该模块把现有企业知识库 RAG 扩展为“规划—执行—记忆—对抗—评测”的多 Agent 深度研究系统。实现坚持两个边界：底层检索继续复用现有 Dense + BM25 + RRF；上层编排只依赖 Python 标准库，因此可以用假后端做快速测试。

## 架构

系统采用 Harness 分层，而不是把流程绑定到某个 Agent 框架：

```text
DeepResearchAgent / DAG Scheduler
              |
        AgentRuntime.invoke
              |
    AgentSpec + RunContext + lifecycle events
              |
   RoleAwareAgentAdapter / Legacy fallback
              |
         ResearchWorker
              |
          ToolRegistry
              |
 Local Hybrid RAG / Tavily / Composite RRF
```

`harness.py` 定义最小可替换契约：`RetrievalTool` 统一检索接口，
`ToolRegistry` 管理具名能力，`AgentSpec` 声明角色及其依赖，`RunContext`
隔离单次研究任务的产物与事件，`AgentRuntime` 负责执行。默认
`LocalAgentRuntime` 在当前进程运行；已有 Worker 通过 `LegacyAgentAdapter`
无侵入接入。因此本地 Dense + BM25 + RRF 不是被推翻，而是作为
`retrieval` 工具继续复用。后续接入远程 Worker、容器沙箱或动态 Swarm 时，
替换 Runtime/策略即可，不需要重写检索和评测。

### 动态 Swarm 与预算控制

`swarm.py` 在 Planner 产出 DAG 后执行确定性的复杂度评估。它综合 DAG 大小、
问题长度、比较型/时效型/高风险关键词、多维约束等信号，将任务分为
`simple / standard / complex`，并生成本次运行独立的 `SwarmPlan`：

- 简单任务：1 个通用研究角色，串行执行，最多 2 轮 Red/Blue 审查。
- 标准任务：Scope Researcher、Evidence Researcher、Counter Researcher 三类角色。
- 复杂任务：增加 Impact Analyst 与 Evidence Verifier，最多 5 类角色。

`BudgetController` 按运行跟踪 Worker 调用、动态 replan、验证查询、评审轮次和
总耗时。停止条件包括审查通过、收敛、震荡、连续低收益、预算耗尽和 deadline。
策略结果写入 `run_metadata.swarm`，实际消耗与停止原因写入
`metrics.budget`；每条 Evidence 也会记录产生它的 `agent_id` 和 `agent_role`。
`ResearchConfig` 始终是策略不可突破的硬上限，因此正式评测可保持成本口径稳定。

角色不仅改变 `AgentSpec.role`：`ResearchWorker.run_for_agent` 会为不同角色追加
不同的检索意图，扩大候选集后执行确定性质量管线。管线综合查询关键词覆盖、
实体覆盖、搜索后端分数、来源权威度和角色内容匹配度生成 0–1 相关性分数，再按
角色独立阈值过滤低相关页面。阈值已使用 ResearchBench-Live 的 45 个 Tavily
角色检索单元和 Retrieval-Gold v1.0 校准：通用 Researcher 保持 `0.14`，五类
专门角色使用 `0.28–0.48`；无有效查询词的兼容调用不会被任意误杀。

过滤后的候选会按“相关性—来源权威度—后端分数”排序，政府/法规、标准、论文和
官方文档优先于普通网页；随后依次执行规范 URL 去重、全文去重和同域近重复去重。
每次检索都会输出 `retrieval_filter_applied` Trace，包含过滤前候选数、评分数、
最终保留数、各淘汰原因及被淘汰 URL，不记录页面正文。每条 Evidence metadata
同时保留总分、五项分量、命中关键词/实体、规范 URL、来源类别和实际角色阈值，
便于离线调参和消融。Scope 偏向定义与适用范围，
Evidence/Verifier 偏向官方文档、标准、论文和直接证据，Counter 偏向反例、
限制与争议，Impact 偏向风险、成本和落地权衡。简单任务还会在连续批次没有
新增证据时触发 `evidence_saturated`，提前停止剩余 DAG 节点。旧 Worker 没有
`run_for_agent` 时自动回退到 `run(payload)`，兼容原有本地 RAG。

核心代码位于 `src/research_engine/`：

- `planner.py`：LLM/启发式 Planner、DAG 校验、动态 replan、三层 JSON 解析回退。
- `state.py`：`pending / ready / running / succeeded / failed / timed_out / degraded / skipped / cancelled` 九状态任务生命周期。
- `orchestrator.py`：`asyncio + Semaphore` 拓扑并发，任务超时、批量失败重规划、全局超时强制合成。
- `memory.py`：SQLite 跨 Agent 共享记忆、写前语义去重、启发式矛盾检测与消解、L1 向量粗筛—L2 TextRank—L3 原文保留。
- `agents.py`：现有 Hybrid RAG 的 Search Backend 适配器、研究 Worker 和证据约束 Synthesizer。
- `harness.py`：Agent/Tool/Runtime/RunContext 核心接口、旧 Agent 兼容适配器与本地运行时。
- `swarm.py`：任务复杂度评估、动态角色编组、角色路由、运行预算与停止策略。
- `adversarial.py`：Red Agent 多维攻击、Blue Agent 结构化 `ADD / DELETE / MODIFY` JSON Patch、确定性补丁执行、评分收敛和震荡检测；`VERIFY` 由编排器先检索再交给 Blue 修复。
- `claims.py`：原子 Claim 抽取、Claim–Evidence Ledger、规则/LLM 证据蕴含判断，以及补丁写入前的事实支持验证。
- `security.py`：把 Web/知识库检索结果视为不可信数据，检测并脱敏指令覆盖、密钥窃取和工具调用型 Prompt Injection。
- `config.py`：并发、超时、重规划和评审策略。
- `llm_backends.py`：DeepSeek、MiMo、vLLM、OpenAI 四类 OpenAI-compatible 后端热切换。
- `web_backends.py`：Tavily 真实 Web 检索，以及本地/Web/论文后端的并发 RRF 融合。
- `cli.py`：本地命令行入口。

可编辑架构图位于 `evaluation/deep_research_architecture.drawio`。

## 快速运行

先在项目根目录执行一次可编辑安装：

```bash
python3 -m pip install -e .
```

如需现有 Dense + BM25 + RRF 本地检索，使用
`python3 -m pip install -e ".[local-rag]"` 安装对应可选依赖。随后执行：

```bash
python3 -m research_engine \
  "公司远程办公政策的适用范围、风险和改进建议是什么？" \
  --output evaluation/research_result.json
```

也可以使用安装时注册的 `deep-research` 命令。`research_engine` 采用包内相对导入，
`agents.py`、`orchestrator.py` 等文件是库模块，不应逐文件直接运行；IDE 应使用
Module name `research_engine`，工作目录设为项目根目录。仓库内的共享 PyCharm 配置
`Deep Research CLI` 已采用这一模式。

当存在 `DEEPSEEK_API_KEY` 时，Planner、Synthesizer 和 Red Agent 默认使用 DeepSeek；缺少任一受支持后端的配置时，Planner 与合成器自动采用确定性降级实现。`--offline` 可强制关闭 LLM。检索仍复用本项目的本地 Hybrid RAG，因此需要先完成已有向量与 BM25 索引准备。

也可以显式选择后端：

```bash
python3 -m research_engine "研究问题" \
  --provider vllm --model your-local-model
```

后端统一读取 `{PROVIDER}_API_KEY`、`{PROVIDER}_MODEL` 和可选 `{PROVIDER}_BASE_URL`。vLLM 默认地址是 `http://127.0.0.1:8000/v1`；MiMo 与 OpenAI 不硬编码模型和地址，需通过参数或环境变量配置。

启用真实 Web 检索：

```bash
pip install tavily-python
export TAVILY_API_KEY="tvly-..."

python3 -m research_engine "研究问题" \
  --search-provider tavily \
  --output evaluation/research_result_web.json
```

生产或正式评测时建议约束一手来源。域名参数可以重复传入：

```bash
python3 -m research_engine "Tavily SDK 有哪些能力？" \
  --search-provider tavily \
  --web-include-domain docs.tavily.com \
  --web-include-domain github.com
```

Tavily 仍请求原文以便后续扩展抽取，但默认使用搜索返回的聚焦内容作为证据，避免整页导航、页脚等噪声直接进入报告。原文是否存在及长度只记录为元数据；离线合成会把每条证据压缩成一个带引用的关键句。

搜索摘要常用 `[...]`、`[…]` 或 `[……]` 表示非连续片段。证据规范化阶段会把这些
抽取占位符替换为段落边界，避免 Synthesizer 将其误当正文复述；Evidence metadata
记录 `extraction_omission_markers_removed`。最终报告还有一层确定性清理，Trace
事件 `report_omission_markers_removed` 及同名 metrics 记录清理阶段和数量。清理只
删除完整的方括号省略占位符，不删除普通句内省略号。

Red/Blue 默认把引用覆盖率 `80%` 设为硬门槛，同时阻断严重度大于 `1` 的事实、逻辑或证据误用问题。即使综合评分已通过，只要任一硬门槛未满足，Blue 仍会逐句补充有效证据 ID、删除无依据内容或降级为推断；有可执行问题时不会因为评分变化过小而提前判定收敛。代码块、来源索引和报告元信息不会被误算为事实陈述，最后一次 Blue 输出必须再经过 Red 复审。可通过 `--min-citation-coverage`、`--max-pass-issue-severity` 和 `--max-review-rounds` 调整策略。

Blue 不再直接自由重写整篇报告，而是返回 `{"patches": [...]}`。每条补丁包含 `patch_id`、关联的 `issue_ids`、`action`、从原报告逐字复制的唯一 `target`、`replacement`、插入位置和使用的 `evidence_ids`。程序按顺序执行精确匹配编辑，并拒绝目标缺失/歧义、未知证据 ID、非法动作、空替换、无效删除和超出增长上限的补丁。

`MODIFY/ADD` 在写入报告前还会生成临时 Claim–Evidence Ledger：将 replacement 拆成原子 Claim，并逐条判断引用证据是 `supported`、`partially_supported`、`contradicted`、`insufficient`、`uncited` 还是 `unknown`。Verifier 同时输出证据明确支持/未覆盖的限定，并标记 `time / number / entity` 三类冲突。启用 LLM 时采用 fail-closed 策略，语义验证调用失败或返回不完整不会仅凭引用格式放行。验证失败的补丁会回滚；若第一轮没有任何补丁成功，Blue 会接收逐条拒绝原因并进行一次保守重试，之后才降级到整篇重写。

最终报告同样会生成 `ResearchResult.claim_ledger`。每轮 Red 之后，编排器先执行逐句 Claim→Evidence 对齐；对未支持 Claim 生成有界、相似 Claim 合并后的定向检索查询，再以新增证据重建 Ledger。Verifier 使用 Claim 文本与 Evidence 内容的 SHA-256 组合键缓存语义判断，因此未修改且证据未变化的 Claim 不会重复调用 LLM。Blue 随后执行闭集确定性策略：候选证据完整支持则补相邻引用、仅部分支持则只保留已支持部分、发生冲突则删除原句、定向检索后仍无证据则改成显式待验证问题。所有动作都是精确 target Patch，不允许模型自由改写未命中的段落。

完成状态分为四类：`completed` 表示质量门禁通过且没有显式证据缺口；`completed_with_evidence_gaps` 表示事实与引用门禁通过，但报告诚实保留了无法由现有证据回答的问题；`completed_with_review_issues` 只表示仍存在事实错误、推断越界、引用错误、Claim 冲突或语义校验不可用；`partial_timeout` 表示全局超时后的强制合成。Red issue 同时保存 `factual_error / inference_overreach / citation_error / evidence_gap / writing_advice` 分类，正常的证据缺口与写作建议不再冒充阻塞性缺陷，同一语义问题连续出现到第三轮时停止重复攻击并交由最终 Claim Gate 裁决。

`claim_support_rate` 低于默认 `80%` 时不会因为 `review_passed` 提前停止修复。动态 Swarm 首次发现支持率不足会增加 Evidence Verifier；下一轮仍不足或支持率明显下降时，验证路径回退到固定基础 Worker。`BudgetController` 同时记录质量效用、Worker/Review/Verification 调用、耗时比例和代理成本单位，防止仅靠减少调用换取表面效率。`ResearchResult.trace` 会记录两阶段 `claim_evidence_alignment`、`swarm_quality_guardrail`、`deterministic_claim_patch`、`structured_patch`、`rewrite_fallback` 和最终 Ledger 指标；`system_metrics` 额外保存 `completion_issue_reasons`、`evidence_gap_reasons`、Claim cache 命中、verdict/冲突计数以及 Red issue 的分类/维度/动作分布。

所有检索证据在进入 Agent Prompt 前都会执行不可信内容扫描。命中的可疑指令片段会替换为 `[UNTRUSTED_INSTRUCTION_REDACTED]`，并在 evidence metadata 中记录 `prompt_injection_detected`、信号类型、替换次数和清洗后内容 SHA-256；汇总指标 `prompt_injection_evidence_count` 可用于安全评测。

每次运行都会生成独立 `run_id` 和不含 API Key 的 `run_metadata`，其中保存运行配置、模型/Provider、温度、请求超时、重试次数以及各 Prompt Schema 版本。各 Provider 默认请求超时为 45 秒、自动重试 1 次，也可以通过如 `DEEPSEEK_TIMEOUT_SECONDS` 和 `DEEPSEEK_MAX_RETRIES` 的环境变量调整。

Red 与 Blue 的证据目录使用自适应压缩，保证新增 VERIFY 证据的 ID 不会因固定字符上限而从上下文末尾丢失。每轮默认最多执行 4 个去重后的验证查询，其中至少保留一部分给 Claim 级定向检索，防止问题数量驱动证据和费用无限膨胀；连续两轮未通过时，Blue 会优先删除无依据内容并执行最小修改。可用 `--max-verification-queries-per-round` 调整验证上限。设置域名白名单时，Tavily 返回结果还会经过客户端二次校验；可用 `--web-exclude-url-fragment` 排除不匹配的 SDK 路径。

如果 Codex 进程无法继承另一个终端中执行的 `export`，可以使用隐藏输入配置工具：

```bash
python3 evaluation/configure_tavily.py
```

它将 Key 写入被 Git 忽略的 `evaluation/.env`，并设置文件权限为 `600`。CLI 和 Web 评测脚本会自动加载项目根目录或 `evaluation/` 下的 `.env`。

同时检索企业知识库和 Web：

```bash
python3 -m research_engine "研究问题" \
  --search-provider hybrid-web \
  --local-weight 1.2 \
  --web-weight 1.0
```

## 作为 Python 组件使用

```python
import asyncio

from research_engine import (
    DeepResearchAgent,
    HeuristicPlanner,
    LocalHybridSearchBackend,
    ResearchWorker,
    Synthesizer,
)


agent = DeepResearchAgent(
    planner=HeuristicPlanner(),
    worker=ResearchWorker(LocalHybridSearchBackend()),
    synthesizer=Synthesizer(),
)

result = asyncio.run(agent.run("研究问题"))
print(result.answer)
```

生产接入时可分别替换 `Planner`、`SearchBackend` 与传给 `Synthesizer/RedTeamReviewer` 的 LLM callable，而无需改动编排器。

## 降级语义

1. 单任务超时：任务依次进入 `timed_out -> degraded`，下游使用部分证据继续执行。
2. 批量失败：当一批 DAG 节点失败比例达到阈值，Planner 产生无冲突的新节点并把计划版本加一。
3. 全局超时：取消未完成节点，立即用已持久化证据强制合成，并把结果状态标为 `partial_timeout`。

所有状态转换、replan、Red Review、Blue Repair 和强制合成都写入 `ResearchResult.trace`，便于复盘和评测。

## 评测

评测数据被明确拆成两条互不混分的 Track：

- `evaluation/datasets/researchbench_frozen_v1.0.json`：正式版 11 领域/35 题，包含冻结证据快照、关键事实和禁止陈述，用于可复现回归、消融和版本比较。`v0.1` 的 6 道种子题保留作历史兼容。
- `evaluation/datasets/researchbench_adversarial_v0.1.json`：以 Frozen v1.0 的 SHA-256 为父版本，确定性生成 35 个对抗样本和 140 个 gold fault，覆盖未知引用、无引用陈述、证据不支持结论和缺失局限说明。
- `evaluation/datasets/researchbench_live_v0.1.json`：只保存问题、时间边界、目标域名和关键词，运行时通过 Tavily 实时检索，用于衡量当前 Web 能力。

`evaluation/research_evaluation.py` 提供：

- 规则指标：事实支持率、幻觉率、引用覆盖率、引用有效性、来源多样性、关键事实召回率和禁止陈述命中率。
- 可选 LLM-as-Judge：事实性、逻辑性、引用质量、完整性、可执行性五维评分。
- 统计推断：Bootstrap 95% 置信区间、独立样本 Cohen's d 和成对 Cohen's dz。

六组一键消融实验：

```bash
PYTHONPATH=src python3 evaluation/run_ablation_experiments.py \
  --provider deepseek \
  --model deepseek-chat \
  --repeats 3 \
  --experiment-concurrency 4 \
  --bootstrap-samples 10000
```

运行器会在同一 Frozen 样本上比较：

1. `single_agent`：单检索任务与单次报告合成。
2. `no_red_blue`：保留多任务 DAG，但关闭 Red/Blue。
3. `rewrite_blue`：Red 审查后使用整篇重写修复。
4. `structured_patch_blue`：Red 审查后使用结构化 Patch、语义验证和失败回滚。
5. `fixed_harness`：完整 Structured Patch Blue，但固定单一 Research Worker
   角色、配置并发与完整 DAG，作为 Swarm 对照基线。
6. `dynamic_swarm`：使用复杂度分级、角色差异化、动态并发、预算和停止条件。

结果写入 `evaluation/results/ablations/`，包含逐题报告、冻结证据、Trace、规则指标、关键事实指标、Worker/LLM 调用次数、角色调用分布、停止原因、估算 Token、估算美元成本、延迟、题目级配对 Bootstrap 95% CI，以及关键变体间的 Cohen's dz。三次随机运行先在题内聚合，不会被错误地当成 105 道独立题。可用 `python3 evaluation/summarize_ablation_results.py <结果.json>` 生成面试与简历用 Markdown 报告。`--provider offline` 只用于验证流水线；由于它不调用模型，不能把各组相同的质量分数解释为有效消融结论。

美元成本使用可审计的命令行费率，不在代码里硬编码易变化的厂商价格：

```bash
PYTHONPATH=src python3 evaluation/run_ablation_experiments.py \
  --provider deepseek --model deepseek-chat --repeats 3 \
  --variants fixed_harness dynamic_swarm \
  --input-cost-per-million-usd <当前输入价格> \
  --output-cost-per-million-usd <当前输出价格>
```

当费率为 0 时仍会报告 provider-neutral 的估算 Token，但美元成本保持为 0。
DeepSeek 存在缓存命中、非高峰和高峰多档价格；正式对照应固定一种价格情景，
不要把估算值冒充实际账单。断点续跑时，历史成功样本会使用新命令传入的费率
重新计价，但不会重新调用模型。
Frozen v1.0 每题只有一条证据，因此适合隔离编排和成本差异；角色检索召回率的
提升应另外在 ResearchBench-Live 或多文档 Frozen 语料上评测，不能由当前单证据
Frozen 结果推断。

ResearchBench-Live v1.0 新增 15 道复杂度标注题，按 `simple / standard / complex`
各 5 道覆盖 1/3/5 Agent 编组，并为每题声明期望角色组合。先运行不产生网络费用的
确定性策略回归：

```bash
PYTHONPATH=src python3 evaluation/evaluate_live_swarm.py \
  --provider offline --strict
```

再用真实 LLM Planner 重复验证规划波动是否导致编组漂移：

```bash
PYTHONPATH=src python3 evaluation/evaluate_live_swarm.py \
  --provider deepseek --model deepseek-v4-flash \
  --repeats 3 --experiment-concurrency 4 --strict
```

评测结果同时记录复杂度分数、触发信号、Planner 子任务数、Agent 数、并发预算、完整
角色组合以及标准化角色路由探针。它只验证 Swarm 编组契约，不把策略分类通过率冒充
检索质量；真实 Web 召回率继续由 `evaluate_web_search.py` 单独衡量。

正式 DeepSeek Planner 三次重复结果见
`evaluation/results/live_swarm/live-swarm-20260916T113641Z.md`：45/45 次编组断言
全部通过。simple 复杂度分数为 0.08–0.22，对应单一 `researcher`；standard 为
0.44–0.60，对应 `scope/evidence/counter` 三角色；complex 为 0.90–1.00，对应
`scope/evidence/counter/impact/verifier` 五角色。三档 Agent 数量分布均为 15 次，
没有跨档漂移。该结果验证的是 LLM Planner + SwarmPolicy 的配置决策稳定性，不是
端到端 Web 检索质量结论。

校准前 `heuristic-v1`（simple < 0.22，complex >= 0.58）的正式 Dynamic Swarm 对照结果见
`evaluation/results/ablations/ablation-20260916T104327086587Z.md`。Fixed Harness 与
Dynamic Swarm 各完成 35 题 × 3 次、共 210 次运行，失败数为 0。相较固定编排，
Dynamic Swarm 的 Worker 调用降低 21.06%（题目级配对 Bootstrap 95% CI：
19.06%–23.39%）、LLM 调用降低 9.36%（6.64%–12.33%）、估算 Token 降低
10.20%（6.50%–14.12%）、平均延迟降低 9.31%（5.63%–13.35%）、按固定高峰
非缓存费率估算的成本降低 8.09%（4.46%–11.86%）。综合质量差异未达显著：
Δ -0.57 个百分点，95% CI [-2.12, +1.02]，paired Cohen's dz=-0.12，不能宣称
质量提升，也没有证据表明存在稳定质量下降。

该次运行记录了 `evidence_researcher`、`counter_researcher`、`scope_researcher`
三类专门角色及其差异化查询，55/105 次运行触发证据饱和停止。复杂度分类为
2 次 simple、103 次 standard、0 次 complex，因此这轮数据主要证明动态停止和
预算收缩的收益。阈值探索发现 `heuristic-v2` 的 simple < 0.36 过于激进：虽然
Worker、Token 和估算成本分别下降 32.38%、21.13% 和 18.17%，综合质量却显著下降
2.30 个百分点（95% CI [-3.72, -0.93]）。`heuristic-v3` 恢复 simple 的第三轮
Red/Blue 后仍未形成理想的全量质量—成本权衡，因此两版均保留为失败消融证据，
不作为默认策略。

当前默认是更保守的 `heuristic-v4`：simple < 0.23，complex >= 0.72；单 Agent
与审查深度解耦，simple 仍保留三轮 Red/Blue 和四次验证查询。正式同配置结果见
`evaluation/results/ablations/ablation-20260919T122639378060Z.md`：Fixed 与 Dynamic
各完成 35 题 × 3 次，共 210 次执行且运行异常数为 0。Fixed 有 103/105、Dynamic
有 105/105 标记为 `completed_with_review_issues`，因此“执行成功”不表示通过质量
门禁。对该历史产物的 Gate 诊断显示：208 次问题运行中，204 次 Red 未通过、192 次
Claim 支持率低于 80%、176 次引用覆盖率低于 80%，188 次同时卡在 Red 与 Claim
两道门禁；详见 `evaluation/results/diagnostics/completed-review-issues-ablation-20260919T122639378060Z.md`。
该产物生成时尚未保存逐 Claim verdict 和 Red issue 维度，因此更细分类需由升级后的
链路在新回归中采集。Dynamic 的综合质量差异不显著
（Δ -0.12 个百分点，95% CI [-1.58, +1.25]，paired Cohen's dz=-0.03）；Worker
调用降低 21.32%（95% CI 18.21%–24.60%），LLM 调用降低 6.00%
（3.00%–8.93%），Token 降低 7.85%（3.01%–12.41%），固定费率下估算成本
降低 5.81%（0.96%–10.43%）。平均延迟增加 6.00%（1.41%–10.74%），因此不能
宣称 Dynamic 同时改善延迟。本次因 API 余额中断使用同一新实验断点续跑：Fixed
复用本轮已成功的 60 条并补跑 45 条，Dynamic 全新运行 105 条；未复用旧策略或
兼容回放结果。先前 v4 兼容回放仅保留为策略筛选历史，不再用于正式简历数字。

可对任一消融实验产物复跑原因统计：

```bash
python3 evaluation/analyze_review_issue_distribution.py \
  evaluation/results/ablations/<artifact>.json
```

三档各 1 题的 DeepSeek + Tavily 端到端结果见
`evaluation/results/live_swarm/e2e-v4-summary.md`。系统实际形成 1/3/5 Agent 与
1/3/5 并发；simple 配置/实际角色为 1/1，standard 为 3/2，complex 为 5/5。
Claim 支持率分别为 81.82%、62.50% 和 44.12%，说明角色编组已经生效，但复杂真实
Web 任务的主要瓶颈转为搜索结果主题相关性和引用支持，不应继续单纯增加 Agent。

若 API 余额、限流或网络问题导致部分样本失败，可以复用已成功的 `pair_key`，只补跑失败项：

```bash
PYTHONPATH=src python3 evaluation/run_ablation_experiments.py \
  --provider deepseek --model deepseek-chat --repeats 3 \
  --experiment-concurrency 4 --bootstrap-samples 10000 \
  --resume-from evaluation/results/ablations/<上次结果.json>
```

`--resume-from` 可以重复传入多个结果文件；后传入的兼容样本覆盖先前样本。断点续跑
会校验数据集 SHA-256、Provider、Model、重复次数、Policy 版本、复杂度档位和具体
档位预算，配置不兼容时拒绝合并。

正式 DeepSeek 三次重复结果见 `evaluation/results/ablations/ablation-20260915T143700857823Z.md`。四组共 420 次运行全部成功。Structured Patch Blue 相对整篇重写 Blue 的综合质量差异未达显著，但端到端延迟降低 6.32%（题目级配对 Bootstrap 相对变化 95% CI：3.38%–9.31%，3/3 重复同向），输出字符量降低 4.87%（95% CI：1.10%–8.53%）；模型调用数增加 17.94%。因此不应把本轮结果表述为稳定的质量提升。

对抗修复评测：

```bash
PYTHONPATH=src python3 evaluation/run_adversarial_benchmark.py \
  --provider deepseek --model deepseek-chat \
  --repeats 3 --experiment-concurrency 4 --bootstrap-samples 10000
```

该流水线比较 corrupted no-repair、整篇重写 Blue、Structured Patch Blue 与 oracle clean，报告 fault repair rate、Red detection recall、关键事实保留率、collateral damage rate、引用有效性/覆盖率/保留率和补丁接受率。另有 4 类确定性恶意补丁用于验证未知引用、缺失锚点、歧义锚点和非法 DELETE 的回滚正确率。`--provider offline` 只验证协议、规则 Red 与 oracle ceiling，不代表模型修复效果。

单题 DeepSeek 冒烟结果见 `evaluation/results/adversarial/adversarial-20260916T022514213257Z.json`：语义 Red 检出 4/4 已知故障；Structured Patch 修复 4/4、关键事实与引用保留率 100%、规则事实准确率和引用覆盖率均为 100%，4 个补丁应用、1 个候选补丁被验证层拒绝。该结果只验证在线链路，不能作为总体提升结论。

35 题 × 3 次 DeepSeek 四变体正式在线结果见 `evaluation/results/adversarial/adversarial-20260916T031514890111Z.md`，四个变体共 420 次运行全部成功。当前安全版本只增量复跑受代码变更影响的 Structured Patch 105 个样本，原始结果见 `evaluation/results/adversarial/adversarial-20260916T065323635595Z.json`，与最初 Structured 运行的严格配对报告见 `evaluation/results/adversarial/adversarial-20260916T065323635595Z-vs-adversarial-20260916T024120075281Z.md`。数据集 SHA、协议、模型、题数、重复次数与统计口径均保持不变。结果如下：

- Structured Patch Blue 的保守字符串 fault repair 从 86.43% 提升到 99.29%，题目级前后配对 Δ +12.86 个百分点（Bootstrap 95% CI：+10.00–+15.71），三次重复均为正。
- 规则事实准确率从 97.34% 提升到 100%，配对 Δ +2.66 个百分点（95% CI：+0.97–+4.64）；规则幻觉率降至 0，引用覆盖率、引用有效性、干净事实保留率与引用保留率均为 100%，collateral damage 为 0。
- `rule-uncertainty` 不再交给模型生成；程序仅在报告仍缺少 epistemic limitation 时追加 `AUTO-rule-uncertainty`，并区分“报告/证据局限”与“研究对象自身限制”。缺失局限说明从 49.52% 提升到 100%，本轮触发 102 次。
- high-severity factual DELETE/MODIFY 只有在目标整行完全匹配、引用 ID 合法且描述明确为冲突或不支持时才降级为 `AUTO-factual-delete`，本轮触发 2 次；模型已产生否定式纠正时禁止按子串删除。
- citation-only DELETE 会被执行器拒绝；未知引用只有在词法候选唯一且通过 Claim–Evidence 验证后才执行 `AUTO-unknown-citation`，本轮触发 6 次。
- 3/420 个字符串 fault 未计为修复，其中两条是“并非只要一台服务器写入就提交”“不能假定模型输出天然安全”一类语义正确的否定式纠正；另一条真实残余已由 high-severity MODIFY fallback 在后续 3/3 在线定向回归中修复。项目不再通过危险的子串删除追求表面 100%。

`evaluation/evaluate_web_search.py` 使用 `ResearchBench-Live` 运行真实 Tavily 检索，并把每次结果按 UTC 时间写入 `evaluation/results/`。它不会覆盖 Frozen 消融结果：

```bash
PYTHONPATH=src python3 evaluation/evaluate_web_search.py
```

若评测题指定必须使用官方或一手来源，可运行受约束版本：

```bash
PYTHONPATH=src python3 evaluation/evaluate_web_search.py \
  --enforce-required-domains
```

无约束首跑用于衡量普通搜索的自然来源质量；受约束结果用于衡量在来源策略生效后的检索质量，两者应分别保留。

### Live 检索阈值校准

`evaluation/calibrate_retrieval_thresholds.py` 会对 Live 数据集中每道题的实际 Swarm
角色分别执行 Tavily 检索，并在同一批候选缓存上扫描全部阈值，避免 Web 漂移污染
阈值对照。最近一次正式运行使用 15 题、45 个角色检索单元、每单元 10 条候选，
45/45 均成功。Proxy 校准报告位于
`evaluation/results/retrieval_calibration/retrieval-calibration-20260916T150417Z.md`；
接入正式人工 Gold 后的报告位于
`evaluation/results/retrieval_calibration/retrieval-calibration-20260917T023454Z.md`。

校准后候选保留率从 98.7% 降至 83.8%，基于 required domain/term 弱标签的
Proxy 精确率从 73.2% 提升至 77.7%，平均来源权威度从 0.446 提升至 0.458，
一手来源占比从 16.7% 提升至 18.8%，Proxy 误杀率为 10.0%。误杀率不是人工
相关性标注结果，正式论文或简历中必须保留 `proxy` 限定。

对 33 条 Proxy 误杀候选的正式人工审核结果保存在
`evaluation/datasets/researchbench_retrieval_gold_v1.0.json`：18 条判定为真正误杀，
15 条为正确淘汰。Counter 的 13 条候选中有 8 条真误杀，Scope 的 10 条中有
6 条真误杀，暴露出中文问题—英文证据和角色语义仅靠词面分数难以处理。该集合只
抽样了被过滤候选，`54.5%` 表示“复核候选中的真误杀占比”，不是全体检索结果的
FNR。数据集状态为 `adjudicated_gold`，相关性、来源质量和保留决定已由项目负责人
确认，可作为后续检索阈值与语义相关性模型的正式 Gold 子集。

Gold 校准在四个有人工样本的角色上按最大 F1 选阈值：Scope `0.44`、Evidence
`0.48`、Counter `0.28`、Impact `0.42`；Researcher `0.14` 和 Verifier `0.42`
因没有 Gold 样本而保留 Proxy 推荐。相较纯 Proxy 阈值，Gold 审核切片的误杀率
从 100% 降至 0%，Precision/Recall/F1 为 `66.7%/100%/80.0%`；同时全量弱标签
Proxy 误杀率由 10.0% 降至 1.5%，但保留率从 83.8% 升至 96.2%。这些 Gold 数字
只适用于被过滤候选切片，不是总体检索 Precision/Recall 的无偏估计。

可复用已有原始候选重新扫描阈值，不会再次消耗 Tavily credits：

```bash
PYTHONPATH=src python3 evaluation/calibrate_retrieval_thresholds.py \
  --cache evaluation/results/retrieval_calibration/retrieval-calibration-20260916T145324Z.json \
  --gold evaluation/datasets/researchbench_retrieval_gold_v1.0.json
```

运行单元测试：

```bash
PYTHONPYCACHEPREFIX=/tmp/deep_research_pycache \
PYTHONPATH=src \
python3 -m unittest discover -s tests -p 'test_research_engine.py' -v
```

## 下一阶段

当前 `v1.0` Frozen 数据集包含 11 领域/35 题，Adversarial v0.1 包含 140 个可追踪故障，Live v1.0 包含覆盖三档复杂度的 15 道真实检索题。Frozen 35×3 对照、Adversarial 三次正式在线实验、确定性 ADD/DELETE、未知引用恢复、安全边界回归、Retrieval-Gold 人工标注与六类角色阈值校准均已完成。下一步可接入 OpenAlex/语义学术搜索，并增加时效性、论文版本合并、撤稿识别与端到端成本评分。
