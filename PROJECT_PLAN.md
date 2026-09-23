# Deep Research Agent Harness Project Plan

## 1. 项目定位

本项目构建一套面向复杂研究任务的 Agent Harness。核心目标不是封装某个 Agent 框架，而是把规划、执行、检索、验证、修复、预算与评测变成显式、可替换、可审计的程序契约。

本地 Hybrid RAG 保留为可插拔 Retrieval Backend，用于企业私有文档检索；它与 Tavily、论文和其他检索来源处于同一工具层。

## 2. 核心能力

- 结构化 ResearchPlan DAG 与依赖、环路校验
- `asyncio + Semaphore` 拓扑并发和九状态任务生命周期
- SwarmPolicy 按复杂度动态选择 1/3/5 个角色
- BudgetController 联合约束调用、时间、成本和停止条件
- 角色差异化检索、相关性过滤、权威度排序与 RRF 融合
- SQLite 共享记忆、去重、矛盾检测和三级上下文压缩
- Claim–Evidence Ledger 与时间、数值、实体冲突验证
- Red Review 与 Structured Patch Blue 确定性修复
- Dynamic/Fixed 质量候选比较和降质回退
- Frozen、Live、Adversarial、Retrieval-Gold 四类评测数据
- 题目级配对 Bootstrap、Cohen's dz、调用与成本统计

## 3. 系统边界

```text
研究问题
  → Planner / SwarmPolicy / BudgetController
  → Async DAG Orchestrator
  → Role-aware Workers
  → Retrieval Backends（Tavily / Local Hybrid RAG / Composite）
  → Shared Memory / Context Compression
  → Synthesizer
  → Claim Verifier / Red Review / Structured Patch Blue
  → Quality Guard
  → Report / Ledger / Trace / Metrics
```

Harness 只依赖 `AgentSpec`、`RunContext`、`ToolRegistry`、`AgentRuntime` 和 `RetrievalTool` 等最小契约。模型、检索后端与 Worker Runtime 可以独立替换。

## 4. 已完成里程碑

- [x] Harness 接口与旧 Worker 兼容适配
- [x] DAG 并发执行、九状态机与三级失败降级
- [x] Dynamic Swarm 与 BudgetController
- [x] 六类角色检索策略与 Retrieval-Gold 阈值校准
- [x] Claim–Evidence Verifier 与定向补证据
- [x] Red/Blue 结构化 Patch、验证和回滚
- [x] Dynamic 最终质量护栏与 Fixed fallback
- [x] ResearchBench-Frozen 11 领域/35 题
- [x] 35 题 × 3 次 Fixed/Dynamic 正式实验
- [x] 35 题 × 3 次 Red/Blue 对抗实验
- [x] Python 3.12.7、依赖锁、112 项测试和离线 CI smoke
- [x] README、技术文档、数据哈希和正式实验产物整理

## 5. 后续方向

- 扩大 Live Track 的时间跨度与外部来源覆盖
- 增加跨来源论文版本合并、撤稿和引用图快照
- 引入真实企业知识库的权限隔离与审计场景
- 增加远程/容器化 AgentRuntime 和分布式执行实验
- 扩充人工 Claim–Evidence Gold，降低规则评测偏差
- 为长期运行增加 Trace 可视化与成本告警

后续结果必须继续区分 Frozen 可复现指标与 Live 在线指标，不以成本下降替代质量门禁，也不把非显著点估计写成稳定收益。
