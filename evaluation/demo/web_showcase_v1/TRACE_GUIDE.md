# Trace 导航

使用支持 JSON 折叠和搜索的编辑器打开 [`trace_bundle.json`](trace_bundle.json)。

| 想展示的能力 | JSON 路径或搜索词 |
|---|---|
| 研究问题和运行状态 | `run.question`、`run.status` |
| 模型、配置和 Prompt 版本 | `run.run_metadata` |
| 8 节点 DAG | `dag.subtasks` |
| 动态复杂度与五类角色 | `run.run_metadata.swarm` |
| 每个任务实际路由角色 | 搜索 `"event": "role_strategy_applied"` |
| 过滤前后数量、阈值和淘汰原因 | 搜索 `"event": "retrieval_filter_applied"` |
| 不含正文的 Evidence 元数据 | `evidence_catalog` |
| Red 轮次、分数和问题数 | 搜索 `"event": "red_review"` |
| 定向补检 | 搜索 `"event": "review_verification"` |
| Claim 支持率变化 | 搜索 `"event": "claim_evidence_alignment"` |
| DELETE/MODIFY 补丁 | 搜索 `"event": "blue_repair"` |
| Dynamic 与 Fixed 最终候选 | 搜索 `dynamic_swarm_final_quality_guard` |
| 最终逐句支持关系 | `claim_ledger.claims` |
| 最终非阻塞问题 | `final_review.structured_issues` |
| 最终报告 | `final_report` 或 [`final_report.md`](final_report.md) |

## 三个最有说服力的 Trace 片段

1. `claim_evidence_alignment`：第一轮补检前支持率 88.24%，补检后为 90.59%，并消除实体冲突；
   Blue Patch 后第二轮达到 98.77%。
2. `blue_repair`：Structured Patch 删除 17 条无法由证据支撑的过度陈述，没有整篇重写。
3. `blue_repair`：一个 Patch 因 `claim_target_not_unique` 被程序拒绝；第二轮 Red 通过后以
   `review_passed` 停止，没有继续消耗剩余审查预算。
