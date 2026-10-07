# v1.2.1 真实 Web 演示：一页结论

## 研究任务

基于 Kubernetes 与 Gateway API 官方资料，对比 Ingress 与 Gateway API 的 API 成熟度、
核心资源模型、跨命名空间安全和实现兼容性，并给出采用建议。

## 运行结果

| 指标 | 结果 |
|---|---:|
| 端到端墙钟时间 | 142.85 秒 |
| DAG / 动态角色 / Worker 调用 | 8 / 4 / 11 |
| 检索候选 / 保留 / 过滤 | 104 / 52 / 52 |
| 保留结果中的官方文档 | 47 / 52 |
| LLM 调用 / Token | 12 / 193,451 |
| Claim 支持率 | 100.00% |
| 引用覆盖率 | 100.00% |
| 引用正确率 | 97.73% |
| A/B 证据矩阵覆盖率 | 87.50%（7/8） |
| Blue Patch 应用 / 拒绝 | 8 / 3 |
| Replay Trace 完整性 | 通过 |
| 成本 | 未配置费率，明确报告为 unknown |
| 最终状态 | `completed_with_review_issues` |

## 核心结论

1. Dynamic Swarm 将任务判定为 complex，选择 scope、evidence、counter、impact 四类角色，
   并以 4 并发执行 8 节点 DAG。
2. 检索层执行相关性排序、URL/内容去重与来源权威度过滤；跨命名空间任务自动补充
   `ReferenceGrant`、`HTTPRoute`、`backendRef`、`same namespace` 等规范英文实体。
3. Claim Verifier 最终确认 40/40 条可审核 Claim 获得支持；Structured Blue 应用 3 个
   MODIFY 和 5 个 DELETE Patch，并拒绝 3 个不安全 Patch。
4. Fixed fallback 被触发，但其候选同样存在 7 个阻断问题，因此质量比较保留动态结果；
   成本节省没有覆盖质量判断。
5. Red Agent 最终发现严重事实错误：正文称“Gateway API 具有四种稳定 API 类别”，而相邻
   证据只列出 GatewayClass、Gateway、HTTPRoute 三种。系统保留报告用于复盘，但拒绝标记为
   `completed`。

## 这次失败说明了什么

- 100% Claim 支持率不等于 100% 事实正确率：一个复合 Claim 可能因部分引文支持而通过，
  Red 的逐句事实核对仍然必要。
- 有引用不等于引用完整支持；本次引用正确率为 97.73%，与 100% 引用覆盖率必须分开解释。
- A/B 证据矩阵仍缺 1 个单元，证明系统应透明返回 evidence gap，而不是为追求全绿保留
  无法验证的断言。
- Harness 的价值不是保证每次都成功，而是把失败状态、触发原因、候选比较与修复轨迹完整
  持久化，使结果可审计、可 Replay、可继续修复。

## 面试表述

“这不是一张只展示成功结果的截图。真实 Web 运行里，Claim Verifier 给出了 100% 支持率，
但 Red Agent 仍发现三类与四类的事实冲突。系统因此拒绝发布，并用确定性 Replay 复现整条
Trace。这个案例说明我的重点不是堆 Agent，而是用 Harness 把预算、恢复、证据对齐、对抗
修复和质量状态做成可验证的工程闭环。”
