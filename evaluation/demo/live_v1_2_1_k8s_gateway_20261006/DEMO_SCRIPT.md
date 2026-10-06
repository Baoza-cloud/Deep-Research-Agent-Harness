# 2–3 分钟演示讲稿

## 0:00–0:20：问题与目标

“我用一个真实问题演示：基于官方资料对比 Kubernetes Ingress 与 Gateway API。这里的重点
不是让一个 Agent 写长文，而是展示 Harness 如何动态组织角色、控制预算、验证 Claim、执行
结构化修复，并在质量不合格时拒绝发布。”

## 0:20–0:50：动态 Swarm 与 DAG

打开 `trace_viewer.html` 的 DAG 与角色区域。

“策略把任务判定为 complex，选择 scope researcher、evidence researcher、counter researcher
和 impact analyst 四个角色，以 4 并发执行 8 个有依赖的节点。固定编排只会照模板启动角色；
这里的角色数、组合、并发和停止条件来自任务复杂度与预算控制器。”

## 0:50–1:20：检索过滤

展示 Retrieval Trace。

“11 个检索批次返回 104 个候选，保留 52 个，其中 47 个是官方文档。系统记录每一条过滤原因，
并对跨语言技术问题补充规范英文实体，例如 ReferenceGrant、HTTPRoute 和 backendRef，避免中文
查询漏掉官方英文规范。”

## 1:20–1:50：Claim 验证与 Blue Patch

展示 Claim Ledger 与 Blue repair 事件。

“每条可审核 Claim 都与证据逐句对齐。Blue 不重写整篇报告，而是返回 DELETE、MODIFY、ADD
JSON Patch，程序按精确 target 逐条应用。本次应用 8 个、拒绝 3 个；同 URL 的不同证据片段
保留独立身份，Patch 后还检查重复、枚举冲突、残缺引用和未闭合格式。”

## 1:50–2:25：质量护栏与真实失败

展示 Fixed fallback、Red issue 与最终状态。

“Claim 支持率和引用覆盖率都是 100%，但这并不等于报告可发布。Red Agent 发现报告把证据中的
三种稳定 API 类别写成四种，这是 severity 3 事实错误。Fixed fallback 也没有清除阻断问题，
所以 Harness 返回 `completed_with_review_issues`。这里不会为了展示效果把失败隐藏成 completed。”

## 2:25–2:50：Replay 与工程差异

展示 Replay integrity 与瀑布图。

“Run、DAG 节点、预算、中间报告和事件都持久化。Replay 在不调用模型和检索的情况下验证了
146 条事件的哈希链。普通 Agent 框架通常解决工具调用；这个项目重点解决执行治理、质量门禁、
崩溃恢复、确定性 Patch、实验评测和可审计性。”

## 结束句

“如果继续迭代，我会把复合 Claim 拆成原子 Claim，并把 A/B 语义证据矩阵纳入最终候选排序，
目标不是让分数看起来更高，而是让错误更难越过发布门禁。”
