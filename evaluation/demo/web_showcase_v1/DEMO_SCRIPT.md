# 2–3 分钟演示讲稿

## 0:00–0:20：一句话定位

“这个项目不是把多个 Prompt 串在一起，而是一个 Agent Harness。它根据任务复杂度动态选择角色，
用 DAG 并发执行检索，再通过 Claim–Evidence Verifier、Red/Blue Patch 和质量护栏决定何时停止。”

打开 [`trace_bundle.json`](trace_bundle.json) 的 `run.question`。

## 0:20–0:45：DAG 与动态角色

打开 `dag.subtasks` 和 `run.run_metadata.swarm`：

- Planner 把问题拆成 8 个有依赖关系的节点，而不是固定执行链。
- SwarmPolicy 给出 0.92 的复杂度分数，原因包括比较、高风险、多约束和大型 DAG。
- 系统选择 5 类角色：Scope、Evidence、Counter、Impact 和 Evidence Verifier，最大并发为 4。

强调：“Agent 数量不是配置常量，而是一次运行的策略结果。”

## 0:45–1:10：检索不是拿到就用

打开 `summary.retrieval`，再查看任意 `trace` 中的 `retrieval_filter_applied`：

- 12 批检索产生 104 个候选，保留 52、淘汰 52。
- 每个角色修改查询词并使用独立阈值；淘汰记录包含 URL、分数和原因。
- 保留结果中有 16 个官方文档、2 个源码仓库来源。
- 本次出现一次 `duplicate_content` 淘汰，说明系统同时处理 URL 之外的正文重复。

## 1:10–1:40：逐句 Claim 验证与定向补检

打开 `claim_ledger.metrics` 和 `trace` 中的 `claim_evidence_alignment`：

- 第一轮定向补检前支持率是 88.24%，补检后提升到 90.59%，并消除了实体冲突。
- Blue Patch 后第二轮支持率达到 98.77%，Red Review 通过。
- 后续只验证新增或修改的 Claim；最终复用了 80 条未变化 Claim 的判断。
- 最终 82 条可审查 Claim 中 80 条完整支持、1 条部分支持、1 条无引用。

强调：“系统不是因为引用格式正确就认为 Claim 正确，而是判断证据是否真正支持这句话。”

## 1:40–2:10：Red/Blue 不是整篇重写

打开 `trace` 中的 `red_review` 和 `blue_repair`：

- 共执行 2 轮 Red 审查，第二轮达到 0.9667 并通过。
- Blue 应用了 8 个确定性补丁：1 个 DELETE、7 个 MODIFY，0 个整篇重写。
- 一个真实例子是删除与证据冲突的断言；另一个例子是降低证据只部分支持的表述强度。
- 每个补丁保存 issue、目标 Evidence、原因和校验结果。
- 另有 1 个 Patch 因 `claim_target_not_unique` 被拒绝，说明模型补丁不能绕过程序校验。

## 2:10–2:35：预算与停止条件

打开 `trace` 中的 `swarm_stop` 和 `claim_ledger_built`：

- 第一轮 Red 未通过后执行定向补检和 Patch。
- 第二轮 Red 通过，系统以 `review_passed` 停止，没有为了用完预算继续调用。
- 最终支持率 97.56%、引用覆盖率 98.78%，保留两个显式缺口。
- 如果硬门禁失败，系统才会生成 Fixed 候选；本次该 fallback 没有触发。

## 2:35–3:00：主动展示不完美

打开 `run.status` 和 `final_review`：

“最终状态不是 completed，而是 completed_with_evidence_gaps。系统保留了一条部分支持和一条无引用 Claim，
并把它作为证据缺口公开。这是我刻意保留的演示点：Harness 的目标不是永远输出满分，而是让失败可见、
修复可追踪、停止有预算边界。”

最后打开 [`final_report.md`](final_report.md)，展示 Evidence ID 能回到真实 URL。
