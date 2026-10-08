# Blue Patch 修复后在线复验

运行 ID：`demo-v122-k8s-gateway-20261008-bluefix-rerun2`

运行后端：Tavily Advanced Search + DeepSeek `deepseek-v4-flash`

本次与上一轮使用相同问题、模型、检索深度、并发、质量门槛和审查预算。代码基线为提交 `d6805b2f34dc6717f29deb2250b364f465bf7118` 加工作区补丁 `33125a105a7201e06259eb66c7738dad15907b2927f0b0d13f175b69cea2449e`。

## 最终结果

| 指标 | 上一轮 | 当前在线复验 | 变化 |
|---|---:|---:|---:|
| 完成状态 | `completed_with_review_issues` | `completed_with_evidence_gaps` | 消除 Review blocker |
| 阻断型 Red issue | 14 | 0 | -100% |
| Claim 支持率 | 84.91% | 93.88% | +8.97 个百分点 |
| 引用覆盖率 | 100.00% | 97.96% | -2.04 个百分点 |
| 引用正确率 | 67.16% | 86.89% | +19.72 个百分点 |
| 数值/时间/实体冲突 | 0 / 0 / 3 | 0 / 0 / 0 | 实体冲突清零 |
| Token | 440,972 | 427,622 | -3.03% |
| Harness 有效耗时 | 305.253 秒 | 285.593 秒 | -6.44% |

最终 Claim Ledger 包含 49 条可审查 Claim：46 条完整支持、2 条部分支持、1 条无引用，0 条矛盾、0 条证据不足。

最终 Red Review 只保留 4 个严重度 1 的 `evidence_gap`：实现兼容性覆盖不足、Ingress 成熟度直接评估不足，以及比较矩阵中两项 A/B 证据缺口。没有 `factual_error`、`inference_overreach` 或 `citation_error` blocker。

`review_passed=false` 的原因是综合 Review 分数 0.6583 未达到通过线且审查预算耗尽，不代表仍有阻断型事实问题。因此最终状态正确标记为 `completed_with_evidence_gaps`，而不是 `completed`。

## 确定性修复率

### Claim Ledger 确定性 Patch

- 请求 18 个。
- 应用 15 个。
- 拒绝 3 个，均因重复 Claim 完整性门禁。
- 修复率：`15 / 18 = 83.33%`。
- 上一轮同口径：`11 / 26 = 42.31%`。
- 提升：`+41.03` 个百分点。

Round 1 的 `semantic_verifier_unavailable:lexical_fallback` 是语义验证不可用信号，`requested_patch_count=0`，不属于可执行 Patch，因此不计入上述分母。

### 全部程序生成 AUTO Patch

Structured Blue fallback 另产生 2 个 AUTO Patch：

- `AUTO-citation-delete-R2` 成功应用。
- `AUTO-citation-relink-R2` 被 Claim–Evidence Verifier 判为矛盾并拒绝。

合并计算得到：16 个成功 / 20 个可执行 AUTO Patch，确定性修复率 `80.00%`。被拒绝的错误重连证明语义门禁仍然生效，并未为了提高修复率而放宽证据标准。

## 拒绝情况

- 当前 Blue 拒绝记录：21 条。
- 上一轮修正 Trace 重复计数后：43 条。
- 降低：51.16%。
- 当前 `unbalanced_markdown_bold`：0 条；上一轮原始 Trace：30 条。
- 当前没有 `claim_target_not_unique`。
- 剩余拒绝主要是语义验证器拦截 partially supported、uncited、contradicted Claim，以及重复 Claim 完整性门禁。

## 质量护栏

Dynamic Swarm 和 Fixed fallback 都完成了最终质量比较：

- Dynamic：Claim 支持率 93.88%，引用覆盖率 97.96%，blocker 0。
- Fixed fallback：Claim 支持率 97.83%，引用覆盖率 100%，blocker 9。

系统没有被更高的单项支持率误导，而是优先选择 blocker 为 0 的 Dynamic 报告。`cost_used_as_quality_override=false`，成本没有覆盖质量决策。

## 尚未解决的限制

- A/B 证据覆盖率仍为 66.67%，缺少 2 个比较单元。
- 最终仍有 1 条无引用 Claim，使引用覆盖率从 100% 降至 97.96%。
- 最终报告存在少量内容重复，例如 ReferenceGrant 机制被连续表述两次；未构成 blocker，但应进入下一轮写作质量清理。
- 费率未配置，运行只能报告 Token，不能给出可信美元成本。

## 完整性与复现

- Replay：`trace_integrity_verified=true`。
- Trace Bundle 校验：通过，0 个错误。
- 没有发生整篇重写 fallback。
- 最终复用了 45 条未修改 Claim 的判断，只验证新增或变更 Claim。
- 运行耗时 293.818 秒；Harness 预算计时 285.593 秒。
