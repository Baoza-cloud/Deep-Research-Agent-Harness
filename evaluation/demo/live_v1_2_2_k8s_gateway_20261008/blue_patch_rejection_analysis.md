# Blue Patch 拒绝分析与确定性修复复验

分析对象：`demo-v122-k8s-gateway-20261008-claimfix`

数据来源：本目录的 `raw_run.json`、持久化中间报告和最终 Red Review。本文中的“复验”是基于已保存报告、证据与 Review 的离线回放，不等同于重新运行 Tavily + DeepSeek 在线实验。

## 1. 61 条拒绝的真实口径

原始 Trace 包含 61 条拒绝记录，但其中两轮 `deterministic_claim_patch` 在“同时有成功和拒绝”时重复写入了拒绝列表：

- Round 2 重复 10 条。
- Round 3 重复 8 条。
- 去除事件级重复后为 43 条拒绝记录。
- 按 `patch_id + reason` 去重后为 41 个独立拒绝结果。

本次已把编排器的第二个 `if` 改为 `elif`。新运行不会再重复记录同一轮 deterministic rejection。

| 原始拒绝原因 | 次数 | 判断 |
|---|---:|---|
| `post_patch_integrity:unbalanced_markdown_bold` | 30 | Claim 切分缺陷，不是模型内容错误 |
| `claim_evidence_validation_failed:*` | 20 | 安全门禁正确拒绝，不应放宽 |
| `claim_target_not_unique` | 6 | 目标解析能力不足，且含重复 Trace |
| 其他 Markdown 重复 Claim | 2 | 完整性门禁正确拒绝 |
| `target_not_found` | 1 | 模型复制的目标已过期或不精确 |
| `target_ambiguous` | 1 | 目标不唯一 |
| `semantic_verifier_unavailable` | 1 | 语义验证不可用时的安全降级 |

去掉重复 Trace 后的 43 条构成为：20 条语义验证拒绝、17 条完整性拒绝、3 条目标不唯一，以及 3 条其他定位/验证拒绝。

## 2. 根因

### 2.1 Markdown 粗体边界被错误切开

旧切分器会在 `。**` 中的句号后切分，生成只有开头 `**`、没有闭合 `**` 的 `source_text`。Blue 删除或修改该片段时，完整性门禁正确报出 `unbalanced_markdown_bold`。

保存的 7 版中间报告在旧逻辑下每版有 17–19 个不平衡 Claim target；新逻辑下均为 0。

### 2.2 Red target 是语义位置，Blue 只接受逐字目标

真实 Review 使用了以下定位方式：

- `执行摘要及“一、API 成熟度”第 3 条`
- `“三、跨命名空间安全”第 2 条`
- `建议第 2 条`

旧确定性修复器只能匹配完整原句，因此无法处理这些高价值问题。

### 2.3 引用错配与推断越界缺少安全 fallback

旧实现只有严重事实错误和未知引用的有限 fallback。Structured Blue 生成的补丁一旦被语义门禁拒绝，引用错配与推断越界就会留到最终报告。

## 3. 本次实现

- Markdown-aware Claim 切分：禁止在闭合 `**` 前切分，并在闭合标记后安全分句。
- Red 目标解析：支持逐字 Claim、描述中的问题引文、`章节第 N 条` 三种保守定位。
- 推断越界：删除问题 Claim；复合 Claim 只移除越界子句，保留独立兄弟断言。
- 引用错配：事实句删除错配子句；建议句保留建议，只移除装饰性引用。
- 明示证据重连：当 Red 明确指出“实际由/需要某 Evidence 直接支持”时，程序生成引用替换 Patch，并通过 Claim–Evidence Verifier 后应用。
- 误删保护：若 Red 自己承认被引述事实已获证据支持，则不执行确定性删除。
- 已由 Structured Blue 成功处理的 `issue_id` 不再被 fallback 重复修改。
- JSON 解析失败、`patches` 类型错误或 Blue LLM 不可用时，也会执行同一套安全 fallback。

## 4. 保存运行的离线复验

对最终报告、73 条证据和最终 Red Review 进行离线回放：

- 生成并成功应用 14 个确定性 Patch。
- 覆盖 13/14 个阻断型 issue ID。
- 2 个显式支持证据引用被安全重连。
- 3 个引用错配问题被删除或解除装饰性引用。
- 8 个推断越界 Patch 覆盖 7 个 issue ID，并执行删除或局部收窄。
- 1 个严重版本事实错误被删除。
- 0 个 Patch 引入 Markdown 完整性错误。
- 剩余 1 个问题 `R5` 被有意跳过：Red 描述本身承认证据支持该事实，直接删除会构成误修复。

该结果证明修复策略能覆盖已保存事故，但不能替代新的在线 A/B 运行。正式指标需要用当前代码重新执行同一 Tavily + DeepSeek 演示，并由新一轮 Red Review 验证最终 blocker 数量。

## 5. 回归门禁

新增测试覆盖：

- 粗体 Claim target 始终成对闭合。
- 语义章节 locator 能定位并删除推断越界。
- 删除引用错配子句时保留同段正确兄弟事实。
- 建议文本仅解除装饰性引用，不删除建议。
- Red 明示的直接支持 Evidence 可经验证后重连。
- Red 自己承认支持的事实不得被确定性删除。
- 无引用、未知引用、低严重度和逻辑类问题仍不能触发高危事实删除。

当前全套离线测试：151 项通过。
