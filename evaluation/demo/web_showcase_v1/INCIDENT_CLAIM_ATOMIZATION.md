# 事故复盘：中文复合 Claim 导致日期限定漏检

## 影响

首次真实 Web 运行生成了以下事实断言：

> Gateway API v1.5 于 2026 年 2 月 27 日发布，将部分特性移入 Stable。

它引用的 `t7-E2` URL 路径为 `kubernetes.io/blog/2026/04/21/gateway-api-v1-5`。人工复核发现日期
不一致，但最终 Claim Ledger 的 `time_conflict_count` 为 0，说明自动验证出现假阴性。该运行未作为最终演示。

## 根因

原 `CLAIM_SPLIT` 只在句号后存在空白且后面不是 Citation 时分句。中文报告常使用：

```text
事实 A。 [facts-E1] 事实 B。 [facts-E2]
```

因为句号后紧跟 Citation，四个事实句被合并成一个 Claim。Ledger 的聚合规则在至少一个 cited link 获得
完整支持时可能把整个复合 Claim 判为 supported，使局部日期限定被其他子句连带通过。

## 修复

- 中文句号、问号和感叹号允许无空格分句。
- Citation 仍保留在前一句；连续 Citation 不会被拆散。
- 英文句点仍要求后续空白，避免把 `v1.0` 和小数拆开。
- 增加 `test_claim_extraction_atomizes_chinese_sentences_with_trailing_citations` 回归测试。

修复提交：`ad3331f`。首次运行源提交：`5eab26f9b6f05601f07b4a54d52dd355473ef790`。

## 同口径复跑

使用相同问题、模型、Tavily 深度、并发、超时和预算复跑：

| 指标 | 修复前 | 修复后 |
|---|---:|---:|
| 提取 Claim | 41 | 90 |
| 可审查 Claim | 36 | 82 |
| Red Review 轮次 | 4 | 2 |
| 最终 Red 结果 | 未通过，低严重度问题残留 | 通过，0 个 Red issue |
| 停止原因 | `review_budget_exhausted` | `review_passed` |
| 最终 Claim 支持率 | 97.22% | 97.56% |
| 最终引用覆盖率 | 97.30% | 98.78% |

两次 Live 运行不能用于推断平均提升；这里的复跑只用于验证缺陷修复和完整链路。修复后报告不再包含错误的
v1.5 日期断言，但仍保留一条部分支持和一条无引用 Claim，因此状态仍为
`completed_with_evidence_gaps`。

## 完整性

- 首次原始运行 SHA-256：`b357e71de9ac101b660b28791a5a7499b88e4be811ae3ae5e7207302fe40c87d`
- 修复后原始运行 SHA-256：记录在 [`artifact_manifest.json`](artifact_manifest.json)

原始网页正文不进入公开仓库；公开 Trace 保留来源 URL、内容哈希、判断、Patch 和最终状态。
