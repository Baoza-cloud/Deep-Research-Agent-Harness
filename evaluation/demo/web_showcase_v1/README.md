# Real Web Showcase v1

这是一个可复验的真实 Web 演示快照。问题不是为了迎合系统能力而简化，而是要求同时处理
API 成熟度、资源模型、流量能力、安全边界、实现兼容性、迁移步骤、实验特性与证据缺口：

> 截至 2026-10-05，对比 Kubernetes Ingress 与 Gateway API。请基于真实 Web 资料分析
> API 成熟度、核心资源模型、流量能力、跨命名空间安全、实现兼容性、迁移步骤与仍属
> 实验性的限制；给出生产迁移建议，并明确区分已证实事实、合理推断和证据缺口。

## 快照结论

- 运行时间：2026-10-05 08:52–08:54（Asia/Shanghai），有效执行耗时 126.362 秒。
- 后端：DeepSeek `deepseek-v4-flash` + Tavily Advanced Search。
- 状态：`completed_with_evidence_gaps`，没有把证据缺口伪装成完整成功。
- DAG：8 个节点；Dynamic Swarm 判定为 `complex`，选择 5 类角色，最大并发 4。
- 检索：12 批、104 个候选，保留 52、淘汰 52；16 个保留结果属于官方文档。
- 验证：82 条可审查 Claim 中 80 条获得完整支持，Claim 支持率 97.56%，引用覆盖率 98.78%。
- Red/Blue：第 2 轮审查通过；应用 8 个确定性补丁，其中 DELETE 1、MODIFY 7，另有 1 个目标不唯一的 Patch 被拒绝；没有整篇重写。
- 增量验证：最终复用了 80 条未修改 Claim 的判断，没有重复验证整份报告。
- 停止条件：`review_passed`；本次没有触发 Fixed fallback，但相关护栏仍由 Frozen 正式实验和首次故障运行覆盖。

## 文件

- [`trace_bundle.json`](trace_bundle.json)：完整 DAG、角色策略、检索过滤事件、Claim Ledger、Red/Blue Patch、质量候选与最终报告。原始网页正文已移除。
- [`final_report.md`](final_report.md)：最终研究报告及 Evidence ID 到真实 URL 的索引。
- [`artifact_manifest.json`](artifact_manifest.json)：源代码提交、原始运行、Trace Bundle 和报告 SHA-256。
- [`DEMO_SCRIPT.md`](DEMO_SCRIPT.md)：2–3 分钟现场讲稿。
- [`TRACE_GUIDE.md`](TRACE_GUIDE.md)：演示时应该打开的 JSON 路径。
- [`ONE_PAGE_RESULTS.md`](ONE_PAGE_RESULTS.md)：正式实验结论与真实失败案例。
- [`INTERVIEW_QA.md`](INTERVIEW_QA.md)：项目介绍、技术难点、取舍和事故复盘问答。
- [`INCIDENT_CLAIM_ATOMIZATION.md`](INCIDENT_CLAIM_ATOMIZATION.md)：演示发现、修复并同口径复跑的 Claim 原子化事故。

`raw_run.json` 保留在本地并被 `.gitignore` 排除。原因是它包含 Tavily 返回的网页正文；公开仓库只保存
内容哈希、来源、URL、评分、淘汰原因和完整执行事件，以减少不必要的第三方内容再分发。原始运行的
SHA-256 已写入 `artifact_manifest.json`，可以验证本地原件是否被修改。

## 复跑

从仓库根目录执行；Key 只放在 `evaluation/.env`，不要写入命令或提交：

```bash
PYTHONPATH=src python3 -m research_engine \
  '截至 2026-10-05，对比 Kubernetes Ingress 与 Gateway API。请基于真实 Web 资料分析 API 成熟度、核心资源模型、流量能力、跨命名空间安全、实现兼容性、迁移步骤与仍属实验性的限制；给出生产迁移建议，并明确区分已证实事实、合理推断和证据缺口。' \
  --provider deepseek \
  --model deepseek-v4-flash \
  --search-provider tavily \
  --web-search-depth advanced \
  --concurrency 4 \
  --task-timeout 60 \
  --global-timeout 600 \
  --max-review-rounds 4 \
  --min-citation-coverage 0.80 \
  --min-claim-support-rate 0.80 \
  --max-swarm-agents 5 \
  --max-worker-invocations 18 \
  --memory /tmp/deep-research-web-showcase-v1.sqlite3 \
  --output evaluation/demo/web_showcase_v1/raw_run.json
```

重新生成公开 Bundle：

```bash
python3 evaluation/demo/build_web_showcase.py \
  evaluation/demo/web_showcase_v1/raw_run.json \
  --output-dir evaluation/demo/web_showcase_v1 \
  --source-commit ad3331f3a148561d15d8698618979a47754232e4

python3 evaluation/demo/validate_web_showcase.py \
  evaluation/demo/web_showcase_v1/trace_bundle.json
```

Live Web 会受时间、索引和网页内容变化影响。复跑结果不应覆盖本快照，也不应与 Frozen 正式指标合并。
