# Kubernetes Ingress vs Gateway API：v1.2.1 真实 Web 演示

本目录记录一次 Tavily + DeepSeek 的真实在线研究运行。问题要求基于官方资料，从
API 成熟度、资源模型、跨命名空间安全和实现兼容性四个维度对比 Kubernetes
Ingress 与 Gateway API。

## 结果定位

这是一次“质量门禁发现错误”的演示，不是一次全绿质量基准。运行在 142.85 秒内完成，
Claim 支持率和引用覆盖率均为 100%，但 Red Agent 识别出一条严重事实错误：报告把证据
中的三种稳定 API 类别写成四种。Harness 因此返回
`completed_with_review_issues`，而没有把报告标记为可发布。

## 可公开文件

- `trace_bundle.json`：脱敏结构化运行包；保留 DAG、遥测、检索审计、Claim Ledger、
  Red/Blue 事件与最终报告，不包含抓取的网页正文。
- `trace_viewer.html`：无需服务器的本地 Trace Viewer，展示 DAG 时间线、Agent 并发瀑布、
  角色决策、fallback 和停止原因。
- `final_report.md`：系统原始报告，并在顶部保留质量门禁状态和阻断问题。
- `artifact_manifest.json`：源代码 commit、原始运行哈希、公开包哈希和摘要指标。
- `ONE_PAGE_RESULTS.md`：一页核心结论与失败分析。
- `DEMO_SCRIPT.md`：2–3 分钟面试演示讲稿。

## 仅本地保存

`raw_run.json`、`replay.json` 和各次失败尝试文件由本目录 `.gitignore` 排除。它们包含网页
抓取正文或完整持久化状态，不适合在未逐页核对授权条款时重新分发。最终 Replay 已验证
`trace_integrity_verified=true`，且不会再次调用模型或检索。

## 运行身份

- Engine：`v1.2.1`
- Source commit：`74e493646e75f08962fbfff601033db2ad2a6af9`
- Run ID：`demo-v121-k8s-gateway-20261006-release`
- Model：`deepseek-v4-flash`
- Search：Tavily advanced，仅允许 `kubernetes.io` 与 `gateway-api.sigs.k8s.io`

所有哈希以 `artifact_manifest.json` 和发布目录的 `SHA256SUMS` 为准。
