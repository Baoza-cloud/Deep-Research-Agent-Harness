# 面试问答

## 项目介绍

### 30 秒版本

我做的是一个可评测的 Deep Research Agent Harness，不是固定的 LangChain 工作流。系统先生成研究 DAG，
再由 SwarmPolicy 根据复杂度选择 Agent 数量、角色、并发和预算；检索结果经过角色阈值、权威度和去重过滤，
报告生成后逐句建立 Claim–Evidence Ledger，再由 Red Agent 找问题、Blue Agent 通过结构化 Patch 做确定性修复。
我用 Frozen、Live、Retrieval-Gold 和 Adversarial 四条评测线验证质量、成本与失败恢复，并发布了可复现哈希。

### 90 秒版本

普通 Agent Demo 的问题是流程不可控、错误不可定位，也很难公平评测。我把系统拆成五层：Harness 契约、
DAG 调度、动态 Swarm、证据与记忆、质量门禁。执行层用 asyncio 和 Semaphore 做拓扑并发，并用九状态
状态机管理节点生命周期；策略层根据问题和 DAG 的复杂度选择 1/3/5 个角色以及调用预算；质量层不是只看
LLM 自评，而是建立 Claim–Evidence Ledger，检查部分支持、缺证据以及时间、数字、实体冲突。Red Agent
输出结构化 Issue，Blue Agent 只返回 ADD/DELETE/MODIFY/VERIFY Patch，程序验证后再修改报告。正式 35×3
实验中，Dynamic Swarm 在质量差异不显著的情况下将 Worker 调用降低 16.39%；对抗评测中，安全补丁将
故障修复率从 86.43% 提升到 99.29%，并保持事实和引用 100% 不被误删。

## 架构与技术难点

### 1. 为什么不用 LangChain 直接串 Agent？

LangChain 可以作为工具层，但我的核心问题是可控执行和可证伪评测。Harness 显式定义 AgentSpec、RunContext、
ToolRegistry、预算和状态转换；调度、停止、回退和 Patch 校验都是程序逻辑，不依赖框架黑盒。这样才能说明
一次调用为什么发生、为什么停止，以及哪项策略真正改善了指标。

### 2. 这是真正的多 Agent，还是同一个模型换几个 Prompt？

底层模型可以相同，但 Agent 的角色契约、查询改写、检索阈值、工具权限和预算不同，并且任务由 DAG 路由。
我不把“不同系统 Prompt”本身当作贡献；贡献是角色行为可以在 Trace 中观察，并能用消融实验比较固定编排、
动态 Swarm、无 Red/Blue、Rewrite Blue 与 Structured Patch Blue。

### 3. Dynamic Swarm 如何决定 Agent 数量？

HeuristicSwarmPolicy 对 DAG 大小、问题长度、比较、时效、高风险、多维度和约束数量打分。低、中、高复杂度
分别选择 1、3、5 个角色，并同时设置并发、Worker、Replan、Review、Verification 和总耗时预算。策略版本被
写入结果，防止实验复用不同口径。

### 4. DAG 并发如何保证依赖正确？

Planner 输出依赖更早节点的有向无环图；执行前用 Kahn 算法检测未知依赖和环。运行时只调度依赖已成功的节点，
Semaphore 控制并发，状态机记录 pending、running、succeeded、failed 等转换。批量失败达到阈值才 Replan，
单任务超时、批量失败和全局超时分别对应三级降级。

### 5. 最难的问题是什么？

最难的不是生成长报告，而是统一 Red、Rule Evaluator 和 Claim Verifier 对“事实 Claim”的口径。早期系统会把
“当前证据无法确认”这样的 Gap 声明当成无支撑事实，造成假阴性。我统一了 Claim 识别、给 Gap 单独状态，
并只对新增或修改 Claim 做增量验证，避免修复后再次攻击诚实的不确定性。

### 6. Claim Verifier 如何判断支持关系？

先做引用和词法对齐，再对候选 Claim–Evidence 对进行语义判断，输出 supported、partially_supported、
insufficient 或 contradicted，并检测时间、数字、实体冲突。判断按 Claim + Evidence 缓存；未变化 Claim 直接复用。
最终硬门禁看支持率和冲突，而不是仅看引用格式。

### 7. 为什么 Blue Agent 使用 Patch，而不是重写全文？

全文重写会引入回归，而且无法定位哪条修改修复了哪个问题。Patch 包含 action、target、replacement、issue_ids、
evidence_ids 和 reason。程序校验目标唯一性、Evidence 合法性、增长上限和修复后的 Claim Ledger；失败 Patch
被拒绝。这样可以统计修复率和干净事实保留率。

### 8. 如何处理检索噪声？

每个角色先改写查询，再计算关键词、实体、Provider 分数、来源权威度和角色内容匹配。随后执行阈值过滤、
规范 URL 去重、同域重复控制和权威来源排序。Trace 保存过滤前后数量、每条淘汰原因和角色阈值。

### 9. 如何防 Prompt Injection？

Web 内容进入模型前被当作不可信数据，检测“忽略指令、泄露密钥、输出系统提示”等信号并替换危险片段；
Trace 只记录信号和内容哈希，不记录 Key。真实演示中检测并清洗了一条 secret-exfiltration 页面。

### 10. 为什么同时保留 Rule 和 LLM Verifier？

Rule 快、稳定、适合引用格式与显式冲突；语义 Verifier 能处理同义词、跨语言和部分支持。只用 Rule 假阴性高，
只用 LLM 又成本高且不稳定，所以先粗筛，再只调用语义判断处理候选对，并缓存结果。

## 评测与取舍

### 11. 如何证明 Dynamic Swarm 有用？

同一 Frozen 数据、模型、费率、并发和随机口径下做 35 题 × 3 次配对实验。Dynamic 与 Fixed 的综合质量
差异为 -0.13 pp，95% CI 跨 0，不能说质量提升；但 Worker 调用降低 16.39%，95% CI [10.39%, 22.13%]。
正确结论是它在质量基本持平时减少了 Worker，而不是“多 Agent 提升质量”。

### 12. 为什么不是把 105 次运行当作 105 个独立样本？

三次重复共享同一道题，不独立。先对题内重复求均值，再把 35 道题作为独立单位做配对 Bootstrap；同时报告
95% CI 和 Cohen's dz，避免伪造样本量。

### 13. LLM-as-Judge 是否会偏？

会。因此我把它放在规则指标和 Claim Ledger 之后，只作为多维质量信号；正式结论同时报告事实准确率、幻觉率、
引用覆盖率、支持率和保留率。下一步会加入独立 Judge 后端和一部分人工双标数据。

### 14. 为什么 Final Guard 有时回退 Fixed？

Dynamic 的资源节省不能覆盖质量下降。当引用覆盖率低于 80%、支持率明显低于 Fixed 或存在阻塞 Issue 时，系统
生成 Fixed 候选并逐项比较。首次故障运行中 Fixed 引用覆盖率更高，但有 22 个阻塞问题，因此仍保留 Dynamic；
修复后的最终演示在第二轮已经通过，没有触发 fallback。

### 15. 为什么最终状态不是简单成功/失败？

证据本身不足与系统生成错误不是一回事。状态拆成 completed、completed_with_evidence_gaps、
completed_with_review_issues 和 partial_timeout，便于分别统计召回问题、生成问题和基础设施问题。

## 事故复盘

### 16. 讲一个真实事故：评测假阴性

**现象**：语义 Claim 支持率达到 100%，规则指标只有 50%。
**根因**：规则把“证据未提供”“当前无法确认”等 Gap 声明识别成事实 Claim，同时缺少 consumer group/组等别名。
**修复**：统一 Rule Evaluator、Claim Ledger 和 Red Agent 的 Claim 识别；建立别名表；Gap 单独分类；新增回归测试。
**结果**：Gap 不再被重复攻击，真正的数值、时间、实体冲突仍会触发处理。

### 17. 讲一个实验事故：漂亮数字不能直接写简历

**现象**：早期 heuristic-v4 回放显示 Worker 调用降低 22.64%、Token 降低 9.96%。
**根因**：结果来自兼容回放，不是新代码在相同批次的完整运行。
**处理**：冻结数据集与统计协议，重新运行 35×3 Fixed/Dynamic，并保存数据、代码和产物哈希。
**最终结论**：正式 Worker 降幅是 16.39%，而不是 22.64%；我只在简历使用正式结果。

### 18. 这次真实 Web 演示暴露了什么问题？

首次运行发现中文句号后紧跟引用时，多个句子会被合并成复合 Claim，导致日期限定可能被其他已支持子句连带通过。
我修复了 Claim 分句规则并加回归测试，再用同一问题和预算复跑；Claim 数从 41 增加到 90，第二轮 Red Review
通过。复跑仍留下一个部分支持和一个无引用 Claim，因此状态保持为 evidence gaps，而不是为了演示改成满分。

## 工程边界

### 19. 为什么使用 SQLite + NumPy，而不是直接上向量数据库？

目标是单机可复现和降低部署复杂度。SQLite 支持事务和可审计持久化，NumPy 足够覆盖当前规模；接口与存储实现
分离，规模扩大后可以替换为专用向量库。现在提前引入分布式数据库反而会掩盖 Harness 本身的贡献。

### 20. 如果上线，还缺什么？

运行时 Checkpoint/Resume、Provider 限流与熔断、OpenTelemetry、任务队列、多租户隔离、真实账单校准和更大的
人工 Gold。当前版本定位为可复现研究与面试项目，不宣称已经具备生产 SLA。

## 简历表述

设计并实现可复现的 Deep Research Agent Harness：基于 asyncio + Semaphore 执行动态 DAG，按复杂度自适应
选择 1/3/5 类研究角色和预算；构建 Claim–Evidence Verifier 与确定性 Red/Blue Patch 门禁。在 35 题 × 3 次
配对实验中，Dynamic Swarm 在综合质量差异不显著的情况下将 Worker 调用降低 16.39%（95% CI 10.39%–22.13%）；
对抗评测将故障修复率从 86.43% 提升到 99.29%，事实与引用保留率均为 100%。
