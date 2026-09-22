# Enterprise-LLM-RAG-Assistant

> 新增：基于现有 Hybrid RAG 的 Deep Research Agent Harness，覆盖动态 Swarm 编组与预算控制、DAG 规划与异步执行、共享记忆、Claim–Evidence Ledger、结构化 Red/Blue 对抗修复和三层评测。详见 [Deep Research Agent](docs/DEEP_RESEARCH_AGENT.md)。

ResearchBench 已拆分为 Frozen、Adversarial 与 Live 三条 Track；Frozen v1.0 覆盖 11 领域/35 题，Adversarial v0.1 从同一题集确定性注入 140 个带 gold ID 的引用、事实与完整性缺陷，可分别评测 Red 检出率、Blue 修复率、误修率与回滚正确率。

heuristic-v4 的正式 Fixed Harness vs Dynamic Swarm 对照已完成：35 题 × 3 次、两组共 210 次执行，运行异常、阻塞 Red 问题与 Claim 冲突均为 0。Dynamic 的综合质量差异不显著（Δ -0.13 个百分点，95% CI [-0.34, +0.07]，paired Cohen's dz=-0.20），Worker 调用稳定降低 16.39%（95% CI 10.39%–22.13%）；LLM 调用、估算 Token、成本与平均延迟的点估计分别降低 7.64%、8.12%、7.09% 和 5.45%，但这些效率指标的置信区间跨 0，不宣称显著改善。详见[正式 35×3 对照摘要](evaluation/results/formal_frozen_v1_35x3/ablation-20260922T135223799454Z-summary.md)与[严格质量门禁](evaluation/results/formal_frozen_v1_35x3/ablation-20260922T135223799454Z-quality-gate.md)。

35 题 × 3 次 DeepSeek 正式在线对抗实验已完成：四个变体共 420 份报告全部成功。当前安全版本加入确定性 uncertainty ADD、整行 high-severity factual DELETE、citation-only DELETE 拒绝与未知引用支持验证后，Structured Patch Blue 的保守字符串 fault repair 从 86.43% 提升到 99.29%（配对 Δ +12.86 个百分点，题目级 Bootstrap 95% CI：+10.00–+15.71），规则事实准确率从 97.34% 提升到 100%（95% CI：+0.97–+4.64），规则幻觉率降至 0；事实与引用保留率均为 100%，collateral damage 为 0。剩余字符串未命中主要是“并非错误结论”这类安全否定式纠正，不再通过子串删除追求虚假的 100%。详见 [正式四变体实验](evaluation/results/adversarial/adversarial-20260916T031514890111Z.md)和[当前安全版本前后对照](evaluation/results/adversarial/adversarial-20260916T065323635595Z-vs-adversarial-20260916T024120075281Z.md)。

## 安装与运行

项目采用标准 `src-layout`。在项目根目录执行一次可编辑安装：

```bash
python3 -m pip install -e .
```

如需复用原有 Dense + BM25 + RRF 本地检索，再安装本地 RAG 可选依赖：

```bash
python3 -m pip install -e ".[local-rag]"
```

安装后可使用任一正式入口：

```bash
deep-research "什么是 RAG？" --offline
python3 -m research_engine "什么是 RAG？" --offline
```

`src/research_engine/` 下除 `__main__.py` 外均为包内模块，不应使用
`python src/research_engine/agents.py` 等方式逐文件运行。IDE 中也应将启动目标配置为模块
`research_engine`，避免破坏相对导入上下文；个人 IDE 运行配置不纳入版本控制。


## 1. 项目简介

本项目旨在构建一个企业级大语言模型知识库问答系统。

针对大语言模型存在的以下问题：

- 知识更新困难
- 无法直接访问企业私有数据
- 容易产生幻觉（Hallucination）

本项目结合：

- RAG（Retrieval-Augmented Generation，检索增强生成）
- Embedding向量检索
- FAISS向量数据库
- LoRA参数高效微调

实现一个基于企业文档知识库的智能问答系统。

---

## 2. 项目目标

实现一个完整的大模型应用流程：

用户问题

↓

问题理解

↓

知识库检索

↓

相关文档召回

↓

大语言模型生成答案

↓

返回带来源的回答


最终实现：

- 企业文档问答
- 私有知识库检索
- 降低模型幻觉
- 支持知识动态更新


---

## 3. 技术路线


原始企业文档

↓

PDF/文本解析

↓

文本清洗

↓

Chunk切分

↓

Embedding向量化

↓

FAISS建立索引

↓

Retriever检索

↓

Prompt构造

↓

LLM生成

↓

LoRA微调优化


---

## 4. 技术栈


### 编程语言

Python


### 深度学习框架

PyTorch


### 大模型相关

- Transformers
- Qwen
- PEFT(LoRA)


### RAG相关

- LangChain
- FAISS
- BGE Embedding


### 部署

- Streamlit


---

## 5. 项目开发阶段


### Phase 1：基础RAG系统

完成：

- PDF文档读取
- 文本切分
- Embedding
- 向量数据库
- 相似度检索
- LLM回答


### Phase 2：RAG优化

增加：

- Hybrid Search
- BM25检索
- Rerank模型
- Prompt优化


### Phase 3：模型微调

使用：

LoRA + SFT

优化模型对于特定领域问题的回答能力。


### Phase 4：系统评估与部署

包括：

- 检索效果评估
- 回答质量评估
- Web Demo


---

## 6. 项目结构

```text
Enterprise-LLM-RAG-Assistant

├── data
├── src
├── retrieval
├── finetuning
├── evaluation
├── deployment
├── scripts
└── docs
