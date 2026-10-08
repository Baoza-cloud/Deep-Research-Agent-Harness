> **运行状态：`completed_with_evidence_gaps`。** 本报告未通过发布门禁；保留原始输出用于演示 Harness 如何暴露质量问题。
> Claim 支持率 93.88%；引用覆盖率 97.96%；A/B 证据覆盖率 66.67%；阻断问题 0 个。

# Kubernetes Ingress 与 Gateway API 对比深度研究报告

**报告日期：2026-10-08**

## 执行摘要

本报告基于 Kubernetes 与 Gateway API 官方资料，从 API 成熟度、核心资源模型、跨命名空间安全和实现兼容性四个维度对比 Ingress 与 Gateway API。Gateway API v1.0 版本正式发布（GA）[t1-E5]。Gateway API 是一组 API 类别，可提供动态基础设施制备和高级流量路由 [t3-E3]。 Ingress 是 Kubernetes 中管理集群外部访问的 API 对象，典型访问方式为 HTTP [t2-E1]。Ingress-NGINX 计划于 2026 年 3 月正式退役 [t6-E3]。Gateway API 安全模型文档说明，跨命名空间引用所需的安全握手通过 ReferenceGrant 资源完成 [t4-E4]。多个实现已支持 Gateway API，包括 AWS Load Balancer Controller [t5-E2]。迁移工具 ingress2gateway 已发布 1.0 版本 [t6-E3]。

## 关键发现

### 一、API 成熟度

**已证实事实：**

Gateway API v1.0 版本正式发布（GA）[t1-E5]。证据明确说明此版本引入了 experimental 和 standard 两个发布渠道，并分别描述了 standard 和 experimental 渠道包含的内容；证据在发布渠道部分明确区分实验性渠道和标准渠道，并说明功能从实验性渠道毕业到标准渠道；证据明确提到标准（Standard）通道和实验（Experimental）通道，并说明新特性通过实验通道接收；standard 渠道包括已进入 Beta 阶段的资源和已进入 standard 的字段；存在 standard 和 experimental 发布渠道；`experimental` 渠道包括 `standard` 渠道的所有内容，另外还有 alpha API 资源和视为 experimental 且还未进入 `standard` 渠道的字段。 [t1-E1] [t5-E3]

Gateway API 的版本控制策略规定：次要版本中，标准渠道的字段或资源从实验性渠道毕业到标准渠道，并根据 Kubernetes 弃用策略删除 API 资源；主要版本更改时，没有 API 兼容性保证 [t5-E3]。

Ingress 的 API 版本为 `networking.k8s.io/v1` [t2-E4]。Ingress 资源本身仅创建没有任何效果，可能需要部署 Ingress 控制器 [t2-E1]。

### 二、核心资源模型

**已证实事实：**

GatewayClass 定义了一组共享通用配置并由实现该类的控制器管理的网关；Gateway 定义了流量处理基础设施的实例，如云负载均衡器；HTTPRoute 定义了将流量从网关映射到服务的 HTTP 特定规则 [t1-E2]。

Gateway API 的设计遵循角色导向原则：基础设施提供者管理使用多个独立集群为多个租户提供服务的基础设施；集群操作员管理集群，通常关注策略、网络访问、应用程序权限等；应用程序开发人员管理在集群中运行的应用程序，通常关注应用程序级配置和 Service 组合。 [t3-E3] [verify_r1_3-E3]

Gateway API 支持多种协议：HTTP/HTTPS 通过 HTTPRoute 资源，TLS 通过 TLSRoute 资源，TCP 通过 TCPRoute 资源，UDP 通过 UDPRoute 资源。 [t7-E4]

Ingress 是允许入站连接到达后端定义的端点的规则集合，可以配置为向服务提供外部可访问的 URL、负载均衡流量、终止 SSL、提供基于名称的虚拟主机等 [t2-E4]。`ingressClassName` 是 IngressClass 集群资源的名称，Ingress 控制器实现使用此字段来了解它们是否应该为该 Ingress 资源提供服务 [t2-E2]。

### 三、跨命名空间安全

**已证实事实：**

Gateway API 安全模型文档说明，跨命名空间引用所需的安全握手通过 ReferenceGrant 资源完成，该资源存在于目标命名空间中 [t4-E4]。 在这些情况下，所需的安全握手通过 ReferenceGrant 资源完成，该资源存在于目标命名空间中 [t4-E4]。

面向角色的设计允许集群运营人员定义许多不同非协调开发者团队如何使用共享基础架构 [t3-E2]。

**证据缺口：**

关于 Ingress 在跨命名空间安全方面的具体机制，现有证据未提供直接说明。 [t1-E5] [verify_r1_4-E2] Gateway API 的 ReferenceGrant 机制在防止未授权跨命名空间引用方面的具体安全保证，需要查阅 Gateway API 安全模型文档进一步验证。 [t1-E5]

### 四、实现兼容性

**已证实事实：**

Gateway API 是一种开源标准，有很多实现方案，可使概念和核心资源在不同实现方案和环境中保持一致 [t3-E2]。AWS Load Balancer Controller 对 Gateway API 的支持已对 Layer 4（L4）和 Layer 7（L7）路由达到 GA，使客户能够直接从 Kubernetes 集群配置和管理 AWS NLB 和 ALB [t5-E2]。

Gateway API 进入 Beta 阶段的达成标准之一是合规性测试基本覆盖了所有资源且可以让多种实现通过测试 [t1-E1]。当考虑 Gateway API 一致性时，该 API 覆盖了广泛的功能和用例，并已被广泛实现 [t5-E4]。

证据缺口：现有证据未提供 Gateway API 各实现一致性认证状态的直接官方说明。

Ingress 控制器实现使用 `ingressClassName` 字段来了解它们是否应该为该 Ingress 资源提供服务 [t2-E2]。

证据缺口：现有证据未提供 Ingress 跨命名空间安全机制的直接官方说明。

## 反方证据/局限

1. **Ingress 的持续存在**：Ingress 作为 Kubernetes 已有 API [t1-E4]，其 `networking.k8s.io/v1` 版本仍然有效 [t2-E4]。对于已使用 Ingress 且功能满足需求的场景，迁移的必要性取决于具体需求。

2. **Gateway API 的版本兼容性**：Gateway API 主要版本更改时没有 API 兼容性保证 [t5-E3]。

3. **证据覆盖不完整**：本报告未涵盖 Gateway API 所有实验性功能的成熟度评估，也未涵盖各实现的具体一致性认证状态。关于 Ingress 跨命名空间安全机制的官方说明在现有证据中缺失。

## 结论与建议

### 已证实事实总结

- Gateway API 通过使用可扩展的、角色导向的、协议感知的配置机制来提供网络服务 [t3-E3]
- Gateway API 通过 ReferenceGrant 实现跨命名空间引用 [t4-E4]
- AWS Load Balancer Controller 对 Gateway API 的支持已对 Layer 4（L4）和 Layer 7（L7）路由达到 GA，使客户能够直接从 Kubernetes 集群配置和管理 AWS NLB 和 ALB [t5-E2]
- Ingress-NGINX 计划于 2026 年 3 月正式退役 [t6-E3]
- ingress2gateway 工具已发布 1.0 版本 [t6-E3]

### 合理推断

- 待验证问题：Gateway API 的标准化特性是否降低对特定实现的依赖，需查阅具体实现的一致性认证报告。
- 待验证问题：角色分离模型是否改善多团队协作效率，需查阅组织实践案例。

### 建议与验证路径

1. 迁移时，需要将现有 Ingress 资源一次性转换为 Gateway API 资源 [t6-E1]。

2. 请参阅合规性相关的文档，以了解发布渠道、支持级别和运行合规性测试等详细信息；毕业标准包括完整的符合性测试覆盖率、多个符合性实现。

3. Ingress2Gateway 是一款辅助工具，旨在帮助团队从 Ingress API 迁移到 Gateway API；ingress2gateway 通过将现有的 Ingress 资源转换为 Gateway API 资源来帮助进行迁移。 [t6-E3] [t6-E5] 转换现有 Ingress 配置，验证转换结果的正确性。

4. **分阶段迁移**：先在测试环境验证 Gateway API 配置，再逐步迁移生产环境。 [t7-E4]

5. **补充证据缺口**：查阅 Gateway API 安全模型文档，确认 ReferenceGrant 的具体安全保证和 Ingress 跨命名空间访问的官方说明。

## 来源

本报告引用的证据来源：

1. [t1-E1] Kubernetes Gateway API 进入 Beta 阶段 | Kubernetes — https://v1-35.docs.kubernetes.io/zh-cn/blog/2022/07/13/gateway-api-graduates-to-beta
2. [t1-E2] Gateway API | Kubernetes — https://kubernetes.io/docs/concepts/services-networking/gateway
3. [t1-E4] Kubernetes Gateway API 深入解读和落地指南 - 云原生社区 — https://cloudnativecn.com/blog/kubernetes-gateway-api-explained
4. [t1-E5] Gateway API v1.0：正式发布（GA） | Kubernetes — https://k8s.io/zh-cn/blog/2023/10/31/gateway-api-ga
5. [t2-E1] Ingress | Kubernetes — https://v1-31.docs.kubernetes.io/zh-cn/docs/concepts/services-networking/ingress
6. [t2-E2] Ingress | Kubernetes — https://v1-35.docs.kubernetes.io/zh-cn/docs/reference/kubernetes-api/service-resources/ingress-v1
7. [t2-E4] Ingress | Kubernetes — https://kubernetes.ac.cn/zh-cn/docs/reference/kubernetes-api/service-resources/ingress-v1
8. [t3-E2] Gateway API 简介 | GKE networking | Google Cloud Documentation — https://docs.cloud.google.com/kubernetes-engine/docs/concepts/gateway-api?hl=zh-cn
9. [t3-E3] Gateway API | Kubernetes — https://kubernetes.io/zh-cn/docs/concepts/services-networking/gateway
10. [t4-E4] Security | Gateway API — https://gateway-api.sigs.k8s.io/docs/concepts/security
11. [t5-E2] Implementations | Gateway API — https://gateway-api.sigs.k8s.io/docs/implementations/list
12. [t5-E3] 版本控制 - Kubernetes 网关 API — https://gateway-api.kubernetes.ac.cn/concepts/versioning
13. [t5-E4] Conformance | Gateway API — https://gateway-api.sigs.k8s.io/docs/concepts/conformance
14. [t6-E3] Ingress2Gateway 1.0 正式发布：通往 Gateway API 的途径 | Kubernetes — https://kubernetes.io/zh-cn/blog/2026/03/20/ingress2gateway-1-0-release
16. [t7-E4] Gateway API | Kubernetes指南 — https://kubernetes.feisky.xyz/concepts/objects/gateway-api

## 证据索引

- `t1-E1` [Kubernetes Gateway API 进入 Beta 阶段 | Kubernetes](https://v1-35.docs.kubernetes.io/zh-cn/blog/2022/07/13/gateway-api-graduates-to-beta)
- `t1-E2` [Gateway API | Kubernetes](https://kubernetes.io/docs/concepts/services-networking/gateway)
- `t1-E3` [Gateway API vs Ingress 在服务网格中的选型：从稳定性、功能到 Ambient 模式的深度对比 - WEBKT](https://www.webkt.com/article/13955)
- `t1-E4` [Kubernetes Gateway API 深入解读和落地指南 - 云原生社区](https://cloudnativecn.com/blog/kubernetes-gateway-api-explained)
- `t1-E5` [Gateway API v1.0：正式发布（GA） | Kubernetes](https://k8s.io/zh-cn/blog/2023/10/31/gateway-api-ga)
- `t2-E1` [Ingress | Kubernetes](https://v1-31.docs.kubernetes.io/zh-cn/docs/concepts/services-networking/ingress)
- `t2-E2` [Ingress | Kubernetes](https://v1-35.docs.kubernetes.io/zh-cn/docs/reference/kubernetes-api/service-resources/ingress-v1)
- `t2-E3` [kubernetes-handbook/service-discovery/ingress/index.md ...](https://github.com/rootsongjc/kubernetes-handbook/blob/main/service-discovery/ingress/index.md)
- `t2-E4` [Ingress | Kubernetes](https://kubernetes.ac.cn/zh-cn/docs/reference/kubernetes-api/service-resources/ingress-v1)
- `t2-E5` [Ingress简介 | CloudNative knowledge](https://cloudnative365.github.io/keynotes_L3_senior_3_ingress_1_ingress.html)
- `t3-E1` [Gateway API介绍-容器计算服务 ACS-阿里云帮助中心](https://help.aliyun.com/zh/cs/user-guide/gateway-api-management)
- `t3-E2` [Gateway API 简介  |  GKE networking  |  Google Cloud Documentation](https://docs.cloud.google.com/kubernetes-engine/docs/concepts/gateway-api?hl=zh-cn)
- `t3-E3` [Gateway API | Kubernetes](https://kubernetes.io/zh-cn/docs/concepts/services-networking/gateway)
- `t3-E5` [k8s 中的 Gateway API 的背景和简介【k8s 系列之四】 - 橙子家 - 博客园](https://www.cnblogs.com/hnzhengfy/p/k8s_gatewayapi.html)
- `t4-E1` [Gateway API 简介  |  GKE networking  |  Google Cloud Documentation](https://docs.cloud.google.com/kubernetes-engine/docs/concepts/gateway-api?hl=zh-cn)
- `t4-E2` [Kubernetes Gateway API vs Ingress](https://imesh.ai/blog/kubernetes-gateway-api-vs-ingress)
- `t4-E3` [Ingress控制器与Gateway API : r/kubernetes](https://www.reddit.com/r/kubernetes/comments/1kri73b/ingress_controller_v_gateway_api?tl=zh-hans)
- `t4-E4` [Security | Gateway API](https://gateway-api.sigs.k8s.io/docs/concepts/security)
- `t4-E5` [Gateway API: Future of Kubernetes Ingress](https://tetrate.io/blog/why-the-gateway-api-is-the-unified-future-of-ingress-for-kubernetes-and-service-mesh)
- `t5-E1` [Gateway API 简介  |  GKE networking  |  Google Cloud Documentation](https://docs.cloud.google.com/kubernetes-engine/docs/concepts/gateway-api?hl=zh-cn)
- `t5-E2` [Implementations | Gateway API](https://gateway-api.sigs.k8s.io/docs/implementations/list)
- `t5-E3` [版本控制 - Kubernetes 网关 API](https://gateway-api.kubernetes.ac.cn/concepts/versioning)
- `t5-E4` [Conformance | Gateway API](https://gateway-api.sigs.k8s.io/docs/concepts/conformance)
- `t5-E5` [API Reference | Gateway API](https://gateway-api.sigs.k8s.io/reference/api-spec/main/spec)
- `t6-E1` [Gateway API | Kubernetes](https://kubernetes.io/zh-cn/docs/concepts/services-networking/gateway)
- `t6-E2` [将 Ingress 迁移到 Gateway API  |  GKE networking  |  Google Cloud Documentation](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/migrate-ingress-gateway?hl=zh-cn)
- `t6-E3` [Ingress2Gateway 1.0 正式发布：通往 Gateway API 的途径 | Kubernetes](https://kubernetes.io/zh-cn/blog/2026/03/20/ingress2gateway-1-0-release)
- `t6-E4` [告别 Ingress-NGINX：用 Amazon Load Balancer Controller Gateway API 实现更强大的流量管理 | 亚马逊AWS官方博客](https://aws.amazon.com/cn/blogs/china/ingress-nginx-to-alb-controller-with-gateway-api)
- `t6-E5` [Kubernetes Gateway API 正式发布并引入 ingress2gateway 项目用于简化 Gateway API 升级 | 云原生社区（中国）](https://cloudnative.jimmysong.io/blog/gateway-api-ingress2gateway)
- `t7-E1` [Higress 已支持全新 Gateway API / Inference Extension](https://higress.ai/blog/higress-gvr7dx_awbbpb_vbzalyy37xxuoswm)
- `t7-E2` [〔CKS 筆記整理〕從 Ingress 到 Gateway API：Kubernetes 流量管理架構的演進與未來 ｜Marcos的方格子](https://vocus.cc/article/6938fdeafd89780001a6638e)
- `t7-E3` [从 Ingress 迁移到 Gateway API | Jimmy Song](https://jimmysong.io/zh/book/kubernetes-handbook/service-discovery/migrating-from-ingress-to-gateway-api)
- `t7-E4` [Gateway API | Kubernetes指南](https://kubernetes.feisky.xyz/concepts/objects/gateway-api)
- `t7-E5` [Migrating from Ingress | Gateway API](https://gateway-api.sigs.k8s.io/guides/getting-started/migrating-from-ingress)
- `verify_r1_1-E1` [Ingress - Kubernetes](https://people.wikimedia.org/~jayme/k8s-docs/v1.16/zh/docs/concepts/services-networking/ingress)
- `verify_r1_1-E2` [Ingress | Kubernetes](https://kubernetes.io/zh-cn/docs/concepts/services-networking/ingress)
- `verify_r1_1-E3` [Ingress | Kubernetes](https://kubernetes.io/docs/concepts/services-networking/ingress)
- `verify_r1_2-E2` [API Reference | Gateway API](https://gateway-api.sigs.k8s.io/reference/api-spec/main/spec)
- `verify_r1_2-E3` [GEP-709: Cross Namespace References from Routes | Gateway API](https://gateway-api.sigs.k8s.io/geps/gep-709)
- `verify_r1_3-E1` [Gateway API 简介  |  GKE networking  |  Google Cloud Documentation](https://docs.cloud.google.com/kubernetes-engine/docs/concepts/gateway-api?hl=zh-cn)
- `verify_r1_3-E2` [关于由 llm-d 提供支持的 GKE Inference Gateway  |  GKE networking  |  Google Cloud Documentation](https://docs.cloud.google.com/kubernetes-engine/docs/concepts/about-gke-inference-gateway?hl=zh-cn)
- `verify_r1_3-E3` [Gateway API | Kubernetes](https://kubernetes.io/zh-cn/docs/concepts/services-networking/gateway)
- `verify_r1_4-E1` [Gateway API | Kubernetes](https://kubernetes.io/docs/concepts/services-networking/gateway)
- `verify_r1_4-E2` [Gateway API 简介  |  GKE networking  |  Google Cloud Documentation](https://docs.cloud.google.com/kubernetes-engine/docs/concepts/gateway-api?hl=zh-cn)
- `verify_r1_4-E3` [API Overview | Gateway API](https://gateway-api.sigs.k8s.io/docs/concepts/api-overview)
- `verify_r2_1-E1` [Ingress - Kubernetes](https://people.wikimedia.org/~jayme/k8s-docs/v1.16/zh/docs/concepts/services-networking/ingress)
- `verify_r2_1-E2` [Referring to TLS secret from other namespace (i.e. not the namespace in which ingress is created) · Issue #2170 · kubernetes/ingress-nginx · GitHub](https://github.com/kubernetes/ingress-nginx/issues/2170)
- `verify_r2_1-E3` [Ingress | Kubernetes指南](https://kubernetes.feisky.xyz/concepts/objects/ingress)
- `verify_r2_3-E1` [Conformance | Gateway API](https://gateway-api.sigs.k8s.io/docs/concepts/conformance)
- `verify_r2_3-E2` [Getting started with Gateway API | Gateway API](https://gateway-api.sigs.k8s.io/guides/getting-started/introduction)
- `verify_r2_3-E3` [API Reference | Gateway API](https://gateway-api.sigs.k8s.io/reference/api-spec/main/spec)
- `verify_r2_4-E1` [gateway-api/config/crd/experimental/gateway.networking.k8s.io_grpcroutes.yaml at main · kubernetes-sigs/gateway-api · GitHub](https://github.com/kubernetes-sigs/gateway-api/blob/main/config/crd/experimental/gateway.networking.k8s.io_grpcroutes.yaml)
- `verify_r2_4-E2` [GRPCRoute - Amazon VPC Lattice Gateway API Controller](https://www.gateway-api-controller.eks.aws.dev/dev/api-types/grpc-route)
- `verify_r2_4-E3` [gRPC routing | Gateway API](https://gateway-api.sigs.k8s.io/guides/user-guides/grpc-routing)
- `verify_r3_1-E1` [跨集群部署 Ingress  |  GKE networking  |  Google Cloud Documentation](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/multi-cluster-ingress?hl=zh-cn)
- `verify_r3_1-E2` [Ingress - Kubernetes](https://people.wikimedia.org/~jayme/k8s-docs/v1.16/zh/docs/concepts/services-networking/ingress)
