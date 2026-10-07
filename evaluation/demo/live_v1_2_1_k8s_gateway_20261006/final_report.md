> **运行状态：`completed_with_review_issues`。** 本报告未通过发布门禁；保留原始输出用于演示 Harness 如何暴露质量问题。
> Claim 支持率 100.00%；引用覆盖率 100.00%；A/B 证据覆盖率 87.50%；阻断问题 1 个。
> - [R1] 报告称“Gateway API 具有四种稳定的 API 类别”，但所引证据 gateway_model-E1 与 gateway_maturity-E1 均只列出三种稳定 API 类别（GatewayClass、Gateway、HTTPRoute）。将三类写成四类属于事实错误。

# Kubernetes Ingress 与 Gateway API 对比研究报告

## 执行摘要

Ingress 是 Kubernetes 中已进入稳定状态（v1.19 stable）的 API 对象，用于管理集群中服务的外部访问，典型方式是 HTTP [ingress_model-E2]。Gateway API 是一组 API 类别，提供动态基础设施制备和高级流量路由，被定位为 Ingress API 的后继者 [gateway_model-E1]。Gateway API 的 Gateway、GatewayClass 和 HTTPRoute 已升级到 v1 版本，即正式发布（GA）状态 [gateway_maturity-E4]。Gateway API 通过角色导向、可移植、表达能力强和可扩展的设计原则，提供了 Ingress 之外的能力 [gateway_model-E1]。在跨命名空间安全方面，Gateway API 通过 ReferenceGrant 资源实现跨命名空间引用的显式授权 [gateway_security-E3]。在实现兼容性方面，Gateway API 定义了分级的一致性要求 [compatibility_counterexamples-E7]。

---

## 关键发现

### 1. API 成熟度

**Ingress：** Ingress 的特性状态为 `Kubernetes v1.19 [stable]`，是对集群中服务的外部访问进行管理的 API 对象，典型访问方式是 HTTP，可以提供负载均衡、SSL 终结和基于名称的虚拟托管 [ingress_model-E2]。IngressClass 代表 Ingress 的类，被 Ingress 的规约引用，其 `apiVersion` 为 `networking.k8s.io/v1` [ingress_maturity-E1]。

进入 Beta 阶段的标志包括：API 已广泛实现、合规性测试基本覆盖了所有资源且可以让多种实现通过测试、大多数 API 接口正被积极地使用、Kubernetes SIG Network API 评审团队已批准其进入 Beta 阶段 [gateway_maturity-E2]。该版本引入了 `experimental` 和 `standard` 发布渠道，`standard` 渠道包括已进入 Beta 阶段的资源和已进入 standard 的字段，`experimental` 渠道包括 `standard` 渠道的所有内容以及 alpha API 资源和视为 experimental 且还未进入 `standard` 渠道的字段 [gateway_maturity-E2]。

Gateway API v1.0 版本将 Gateway、GatewayClass 和 HTTPRoute 升级到 v1 版本，即正式发布（GA）状态；标准通道中所包含的该版本 API 集合被认为是稳定的，但这并不意味着它们是完整的 [gateway_maturity-E4]。

### 2. 核心资源模型

**Ingress 资源模型：** Ingress 使用一种能感知协议配置的机制来解析 URI、主机名称、路径等 Web 概念，允许通过 Kubernetes API 定义的规则将流量映射到不同后端 [ingress_model-E2]。IngressClass 代表 Ingress 的类，被 Ingress 的规约引用；当某个 IngressClass 资源将 `ingressclass.kubernetes.io/is-default-class` 注解设置为 true 时，没有指定类的新 Ingress 资源将被分配到此默认类 [ingress_maturity-E1]。

**Gateway API 资源模型：** Gateway API 具有四种稳定的 API 类别 [gateway_model-E1]。GatewayClass 定义一组具有配置相同的网关，由实现该类的控制器管理 [gateway_model-E1]。Gateway 可以由不同的控制器实现，通常具有不同的配置 [gateway_model-E1]。HTTPRoute 定义特定于 HTTP 的规则，用于将流量从网关监听器映射到后端网络端点的表示 [gateway_maturity-E1]。一个 Gateway 对象只能与一个 GatewayClass 相关联 [gateway_maturity-E1]。

Gateway API 的设计和架构遵从角色导向原则，基于负责管理 Kubernetes 服务网络的组织角色建模：基础设施提供者管理使用多个独立集群为多个租户提供服务的基础设施；集群操作员管理集群，通常关注策略、网络访问、应用程序权限等；应用程序开发人员管理在集群中运行的应用程序，通常关注应用程序级配置和 Service 组合 [gateway_model-E1]。

### 3. 跨命名空间安全

Gateway API 在跨命名空间引用方面有明确的规则。 [gateway_security-E1] 对于跨命名空间边界的 ParentRefs，跨命名空间引用只有在被引用命名空间中的某些内容显式允许时才有效；例如 Gateway 具有 AllowedRoutes 字段，ReferenceGrant 提供了一种通用方式来启用其他类型的跨命名空间引用 [gateway_security-E4]。

ReferenceGrant 资源存在于目标命名空间中，用于完成跨命名空间引用所需的握手 [gateway_security-E3]。ReferenceGrant 资源是 GA 的，自 `v0.6.0` 起成为 Standard Channel 的一部分 [gateway_security-E5]。该资源最初命名为 "ReferencePolicy"，后更名为 "ReferenceGrant" 以避免与策略附加产生混淆 [gateway_security-E5]。

在 Gateway API 中，允许跨命名空间边界的场景包括：Gateways 引用 Secrets 和 Routes 引用 Backends（通常是 Services），这些情况下所需的握手通过 ReferenceGrant 资源完成 [gateway_security-E3]。

### 4. 实现兼容性

Gateway API 定义了分级的一致性要求：core（强制支持）、extended（如果支持则可移植）和 custom（无可移植性保证），统称为灵活一致性（flexible conformance）；这促进了一个高度可移植的核心 API（类似 Ingress），同时仍为 Gateway 控制器实现者提供灵活性 [compatibility_counterexamples-E7]。

Gateway API 涵盖广泛的功能并得到广泛实现，这种组合需要明确的标准合规性定义和测试，以确保 API 在任何地方使用时都能提供一致的体验 [recommendation-E4]。

提及从 Ingress 迁移到 Gateway API。 [compatibility_counterexamples-E4] Gateway API 提供了模块化、可扩展的 API，并对 Kubernetes 原生 RBAC 有强支持；相反，Ingress API 简单，Ingress-NGINX 等实现通过注解、ConfigMaps 和 CRD 来扩展 API [compatibility_counterexamples-E5]。

---

## 反方证据/局限

1. **Gateway API 标准通道的完整性：** 虽然标准通道中所包含的 v1 版本 API 集合被认为是稳定的，但这并不意味着它们是完整的 [gateway_maturity-E4]。

3. **迁移复杂性：** 从 Ingress-NGINX 等 Ingress 控制器迁移面临捕获 Ingress 控制器所有细微差别并将其行为映射到 Gateway API 的艰巨任务 [compatibility_counterexamples-E5]。

5. **服务网格一致性挑战：** GAMMA 面临的挑战之一是许多测试都认为一个给定实现会提供 Ingress 控制器，而许多服务网格不提供 Ingress 控制器 [compatibility_counterexamples-E8]。

---

## 结论与建议

### 结论

Ingress 是已进入稳定状态的 API，提供基本的 HTTP 流量管理能力 [ingress_model-E2]。Gateway API 作为其后继者，已有多项核心资源进入 GA 状态 [gateway_maturity-E4]，通过角色导向的设计 [gateway_model-E1]、显式的跨命名空间引用授权机制 [gateway_security-E3] 分级一致性框架（flexible conformance）；提及一致性配置文件（conformance profiles）。 [compatibility_counterexamples-E7] [compatibility_counterexamples-E8]

### 建议

1. **评估现有 Ingress 使用情况后规划迁移路径。** 参考 Ingress 迁移指南，将现有 Ingress 资源一次性转换为 Gateway API 资源 [gateway_model-E1]。

2. **在采用 Gateway API 前确认所选实现的一致性级别。** 查阅合规性相关文档以了解发布渠道、支持级别和运行合规性测试等详细信息 [recommendation-E4]。

3. **利用 ReferenceGrant 机制设计跨命名空间访问控制。** 对于需要跨命名空间引用的场景，在目标命名空间中创建 ReferenceGrant 资源以显式授权 [gateway_security-E3]。

4. **安装 Gateway API CRD 并查阅所选实现的文档。** Gateway API 是基于 CRD 的 API，需要安装 CRD 到集群上才能使用 [gateway_maturity-E2]；请务必查看所选实现的文档以了解可能存在的注意事项 [gateway_model-E1]。

5. **关注发布渠道区分，在生产环境中使用 standard 渠道资源。** `standard` 渠道包括已进入 Beta 阶段的资源和已进入 standard 的字段 [gateway_maturity-E2]。

---

## 来源

1. [ingress_model-E2] Ingress | Kubernetes — https://v1-34.docs.kubernetes.io/zh-cn/docs/concepts/services-networking/ingress
2. [ingress_maturity-E1] IngressClass | Kubernetes — https://v1-32.docs.kubernetes.io/zh-cn/docs/reference/kubernetes-api/service-resources/ingress-class-v1
3. [gateway_model-E1] Gateway API | Kubernetes — https://kubernetes.io/zh-cn/docs/concepts/services-networking/gateway
4. [gateway_maturity-E1] Gateway API | Kubernetes — https://v1-31.docs.kubernetes.io/zh-cn/docs/concepts/services-networking/gateway
5. [gateway_maturity-E2] Kubernetes Gateway API 进入 Beta 阶段 | Kubernetes — https://v1-35.docs.kubernetes.io/zh-cn/blog/2022/07/13/gateway-api-graduates-to-beta
6. [gateway_maturity-E4] Gateway API v1.0：正式发布（GA） | Kubernetes — https://v1-33.docs.kubernetes.io/zh-cn/blog/2023/10/31/gateway-api-ga
7. [gateway_security-E3] Security | Gateway API — https://gateway-api.sigs.k8s.io/docs/concepts/security
8. [gateway_security-E4] API Reference | Gateway API — https://gateway-api.sigs.k8s.io/reference/api-spec/main/spec
9. [gateway_security-E5] ReferenceGrant | Gateway API — https://gateway-api.sigs.k8s.io/reference/api-types/referencegrant
10. [compatibility_counterexamples-E5] Announcing Ingress2Gateway 1.0 | Kubernetes — https://kubernetes.io/blog/2026/03/20/ingress2gateway-1-0-release
11. [compatibility_counterexamples-E7] Evolving Kubernetes networking with the Gateway API | Kubernetes — https://kubernetes.io/blog/2021/04/22/evolving-kubernetes-networking-with-the-gateway-api
12. [compatibility_counterexamples-E8] Gateway API v0.8.0：引入服务网格支持 | Kubernetes — https://kubernetes.io/zh-cn/blog/2023/08/29/gateway-api-v0-8
13. [recommendation-E4] Gateway API | Kubernetes — https://v1-33.docs.kubernetes.io/docs/concepts/services-networking/gateway

## 证据索引

- `ingress_maturity-E1` [IngressClass | Kubernetes](https://v1-32.docs.kubernetes.io/zh-cn/docs/reference/kubernetes-api/service-resources/ingress-class-v1)
- `ingress_maturity-E4` [IngressClass | Kubernetes](https://v1-31.docs.kubernetes.io/zh-cn/docs/reference/kubernetes-api/service-resources/ingress-class-v1)
- `gateway_maturity-E1` [Gateway API | Kubernetes](https://v1-31.docs.kubernetes.io/zh-cn/docs/concepts/services-networking/gateway)
- `gateway_maturity-E2` [Kubernetes Gateway API 进入 Beta 阶段 | Kubernetes](https://v1-35.docs.kubernetes.io/zh-cn/blog/2022/07/13/gateway-api-graduates-to-beta)
- `gateway_maturity-E4` [Gateway API v1.0：正式发布（GA） | Kubernetes](https://v1-33.docs.kubernetes.io/zh-cn/blog/2023/10/31/gateway-api-ga)
- `ingress_model-E1` [IngressClass | Kubernetes](https://kubernetes.io/zh-cn/docs/reference/kubernetes-api/service-resources/ingress-class-v1)
- `ingress_model-E2` [Ingress | Kubernetes](https://v1-34.docs.kubernetes.io/zh-cn/docs/concepts/services-networking/ingress)
- `gateway_model-E1` [Gateway API | Kubernetes](https://kubernetes.io/zh-cn/docs/concepts/services-networking/gateway)
- `ingress_security-E1` [Ingress | Kubernetes](https://kubernetes.io/zh-cn/docs/concepts/services-networking/ingress)
- `ingress_security-E2` [Gateway API | Kubernetes](https://kubernetes.io/docs/concepts/services-networking/gateway)
- `ingress_security-E3` [Gateway API v1.5: Moving features to Stable | Kubernetes](https://kubernetes.io/blog/2026/04/21/gateway-api-v1-5)
- `ingress_security-E4` [Gateway API v1.2: WebSockets, Timeouts, Retries, and More | Kubernetes](https://kubernetes.io/blog/2024/11/21/gateway-api-v1-2)
- `ingress_security-E5` [Migrating from Ingress | Gateway API](https://gateway-api.sigs.k8s.io/guides/getting-started/migrating-from-ingress)
- `gateway_security-E1` [Gateway API | Kubernetes](https://kubernetes.io/zh-cn/docs/concepts/services-networking/gateway)
- `gateway_security-E2` [GEP-709: Cross Namespace References from Routes | Gateway API](https://gateway-api.sigs.k8s.io/geps/gep-709)
- `gateway_security-E3` [Security | Gateway API](https://gateway-api.sigs.k8s.io/docs/concepts/security)
- `gateway_security-E4` [API Reference | Gateway API](https://gateway-api.sigs.k8s.io/reference/api-spec/main/spec)
- `gateway_security-E5` [ReferenceGrant | Gateway API](https://gateway-api.sigs.k8s.io/reference/api-types/referencegrant)
- `compatibility_counterexamples-E4` [Gateway API | Kubernetes](https://v1-34.docs.kubernetes.io/zh-cn/docs/concepts/services-networking/gateway)
- `compatibility_counterexamples-E5` [Announcing Ingress2Gateway 1.0: Your Path to Gateway API | Kubernetes](https://kubernetes.io/blog/2026/03/20/ingress2gateway-1-0-release)
- `compatibility_counterexamples-E7` [Evolving Kubernetes networking with the Gateway API | Kubernetes](https://kubernetes.io/blog/2021/04/22/evolving-kubernetes-networking-with-the-gateway-api)
- `compatibility_counterexamples-E8` [Gateway API v0.8.0：引入服务网格支持 | Kubernetes](https://kubernetes.io/zh-cn/blog/2023/08/29/gateway-api-v0-8)
- `recommendation-E3` [Gateway API v1.1：服务网格、GRPCRoute 和更多变化 | Kubernetes](https://v1-31.docs.kubernetes.io/zh-cn/blog/2024/05/09/gateway-api-v1-1)
- `recommendation-E4` [Gateway API | Kubernetes](https://v1-33.docs.kubernetes.io/docs/concepts/services-networking/gateway)
- `recommendation-E5` [Kubernetes Gateway API Graduates to Beta | Kubernetes](https://kubernetes.io/blog/2022/07/13/gateway-api-graduates-to-beta)
- `verify_r1_1-E1` [Ingress | Kubernetes](https://v1-31.docs.kubernetes.io/zh-cn/docs/reference/kubernetes-api/service-resources/ingress-v1)
- `verify_r1_1-E2` [Ingress | Kubernetes](http://kubernetes.io/zh-cn/docs/reference/kubernetes-api/service-resources/ingress-v1)
- `verify_r1_3-E1` [Gateway API | Kubernetes](https://v1-32.docs.kubernetes.io/zh-cn/docs/concepts/services-networking/gateway)
