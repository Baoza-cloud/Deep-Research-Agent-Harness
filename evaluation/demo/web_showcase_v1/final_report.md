# Kubernetes Ingress 与 Gateway API 深度对比研究报告

**报告日期基准：2026-10-05**

## 执行摘要

证据讨论了 Ingress 与 Gateway API 在流量管理方面的差异，并说明 Gateway API 支持更高级的流量管理功能；证据讨论了 Ingress 与 Gateway API 在流量管理方面的差异，并说明 Gateway API 的模块化设计。 [t2-E4] [verify_r1_1-E2]Gateway API 由 Kubernetes SIG-NETWORK 管理，自 2023 年 GA 以来已支持多协议路由、角色分离和灵活策略配置 [t2-E2]。Gateway API 具有四种稳定的 API 类别，包括 HTTPRoute 等资源 [t1-E1]。相比 Ingress，Gateway API 在表现力、扩展性、角色分离、通用性、基础设施共享和类型化后端引用等方面均有显著提升，并支持跨命名空间路由绑定 [t2-E2]。

在功能对比层面，Gateway API 原生支持流量拆分、URL 重写、Header 路由和跨命名空间路由，而 Ingress 需要依赖 annotation 实现部分功能 [t3-E3]。Gateway API 支持 HTTP/HTTPS/gRPC/TCP/UDP 多协议，而 Ingress 仅支持 HTTP/HTTPS [t3-E3]。

跨命名空间安全方面，Gateway API 要求在每一道命名空间边界上显式授权，Route 引用另一命名空间的 Gateway 需通过 `allowedRoutes` 选择，Route 引用另一命名空间的后端需通过 `ReferenceGrant` 允许 [t4-E2]。`ReferenceGrant` 为此提供了保障，实现需要非常小心以避免混淆代理攻击 [t4-E3]。

实现兼容性方面，许多 Ingress controller 正在积极支持 Gateway API，包括 Istio、Kong、Traefik 等 [t3-E4]。APISIX Ingress Controller 对 Gateway API 的支持处于 Alpha 阶段 [t1-E4]。

迁移方面，Kubernetes 开源社区发布了 ingress2gateway 工具，可将集群中的 Ingress 批量转换到 Gateway API 资源 。 对于仍在 EKS 集群内使用 Ingress-NGINX 的用户，可选择迁移到 Amazon Load Balancer Controller 继续使用 Ingress，或转向使用 Gateway API 。 [t3-E3]

## 关键发现

### 1. API 成熟度

Gateway API 自 2023 年 GA 以来，已支持多协议路由、角色分离和灵活策略配置 [t2-E2]。Gateway API 具有四种稳定的 API 类别 [t1-E1]。Gateway API v1.0 的 GA 发布主要聚焦于确保现有 beta API 定义良好且足够稳定以毕业为 GA [t7-E3]。

实验渠道（Experimental channel）是 Gateway API 用于试验新功能的渠道，以便在功能成熟之前积累足够信心，再将其升级为 Standard 渠道功能 [t7-E1]。使用新的实验渠道资源意味着它们可以与 Standard 渠道资源共存，若要将这些资源迁移到 Standard 渠道，则需要使用 Standard 渠道的名称和 API 组来重新创建它们 [t7-E1]。实验渠道包括标准发布渠道中的所有内容，外加一些实验性资源和字段 [t7-E4]。

API 参考中标注为 Experimental 的字段包括：`useDefaultGateways` 字段出现在 GRPCRouteSpec、HTTPRouteSpec、TCPRouteSpec 等中，`defaultScope` 字段出现在 GatewaySpec 中 [t7-E5]。

### 2. 核心资源模型

Gateway API 由多种资源组成，分别承担不同的网络管理职责 [t2-E2]。GatewayClass 定义网关类，由基础设施提供方创建，支持参数化配置 [t2-E2]。Gateway API 将网络配置分解为不同关注点，实现配置解耦和角色分离 [t2-E2]。

GatewayClass 由基础设施提供方创建；Gateway 由集群运维人员管理；HTTPRoute 由应用开发者创建。 [t2-E2] [t2-E1]

Gateway API 将 Ingress 资源的功能拆分为专用 CRD——GatewayClass、Gateway 和 Route [t2-E4]。Gateway 充当网络端点，将流量从集群外部带入集群内部；Route 资源帮助定义从 Gateway 到后端的 HTTP 或 TCP 路由 [t2-E4]。

Gateway API 引入了面向角色的架构（GatewayClass、Gateway、Route 资源），清晰地将基础设施所有权与应用路由关注点分离 [t2-E3]。

### 3. 流量能力

Gateway API 资源可以为基于标头的匹配、流量加权、CORS 和其他只能在 Ingress 中通过自定义注解实现的功能提供内置功能 [t2-E1]。Gateway API 的核心功能包含诸如基于头的匹配、流量加权以及其他在 Ingress 中只能通过各实现者自定义的非标准化 Annotations 等方式实现的功能 [t3-E4]。

功能对比显示：流量拆分在 Ingress 中需要 annotation，在 Gateway API 中原生支持；URL 重写在 Ingress 中需要 annotation，在 Gateway API 中原生支持；金丝雀发布在 Ingress 中需要多个 Ingress，在 Gateway API 中通过单个 HTTPRoute 实现；Header 路由在 Ingress 中需要 annotation，在 Gateway API 中原生支持 [t3-E3]。

Gateway API 支持下游（客户端到网关）和上游（网关到后端服务）的 TLS 配置，支持终止、透传、通配符证书和跨命名空间证书引用 [t2-E2]。

Gateway API v1.3.0 引入了流量复制功能，可以通过调整分数来实现部分流量复制，例如在发送到 “foo-v1” 的请求中，将每 1000 个中的 5 个复制到 “foo-v2” [t7-E1]。

### 4. 跨命名空间安全

跨命名空间引用让平台团队可以运营共享 Gateway、后端和 Secret，同时让应用团队管理各自的 Route 与 Consumer [t4-E2]。Gateway API 要求在每一道命名空间边界上显式授权，避免在一个命名空间创建资源就自动获得另一团队资源的访问权 [t4-E2]。

不同关系所需的授权方式：Route 引用另一命名空间的 Gateway 时，Gateway 监听器通过 `allowedRoutes` 选择 Route 所在命名空间；Route 引用另一命名空间的后端时，后端命名空间中的 `ReferenceGrant` 允许该 Route 引用 [t4-E2]。

为确保 Gateway API 能够安全地提供此功能，需要强制执行握手机制，要求两个命名空间中的资源都同意此引用，为此引入了新的 ReferenceGrant 资源 [t4-E5]。在某些情况下，忽略 `ReferenceGrant` 而采用其他安全机制可能是可以接受的，只有当 `NetworkPolicy` 等其他机制可以通过实现有效限制跨命名空间引用时，才可以这样做 [t4-E3]。选择做出此例外的实现必须清楚地记录其实现不遵守 `ReferenceGrant`，详细说明可用的替代保护措施 [t4-E3]。此 API 的实现需要非常小心，以避免混淆代理攻击，`ReferenceGrant` 为此提供了保障 [t4-E3]。

功能对比表显示，跨命名空间路由在 Ingress 中不支持，在 Gateway API 中支持 [t3-E3]。

### 5. 实现兼容性

许多 Ingress controller 正在积极支持 Gateway API，包括 Istio、Kong、Traefik 等 [t3-E4]。Emissary-Ingress（以前称为 Ambassador API Gateway）是一个开源 CNCF 项目，基于 Envoy Proxy，为 Kubernetes 提供入门控制器和 API 网关 [t1-E3]。Kong 在 Kong Kubernetes 入门控制器 (KIC) 中支持 Gateway API [t1-E3]。

该项目的目的是实现核心 Gateway API——Gateway、GatewayClass、HTTPRoute、TCPRoute、TLSRoute 和 UDPRoute——以配置 HTTP 或 TCP/UDP 负载均衡器、反向代理或运行在 Kubernetes 上的应用程序的 API 网关 [t1-E3]。

Ingress NGINX 尚未计划支持 Gateway API [t3-E4]。APISIX Ingress 已经支持了 Gateway API 的大部分特性，包括 HTTPRoute、TCPRoute、TLSRoute、UDPRoute 等 [t3-E4]。APISIX Ingress Controller 对 Gateway API 的支持正在开发中，处于 Alpha 阶段，目前已支持 HTTPRoute、TCPRoute 等资源 [t1-E4]。在 APISIX Ingress Controller 中，默认没有启用 Gateway API 的支持，可通过参数 `--enable-gateway-api=true` 启用 [t1-E4]。

NGINX 官方推出了 NGINX Gateway Fabric 项目（GA）来支持核心 Gateway API 功能；社区版 ingress-nginx 本身主要仍用旧有 Ingress 资源 [t5-E3]。

### 6. 迁移步骤

Kubernetes 开源社区发布了一个 ingress2gateway 的工具，这个工具可以将 Kubernetes 集群中的 Ingress 批量转换到 Gateway API 资源 [t3-E3]。

对于仍在 Amazon EKS 集群内使用 Ingress-NGINX 组件的用户，有两种方案可供选择：迁移到 Amazon Load Balancer Controller 继续使用 Ingress，或转向使用 Gateway API [t3-E3]。

迁移步骤包括：安装 Amazon Load Balancer Controller，安装完成后启用 Gateway API 功能 [t3-E3]。验证安装时需要检查 controller 状态和 CRD 是否安装 [t3-E3]。

在 Gateway API 中，需要使用 Amazon LBC 专有的 TargetGroupConfiguration CRD 来配置健康检查参数（如检查路径、间隔、超时、健康/不健康阈值等）[t3-E3]。

### 7. 实验性限制

实验渠道（Experimental channel）是 Gateway API 用于试验新功能的渠道 [t7-E1]。使用新的实验渠道资源意味着它们可以与 Standard 渠道资源共存，若要将这些资源迁移到 Standard 渠道，则需要使用 Standard 渠道的名称和 API 组来重新创建它们 [t7-E1]。

API 参考中标注为 Experimental 的字段包括：`useDefaultGateways` 字段出现在 GRPCRouteSpec、HTTPRouteSpec、TCPRouteSpec 等中，`defaultScope` 字段出现在 GatewaySpec 中 [t7-E5]。

实验渠道包括标准发布渠道中的所有内容，外加一些实验性资源和字段 [t7-E4]。

## 反方证据 / 局限

1. **Ingress NGINX 不支持 Gateway API**：Ingress NGINX 尚未计划支持 Gateway API [t3-E4]。社区版 ingress-nginx 本身主要仍用旧有 Ingress 资源 [t5-E3]。这意味着大量使用 Ingress NGINX 的用户无法直接迁移到 Gateway API，需要更换实现。

2. **APISIX Ingress 的 Gateway API 支持处于 Alpha 阶段**：APISIX Ingress Controller 对 Gateway API 的支持正在开发中，处于 Alpha 阶段 [t1-E4]。在 APISIX Ingress Controller 中，默认没有启用 Gateway API 的支持 [t1-E4]。

3. **ReferenceGrant 的例外情况**：在某些情况下，忽略 `ReferenceGrant` 而采用其他安全机制可能是可以接受的，但只有当 `NetworkPolicy` 等其他机制可以通过实现有效限制跨命名空间引用时，才可以这样做 [t4-E3]。选择做出此例外的实现必须清楚地记录其实现不遵守 `ReferenceGrant` [t4-E3]。

4. **实验性字段的存在**：API 参考中标注为 Experimental 的字段包括 `useDefaultGateways` 和 `defaultScope` [t7-E5]。这些字段在 Standard 渠道中不可用。

5. **证据缺口**：本报告未找到关于 Gateway API 与 Ingress 在性能基准测试方面的直接对比证据，也未找到关于大规模生产环境中迁移后运维复杂度的量化数据。此外，关于 Gateway API 各实现的一致性测试通过率的具体数据在现有证据中未涉及。

## 结论与建议

### 已证实事实

- Gateway API 自 2023 年 GA 以来，已支持多协议路由、角色分离和灵活策略配置 [t2-E2]。
- Gateway API 具有四种稳定的 API 类别 [t1-E1]。
- Gateway API 在表现力、扩展性、角色分离、通用性、基础设施共享和类型化后端引用等方面均有显著提升，并支持跨命名空间路由绑定 [t2-E2]。
- Gateway API 原生支持流量拆分、URL 重写、Header 路由和跨命名空间路由，而 Ingress 需要依赖 annotation 实现部分功能 [t3-E3]。
- Gateway API 支持 HTTP/HTTPS/gRPC/TCP/UDP 多协议，而 Ingress 仅支持 HTTP/HTTPS [t3-E3]。
- 跨命名空间引用需要在每一道命名空间边界上显式授权 [t4-E2]。
- 许多 Ingress controller 正在积极支持 Gateway API，包括 Istio、Kong、Traefik 等 [t3-E4]。
- Ingress NGINX 尚未计划支持 Gateway API [t3-E4]。
- Kubernetes 开源社区发布了 ingress2gateway 工具，可将 Ingress 批量转换到 Gateway API 资源 [t3-E3]。
- 实验渠道用于试验新功能，实验性资源可与 Standard 渠道资源共存 [t7-E1]。

### 合理推断

- 鉴于 Ingress NGINX 尚未计划支持 Gateway API [t3-E4]，使用 Ingress NGINX 的用户若希望采用 Gateway API，可能需要更换网关实现。
- 鉴于 APISIX Ingress 的 Gateway API 支持处于 Alpha 阶段 [t1-E4]，在生产环境中使用该功能可能需要等待其进一步成熟。
- 证据列出多个标记为 Experimental 的字段，表明存在实验性字段。 [t7-E5]

### 证据缺口

- 现有证据中未包含 Gateway API 与 Ingress 在性能基准测试方面的直接对比数据。
- 现有证据中未包含大规模生产环境中迁移后运维复杂度的量化数据。
- 现有证据中未包含 Gateway API 各实现的一致性测试通过率的具体数据。
- 现有证据中未包含迁移过程中可能出现的具体故障模式及回滚策略的详细说明。

### 生产迁移建议

1. 证据对比多种网关对 Gateway API 的支持情况，涉及评估 Ingress controller 是否支持 Gateway API 的主题。 [t5-E4]若使用 Ingress NGINX，需注意其尚未计划支持 Gateway API [t3-E4]，可能需要更换实现。

2. **使用 ingress2gateway 工具进行批量转换**：Kubernetes 开源社区发布的 ingress2gateway 工具可将集群中的 Ingress 批量转换到 Gateway API 资源 [t3-E3]。

3. **分阶段迁移**：对于仍在 EKS 集群内使用 Ingress-NGINX 的用户，可选择迁移到 Amazon Load Balancer Controller 继续使用 Ingress，或转向使用 Gateway API [t3-E3]。

4. **配置跨命名空间安全**：在迁移过程中，确保正确配置 `allowedRoutes` 和 `ReferenceGrant` 以保障跨命名空间引用的安全性 [t4-E2]。

5. 证据列出多个标记为 Experimental 的字段，表明存在实验性字段。 [t7-E5]

6. **验证安装**：迁移后检查 controller 状态和 CRD 是否安装 [t3-E3]。

## 来源

1. [t1-E1] Gateway API | Kubernetes — https://v1-34.docs.kubernetes.io/zh-cn/docs/concepts/services-networking/gateway
2. [t1-E2] Kubernetes Gateway API 攻略：解锁集群流量服务新维度！— https://juejin.cn/post/7302993899106680832
3. [t1-E3] 列表 - Kubernetes 网关 API — https://gateway-api.kubernetes.ac.cn/implementations
4. [t1-E4] Gateway API 在 APISIX Ingress 的支持和使用 | 支流科技 — https://apiseven.com/blog/how-to-use-gateway-api-in-apisix-ingress-controller
5. [t2-E1] Gateway API 简介 | GKE networking | Google Cloud Documentation — https://docs.cloud.google.com/kubernetes-engine/docs/concepts/gateway-api?hl=zh-cn
6. [t2-E2] Gateway API | Jimmy Song — https://jimmysong.io/zh/book/kubernetes-handbook/service-discovery/gateway
7. [t2-E3] Kubernetes Ingress vs. Gateway API: Key Differences — https://www.plural.sh/blog/kubernetes-ingress-vs-gateway-api
8. [t2-E4] Kubernetes Gateway API vs Ingress — https://imesh.ai/blog/kubernetes-gateway-api-vs-ingress
9. [t3-E3] 告别 Ingress-NGINX：用 Amazon Load Balancer Controller Gateway API 实现更强大的流量管理 — https://aws.amazon.com/cn/blogs/china/ingress-nginx-to-alb-controller-with-gateway-a
10. [t3-E4] 为什么 APISIX Ingress 是比 Ingress NGINX 更好的选择？| Apache APISIX — https://apisix.apache.org/zh/blog/2023/01/11/apisix-ingress-vs-ingress-nginx
11. [t4-E2] 配置跨命名空间引用 | API7 Ingress Controller 与 APISIX Ingress Controller 文档 — https://docs.apiseven.com/ingress-controller/production/cross-namespace
12. [t4-E3] ReferenceGrant - Kubernetes Gateway API — https://itsfinn.cc/post/referencegrant
13. [t4-E5] GEP-709: Cross Namespace References from Routes | Gateway API — https://gateway-api.sigs.k8s.io/geps/gep-709
14. [t5-E3] Kubernetes Gateway API 网关选型全景对比 | Jimmy Song — https://jimmysong.io/zh/blog/kubernetes-gateway-api-comparison
15. [t3-E3] 告别 Ingress-NGINX：用 Amazon Load Balancer Controller Gateway API 实现更强大的流量管理 — https://aws.amazon.com/cn/blogs/china/ingress-nginx-to-alb-controller-with-gateway-api
16. [t7-E1] Gateway API v1.3.0：流量复制、CORS、Gateway 合并和重试预算的改进 | Kubernetes — https://kubernetes.io/zh-cn/blog/2025/06/02/gateway-api-v1-3
17. [t7-E3] Gateway API v1.0: GA Release | Kubernetes — https://kubernetes.io/blog/2023/10/31/gateway-api-ga
18. [t7-E4] Gateway API Support — Cilium 1.21.0-dev documentation — https://docs.cilium.io/en/latest/network/servicemesh/gateway-api/gateway-api
19. [t7-E5] API Reference | Gateway API — https://gateway-api.sigs.k8s.io/reference/api-spec/main/spec

## 证据索引

- `t1-E1` [Gateway API | Kubernetes](https://v1-34.docs.kubernetes.io/zh-cn/docs/concepts/services-networking/gateway)
- `t1-E2` [Kubernetes Gateway API 攻略：解锁集群流量服务新维度！Kubernetes Gateway API - 掘金](https://juejin.cn/post/7302993899106680832)
- `t1-E3` [列表 - Kubernetes 网关 API](https://gateway-api.kubernetes.ac.cn/implementations)
- `t1-E4` [Gateway API 在 APISIX Ingress 的支持和使用 | 支流科技](https://apiseven.com/blog/how-to-use-gateway-api-in-apisix-ingress-controller)
- `t2-E1` [Gateway API 简介  |  GKE networking  |  Google Cloud Documentation](https://docs.cloud.google.com/kubernetes-engine/docs/concepts/gateway-api?hl=zh-cn)
- `t2-E2` [Gateway API | Jimmy Song](https://jimmysong.io/zh/book/kubernetes-handbook/service-discovery/gateway)
- `t2-E3` [Kubernetes Ingress vs. Gateway API: Key Differences](https://www.plural.sh/blog/kubernetes-ingress-vs-gateway-api)
- `t2-E4` [Kubernetes Gateway API vs Ingress](https://imesh.ai/blog/kubernetes-gateway-api-vs-ingress)
- `t2-E5` [The Complete Guide to Kubernetes Ingress and Gateway API: Features, Differences, and Use Cases](https://medium.com/@deepeshjaiswal6734/the-complete-guide-to-kubernetes-ingress-and-gateway-api-features-differences-and-use-cases-5637377acb89)
- `t3-E1` [Gateway API 简介  |  GKE networking  |  Google Cloud Documentation](https://docs.cloud.google.com/kubernetes-engine/docs/concepts/gateway-api?hl=zh-cn)
- `t3-E3` [告别 Ingress-NGINX：用 Amazon Load Balancer Controller Gateway API 实现更强大的流量管理 | 亚马逊AWS官方博客](https://aws.amazon.com/cn/blogs/china/ingress-nginx-to-alb-controller-with-gateway-api)
- `t3-E4` [为什么 APISIX Ingress 是比 Ingress NGINX 更好的选择？ | Apache APISIX](https://apisix.apache.org/zh/blog/2023/01/11/apisix-ingress-vs-ingress-nginx)
- `t3-E5` [Kubernetes Gateway API 深入解读和落地指南](https://zhuanlan.zhihu.com/p/627229130)
- `t4-E1` [Gateway API 简介  |  GKE networking  |  Google Cloud Documentation](https://docs.cloud.google.com/kubernetes-engine/docs/concepts/gateway-api?hl=zh-cn)
- `t4-E2` [配置跨命名空间引用 | API7 Ingress Controller 与 APISIX Ingress Controller 文档](https://docs.apiseven.com/ingress-controller/production/cross-namespace)
- `t4-E3` [ReferenceGrant - Kubernetes Gateway API - itsfinn's blog.](https://itsfinn.cc/post/referencegrant)
- `t4-E5` [GEP-709: Cross Namespace References from Routes | Gateway API](https://gateway-api.sigs.k8s.io/geps/gep-709)
- `t5-E1` [Gateway API 选型：Envoy Gateway / Istio / Traefik 怎么选？](https://mp.weixin.qq.com/s/f6JPAMV2emhCi_razpI8nQ)
- `t5-E3` [Kubernetes Gateway API 网关选型全景对比 | Jimmy Song](https://jimmysong.io/zh/blog/kubernetes-gateway-api-comparison)
- `t5-E4` [Kubernetes Gateway API 网关选型全景对比](https://blog.csdn.net/mzl87/article/details/149431531)
- `t6-E1` [Nginx Ingress 迁移指引-API 网关(API Gateway)-阿里云帮助中心](https://help.aliyun.com/zh/api-gateway/cloud-native-api-gateway/use-cases/nginx-ingress-migration-guide)
- `t6-E2` [将 Ingress 迁移到 Gateway API  |  GKE networking  |  Google Cloud Documentation](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/migrate-ingress-gateway?hl=zh-cn)
- `t6-E3` [將Ingress 遷移至Gateway API | GKE networking](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/migrate-ingress-gateway?hl=zh-tw)
- `t6-E4` [从 Ingress 迁移到 Kubernetes Gateway API](https://gateway-api.kubernetes.ac.cn/guides/migrating-from-ingress)
- `t6-E5` [告别 Ingress-NGINX：用 Amazon Load Balancer Controller Gateway API 实现更强大的流量管理 | 亚马逊AWS官方博客](https://aws.amazon.com/cn/blogs/china/ingress-nginx-to-alb-controller-with-gateway-api)
- `t7-E1` [Gateway API v1.3.0：流量复制、CORS、Gateway 合并和重试预算的改进 | Kubernetes](https://kubernetes.io/zh-cn/blog/2025/06/02/gateway-api-v1-3)
- `t7-E2` [Gateway API 简介  |  GKE networking  |  Google Cloud Documentation](https://docs.cloud.google.com/kubernetes-engine/docs/concepts/gateway-api?hl=zh-cn)
- `t7-E3` [Gateway API v1.0: GA Release | Kubernetes](https://kubernetes.io/blog/2023/10/31/gateway-api-ga)
- `t7-E4` [Gateway API Support — Cilium 1.21.0-dev documentation](https://docs.cilium.io/en/latest/network/servicemesh/gateway-api/gateway-api)
- `t7-E5` [API Reference | Gateway API](https://gateway-api.sigs.k8s.io/reference/api-spec/main/spec)
- `t8-E1` [测试环境一键迁移到生产环境](https://docs.guandata.com/product/bi/6.6.0/429126781333667840)
- `t8-E2` [Google Cloud 的变更方法  |  Get started  |  Google Cloud Documentation](https://docs.cloud.google.com/docs/cloud-approach-to-change?hl=zh-cn)
- `t8-E3` [什么是云迁移？ 策略和流程 | Google Cloud](https://cloud.google.com/learn/cloud-migration?hl=zh-CN)
- `t8-E4` [环境规制、 重污染企业迁移与协同治理效果](https://ccj.pku.edu.cn/Article/DownLoad?id=270057242&type=ArticleFile)
- `t8-E5` [测试右移——生产环境下的QA | BY林子](https://www.bylinzi.com/2016/06/13/qa-in-production)
- `verify_r1_1-E1` [Kubernetes Ingress vs. Gateway API: Key Differences](https://www.plural.sh/blog/kubernetes-ingress-vs-gateway-api)
- `verify_r1_1-E2` [INGRESS VS Gateway API - whats the difference?](https://www.youtube.com/shorts/YiuRzpf2m18)
- `verify_r1_1-E3` [Kubernetes Gateway API vs Ingress](https://imesh.ai/blog/kubernetes-gateway-api-vs-ingress)
- `verify_r1_2-E1` [Announcing Ingress2Gateway 1.0: Your Path to Gateway API | Kubernetes](https://kubernetes.io/blog/2026/03/20/ingress2gateway-1-0-release)
- `verify_r1_2-E2` [A Welcome Guide for Ingress-NGINX Users | Gateway API](https://gateway-api.sigs.k8s.io/guides/getting-started/migrating-from-ingress-nginx)
- `verify_r1_2-E3` [Ingress NGINX is EOL: A practical guide for migrating to Kubernetes Gateway API | Datadog](https://www.datadoghq.com/blog/migrate-to-gateway-api)
- `verify_r1_3-E1` [GitHub - nginx/nginx-gateway-fabric: NGINX Gateway Fabric provides an implementation for the Gateway API using NGINX as the data plane. · GitHub](https://github.com/nginx/nginx-gateway-fabric)
- `verify_r1_3-E2` [The NGINX Kubernetes Open Source Roadmap: First Half of 2026 – NGINX Community Blog](https://blog.nginx.org/blog/the-nginx-kubernetes-open-source-roadmap-first-half-of-2026)
- `verify_r1_3-E3` [Migrating for k8s nginx ingress to f5 nginx ingress (ingress api) - no response - NGINX Ingress Controller - NGINX Community Forum](https://community.nginx.org/t/migrating-for-k8s-nginx-ingress-to-f5-nginx-ingress-ingress-api-no-response/8694)
- `verify_r1_4-E1` [feat: As a Kubernetes user, I want to leverage Gateway API ...](https://github.com/apache/apisix-ingress-controller/issues/2498)
- `verify_r1_4-E2` [APISIX Ingress 对 Gateway API 的支持和应用 | Apache APISIX](https://apisix.apache.org/zh/blog/2022/12/27/apisix-ingress-with-gatewayapi)
- `verify_r1_4-E3` [Gateway API 在 APISIX Ingress 的支持和使用 | 支流科技](https://www.apiseven.com/blog/how-to-use-gateway-api-in-apisix-ingress-controller)
