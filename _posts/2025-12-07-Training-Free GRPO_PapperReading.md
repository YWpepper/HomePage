---
layout: post
title: 'Training-Free GRPO_PapperReading'
date: 2025-12-07
author: pepper
tags: [papperReading, Note]
comments: true
toc: true
pinned: false
---

这篇博客介绍了腾讯Training-Free GRPO论文的阅读笔记。

<!-- more -->

## 摘要

大型语言模型 (Large Language Model, LLM) 智能体在近期取得了进展，展现了其有前景的通用能力。然而，由于难以有效集成外部工具和特定的提示策略，它们在专业化的现实世界领域中的性能通常会下降。虽然已经提出了诸如智能体强化学习 (agentic reinforcement learning) 等方法来解决这个问题，但*它们通常依赖成本高昂的参数更新*，例如通过一个使用监督微调 (Supervised Fine-Tuning, SFT) 随后进行强化学习 (Reinforcement Learning, RL) 阶段（采用群体相对策略优化 (Group Relative Policy Optimization,**GRPO**)）来改变输出分布的过程。

然而，我们认为 LLMs 可以通过将经验知识 (experiential knowledge) 作为标记先验 (**token prior**) 来学习，从而对输出分布产生类似的效果，这是一种轻量得多 (far more lightweight) 的方法，它不仅解决了实际*数据稀疏性* (practical data scarcity) 的问题，而且避免了常见的*过拟合* (overfitting) 问题。

为此，我们提出了*免训练群体相对策略优化* (Training-Free Group Relative Policy Optimization, Training-Free GRPO)，这是一种*成本效益高的解决方案*，可以在不进行任何参数更新的情况下增强 LLM 智能体的性能。我们的方法利用每组推演 (rollouts) 内的群体相对语义优势而非数值优势，在**最小量的真实数据** (minimal ground-truth data) 上进行多轮 (multi-epoch) 学习过程中，迭代地提炼 (iteratively distilling) 高质量的经验知识。此类知识作为习得的标记先验，在 LLM API 调用期间被无缝集成以指导模型的行为。

在数学推理 (mathematical reasoning) 和网络搜索 (web searching) 任务上的实验表明，Training-Free GRPO 应用于 DeepSeek-V3.1-Terminus 时，显著改善了域外性能 (out-of-domain performance)。仅凭几十个训练样本 (few dozen training samples)，Training-Free GRPO 的性能就超越了具有少量训练数据和成本的微调小型 LLMs (fine-tuned small LLMs)。

## Introduction

大型语言模型 (Large Language Models, LLMs) 正在成为强大的通用智能体 (general-purpose agents)，能够与复杂的现实世界环境进行交互。它们在一系列广泛的任务中展现出卓越的能力，包括复杂问题解决 [4, 5, 6]、高级网络研究 [7, 8, 9, 10]、代码生成与调试 [11, 12]，以及熟练的计算机使用 [13, 14, 15]。

尽管它们的能力令人印象深刻，但 LLM 智能体在专业化的现实世界领域中往往表现不佳。这些场景通常要求集成外部工具 （例如，计算器、API、数据库），以及领域特定的任务定义和提示策略 (prompting strategies)。在这样的设置中，开箱即用 (out-of-the-box) 地部署通用智能体，通常由于对领域特定要求的不熟悉或对必要工具的接触不足而导致次优性能 (suboptimal performance)。


### Challenges in Agentic Training

为了弥合这一差距，智能体训练 (agentic training) 已成为促进 LLM 智能体适应特定领域及其相关工具的一种有前景的策略 [4, 7, 8, 16]。最近，智能体强化学习 (Agentic Reinforcement Learning, Agentic RL) 方法的进展采用了 **群体相对策略优化** (Group Relative Policy Optimization, GRPO) [17] 及其变体 [18, 19, 20] 来在参数空间 (parameter space) 中对齐模型行为。尽管这些方法有效地增强了任务特定能力，但它们依赖于调整 LLM 参数，带来了若干实际挑战：

* 计算成本 (Computational Cost)： 即使对于较小的模型，微调也需要大量的计算资源，这使其既昂贵又对环境不可持续。对于更大的模型，成本变得令人望而却步 (prohibitive)。此外，微调后的模型需要**专用部署** (dedicated deployment)，并且通常局限于特定应用，相对于更通用的模型而言，对于低频用例 (low-frequency use cases) 效率低下。
* 泛化能力差 (Poor Generalization)： 通过参数调整优化的模型通常会遭受不尽如人意的跨域泛化 (unsatisfactory cross-domain generalization)，限制了它们的适用范围仅限于狭窄的任务。因此，必须部署多个专业化模型 (multiple specialized models) 来处理一套全面的任务，这显著增加了系统复杂性 (system complexity) 和维护开销 (maintenance overhead)。
* 数据稀缺性 (Data Scarcity)： 微调 LLMs 通常需要大量高质量、精心标注的数据，而这些数据在专业领域往往稀缺 且获取成本极高 (prohibitively expensive)。此外，样本有限时，模型极易受到过拟合 (overfitting) 的影响，导致泛化能力差。
* 回报递减 (Diminishing Returns)： 令人望而却步的训练成本通常迫使现有方法微调参数少于 320 亿的较小 LLMs，这是由于资源限制而非最优设计选择。虽然更大型的模型会更受青睐，但微调的计算开销迫使了这种妥协。矛盾的是，基于 API 或开源的更大型 LLMs 通常通过可扩展性和持续的模型更新提供更好的成本-性能比 (cost-performance ratios)。然而，这些通用模型在需要微调的专业领域表现不佳，从而产生了成本-性能困境 (cost-performance dilemma)。

### Proposed Solution

参数调整固有的这些限制促使了一个基础性的研究问题：应用 RL 在参数空间中是唯一可行的途径吗？我们能否以非参数方式，用更低的数据和计算成本来增强 LLM 智能体的性能？

我们通过提出**免训练群体相对策略优化** (Training-Free Group Relative Policy Optimization, Training-Free GRPO)，肯定地回答了这个问题。这是一种新颖且高效 的方法，它以类似于原始 GRPO 的方式改进 LLM 智能体的行为，同时保持原始模型参数不变 (preserving the original model parameters unchanged)。

我们的方法源于一个洞察：LLMs 已经拥有适应新场景的基本能力，**只需通过有限样本进行最小量的实践 即可达到强大的性能**。因此，与其通过参数调整来调整它们的输出分布，不如利用**轻量级标记先验 (lightweight token prior)** 的上下文学习 (in-context learning) 也能封装从最小训练数据集中学到的经验知识。

Taining-Free GRPO 保留了原始 GRPO (vanilla GRPO) 的**多轮学习机制** (multi-epoch learning mechanism)。在每一轮中，**系统会为每个查询生成多个输出**，以提供一组推演 (group of **rollouts**)，这有助于探索**策略空间** (explore the **policy** space) 和**评估潜在策略** (evaluate **potential** **strategies**)。

然而，原始 GRPO 依赖基于**梯度的参数更新** (gradient-based parameter updates) 来迭代改进策略性能，而 Training-Free GRPO 通过使用 LLMs（大型语言模型）的仅推理操作 (inference-only operations) 消除了这一要求。在每个优化步骤中，我们的方法不是为每组推演中的梯度上升 (gradient ascent) 计算数值优势 (numerical advantage)，而是利用 LLMs 对每组进行内省 (introspect) 并提炼出语义优势 (semantic advantage)。

这种优势精炼了外部经验知识 (experiential knowledge)，并基于不断演变的上下文先验 (contextual priors) 来指导策略输出，从而在不修改任何模型参数的情况下实现了策略优化效果。

通过评估富有挑战性的数学推理和交互式网络搜索任务，我们证明了该方法能够显著**增强冻结策略模型** (frozen policy models)，例如 DeepSeek-V3.1-Terminus [3] 的性能，**仅需数十个训练样**本。它在性能上超越了**经过微调**的 32B 模型，而所需的计算资源仅占其一小部分，为传统微调技术提供了一种更简单、效率更高的替代方案。

我们的主要贡献有以下三方面：

* 一种新的免训练 RL 范式 (A New **Training-Free** RL Paradigm)：我们引入了 Training-Free GRPO，它通过利用不断演变的经验知识作为标记先验 (token priors)，将**策略优化**从**参数空间**转移到**上下文空间** (context space)，无需梯度更新。
* 语义群体优势 (Semantic Group Advantage)：我们用语义群体优势取代了原始 GRPO 中的数值群体优势，使 LLMs 能够内省自身的推演，并在多个优化步骤中持续更新经验知识。
* 数据和计算效率 (Data and Computational Efficiency)：实验证实，Training-Free GRPO 能用最少的训练样本有效提升冻结策略的性能，为不同领域提供了一种实用且成本效益高的替代方案。
* 卓越的泛化能力 (Superior Generalization)：通过保持模型参数冻结并插入不同的标记先验，我们的方法完全保留了泛化能力，消除了部署多个微调专家模型的成本和复杂性。


<img src="https://images.weserv.nl/?url=cdn.nlark.com/yuque/0/2025/png/40742019/1765115609308-0106c8a6-68e6-47ff-9065-504fecf54408.png?x-oss-process=image%2Fformat%2Cwebp" width="70%"/>


## 2 Training-Free GRPO

本节介绍我们的免训练 GRPO (Training-Free GRPO)，该方法旨在复制 GRPO 算法的对齐效益，而无需对策略模型的参数执行任何基于梯度的更新。

### 原始 GRPO Vanilla GRPO

原始 GRPO 过程首先使用当前策略 LLM $\pi_{\theta}$ 为给定查询 $q$ 生成一组 $G$ 个输出 $\{o_1, o_2, \ldots, o_G\}$，即 $\pi_{\theta}(o_i | q)$。然后，每个输出 $o_i$ 通过一个奖励模型 $R$ 进行独立评分。随后，利用奖励 $r = \{r_1, \ldots, r_G\}$，它为每个输出 $o_i$ 计算一个群体相对优势 (group-relative advantage) $\hat{A}_i = \frac{r_i - \text{mean}(r)}{\text{std}(r)}$。通过结合针对参考模型 (reference model) 的 KL 散度惩罚 (KL-divergence penalty)，它构建了一个 PPO 裁剪目标函数 (PPO-clipped objective function) $J_{\text{GRPO}}(\theta)$，然后通过最大化该函数来更新 LLM 参数 $\theta$。

### Training-Free GRPO 的核心逻辑

Training-Free GRPO 重新利用了这种基于群体的相对评估的核心逻辑，但将其转化为非参数化 (non-parametric) 的推理时过程 (inference-time process)。我们永久冻结参数 $\theta$，并维护一个外部经验知识 (external experiential knowledge) $E$，其初始化为 $\emptyset$，而不是更新参数 $\theta$。

#### 推演与奖励 Rollout and Reward

我们的推演和奖励过程与 GRPO 完全一致。给定一个查询 $q$，我们执行一个并行推演 (parallel rollout)，使用 LLM 生成一组 $G$ 个输出 $\{o_1, o_2, \ldots, o_G\}$。值得注意的是，虽然 GRPO 使用当前的可训练策略 $\pi_{\theta}$，但我们的策略以经验知识 $E$ 为条件，即 $\pi_{\theta}(o_i|q, E)$。与标准 GRPO 设置相同，我们通过奖励模型 $R$ 对每个输出 $o_i$ 进行评分，以获得一个标量奖励 (scalar reward) $r_i = R(q, o_i)$。

#### 群体优势计算 (Group Advantage Computation)

为了为策略参数提供优化方向，原始 GRPO 计算一个数值优势 $\hat{A}_i$，用于**量化每个输出 $o_i$ 在其群体内的相对质量**。类似地，Training-Free GRPO 在每个群体内执行类似的比较，但会以自然语言经验的形式产生群体相对语义优势 (group relative semantic advantage)，如图 3 所示 。

📌 由于在原始 GRPO 中，当所有 $G$ 个输出获得相同奖励（即 $\text{std}(r) = 0$）时，$\hat{A}_i = 0$，因此我们仅对存在明确赢家和输家的群体生成这种语义优势。**具体来说**，对于每个输出 $o_i$，我们首先询问同一个 LLM $M$ 分别提供一个相应的总结 $s_i = M(p_{\text{summary}}, q, o_i)$，其中 $p_{\text{summary}}$ 是一个提示模板，它结合了查询 $q$ 和输出 $o_i$ 来形成一个结构化的总结请求。给定总结 $\{s_1, s_2, \ldots, s_G\}$ 和当前的经验知识 $E$，LLM $M$ 阐明了输出相对成功或失败的原因，然后提取一个简洁的自然语言经验 $A_{\text{text}} = M(p_{\text{extract}}, q, s_i, E)$，其中 $p_{\text{extract}}$ 是另一个用于经验提取的提示模板。

这种自然语言经验 $A_{\text{text}}$ 作为我们的语义优势，在功能上等同于原始 GRPO 的 $\hat{A}_i$，它编码了什么行动导致高奖励的关键经验知识。

#### Optimization
原始 GRPO（vanilla GRPO）通过在单个批次中计算得到的所有优势 (all advantages) 对 $J_{\text{GRPO}}(\theta)$ 进行梯度上升 (gradient ascent) 来更新其模型参数 $\theta$，**📌 而我们则使用当前批次中的所有语义优势 $A_{\text{text}}$ 来更新我们的经验库 (experience library) $E$**。具体来说，给定现有的经验库 $E$，我们提示 LLM 根据所有这些 $A_{\text{text}}$ 生成一个操作列表 (list of operations)，其中每个操作可以是：
    - 添加 (**Add**)： 将 $A_{\text{text}}$ 中描述的经验直接附加到经验库 $E$ 中。
    - 删除 (**Delete**)： 基于 $A_{\text{text}}$，从经验库 $E$ 中移除一条低质量经验。
    - 修改 (**Modify**)： 基于 $A_{\text{text}}$ 中的见解，对经验库 $E$ 中现有的经验进行精炼或改进。
    - 保留 (**Keep**)： 经验库 $E$ 保持不变。



---

