---
title: CS336 Assignment5-alignment
date: 2026-11-11
description: CS336 Assignment5 学习笔记
---

# Introduction
## Background
在预训练阶段，模型通过学习大量互联网文本获得广泛的知识和语言能力，但这并不意味着它能够准确地完成特定任务。因此，我们需要通过后训练将基础模型（Base Model）转化为更有帮助、更安全且能够遵循指令的对话模型。其中，指令微调（Instruction Tuning）和对齐（Alignment）主要用于引导模型形成符合要求的行为，而强化学习（Reinforcement Learning）则侧重于提高模型在特定任务上的表现。与预训练使用大量文本并预测下一个 token 不同，强化学习通常使用较少的任务数据，并通过评分函数（Scoring Function）评估模型回答的质量，而不一定需要提供可供模仿的标准答案文本。例如，在编程任务中，可以通过运行测试用例判断代码是否正确；在数学任务中，可以检查模型给出的最终答案是否正确。其基本流程是先从模型中采样回答，再利用评分函数计算奖励（Reward），最后根据奖励信号更新模型参数，使模型更倾向于生成能够获得高奖励的回答。从数学角度来看，强化学习的目标是最大化模型生成回答的期望奖励，从而提高特定任务的准确率。然而，强化学习也面临一些挑战，包括训练速度较慢、训练过程不稳定、不同随机种子可能导致实验结果存在较大差异，以及一些看似细微的实现细节可能显著影响最终表现。因此，这项作业旨在帮助我们从数学原理和实际实验两个角度理解大语言模型强化学习的工作机制及其面临的挑战。

## Model and dataset
本次作业使用 **OLMo-2-0425-1B** 作为基础模型，该模型拥有约 10 亿个参数，经过预训练和中期训练，总共学习了约 4 万亿个 token。其预训练数据 OLMo-mix-1124 主要由 DCLM-Baseline 构成，中期训练数据 Dolmino-mix-1124 则包含约 50% 的 DCLM 数据，以及指令跟随、数学、代码、STEM 论文和维基百科等数据。

本次作业选择 **GSM8K 数据集** 作为下游任务。GSM8K 是一个包含小学数学应用题的数据集，提供问题、推理过程和最终答案。在强化学习（RL）过程中，模型会生成数学解答，并通过奖励函数评估答案的正确性，再根据奖励信号更新模型参数，使模型逐步学会生成更准确的推理过程，提高数学解题能力。
```
{  "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?", "answer": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72" }
```

选择这一模型与数据集组合的主要原因是计算资源有限。OLMo-2-0425-1B 虽然参数规模较小，但经过了大量 token 的训练，具备一定的数学推理能力，因此适合在有限资源下观察强化学习的效果。**强化学习的效果高度依赖模型能力、数据集难度和训练设置**，因此，本次实验的结果不一定适用于其他模型和任务。

## Notation
符号表
![](\images\CS336_Assignment5\img1.png)

![](\images\CS336_Assignment5\img2.png)

# Prompting
将预训练的 base model 应用于下游任务的第一步是提供 prompt，base models 实在多种行为模式下预训练的，prompting 是一种轻量级的方法，可以引导其行为至 task-solving models。

最基础的 prompt 方法是提供问题，然后从模型的下一个 token 分布中采样生成答案，这种策略叫做 **question_only**。我们将对比这种方法和 **r1_zero** 策略，r1_zero 策略既包含问题也包含指导模型进行“思维链”推理的指令。

## Using vLLM for inference
要从模型中生成响应，我们需要一个推理引擎。实现推理引擎超出了本次作业的范围，因此我们将使用 vLLM 推理引擎，该引擎实现了多种优化，包括快速的 CUDA 内核、用于高效 KV caching 的 PagedAttention 等。cs336_alignment/vllm_utils.py 文件中已提供了启动 vLLM 服务器并生成文本的代码，其接口如下：

![](\images\CS336_Assignment5\img3.png)

## Zero-shot, few-shot, and chain-of-thought prompting
除非另有说明，在GSM8K实验中，我们将使用DeepSeek R1-Zero模型中的以下提示词。我们将其称为r1_zero提示词：

![](\images\CS336_Assignment5\img4.png)

r1_zero 提示词位于文件 cs336_alignment/prompts/r1_zero.prompt

在提示中，“问题”指的是我们插入的某个问题（例如：娜塔莉亚在4月份向48位朋友卖了发夹，然后她在5月份卖的发夹数量是4月份的一半。娜塔莉亚在四月和五月总共卖出了多少个发夹？）我们期望模型扮演助手的角色，在已包含的起始标签 `<think>` 之后开始生成思考过程，用 `</think>` 结束思考过程，然后在答案标签内生成最终的符号答案，例如 `<answer>` 4x + 10 `</answer>`。让模型生成`<answer>` `</answer>`这类标签的目的是，以便我们能够轻松解析模型的输出并将其与真实答案进行对比，同时在检测到闭合的`<answer>`标签`</answer>`时停止响应生成。

另一种提示方法称为“少量样本提示”（few-shot prompting），其做法是在向模型提出实际问题之前，先在提示语前添加几个问题-答案对。r1_zero 提示的少量样本版本如下所示（注意前面的三个 user-assistant 对是有具体的例子）：

![](\images\CS336_Assignment5\img5.png)

Few-shot prompting 通过向模型提供少量待解决任务的示例，从而提升模型的性能。已公布的开放式基础模型基准测试数据通常采用 few-shot 的方法。

最后，作为基准，我们将纳入“question_only” prompt，该提示词可在 cs336_alignment/prompts/ question_only.prompt 中找到（请注意，虽然提示词名为 question_only，但我们要求将最终答案用方括号括起来，以便评分者能更轻松地从响应中解析出答案，具体原因将在下一节中讨论）：

![](\images\CS336_Assignment5\img6.png)

## Grading function
模型生成响应后，我们需要检查其正确性。我们的数学题包含标准答案（例如 0.5），但模型可以通过多种方式给出正确答案：例如，它可以回答 $<answer> 1/2 </answer>$ 或 “The answer is 0.5”。因此，为了正确评分模型的输出结果，我们需要一个答案解析函数：该函数以模型的输出和已知的真实答案作为输入，并返回一个布尔值，用于指示模型的输出是否正确。

实验采用近期推理强化学习研究中使用的一种快速且准确率较高的答案解析器，根据不同的提示词格式选择不同的奖励函数：如果使用 r1_zero prompt，也就是要求模型使用 `<think>` 和 `<answer>` 标签组织推理过程和最终答案，就使用 cs336_alignment.drgrpo_grader.r1_zero_reward_fn；如果使用 question_only prompt，由于它不要求模型使用这些标签，因此应使用 cs336_alignment.drgrpo_grader.question_only_reward_fn，该函数会寻找最终答案中的 \boxed{} 标记。例如，模型可以将最终答案写成 \boxed{12}，供解析器提取并判断是否正确。这些奖励函数会返回总奖励（Total Reward）、格式奖励（Format Reward）和答案奖励（Answer Reward）：格式奖励用于判断回答是否符合预期格式，答案奖励用于判断解析出的答案是否正确。有些研究会给格式正确但答案错误的回答一定的部分奖励（Partial Credit），但本次作业不采用这种方式，总奖励只由答案正确性决定，格式奖励仅用于日志记录和分析，不参与奖励优化。因此，即使模型严格遵守格式，只要最终答案错误，就不会因为格式正确而获得额外奖励。

请注意，这些奖励函数要求 ground_truth 参数仅包含答案。要从 GSM8K 数据集中的 ground truth 响应（其格式为 {rationale} #### {answer}）中提取答案，应以 #### 为分隔符进行拆分，并去除空格。
## Experiments
Generation hyperparameters：当生成回复时，使用 temperature 1.0，top-p 1.0，max generation length 512 进行采样。r1_zero 提示词会要求模型在回答结尾添加字符串 `</answer>`，因此在使用这些提示词时，我们可以指示 vLLM 在模型输出该字符串时停止

![](\images\CS336_Assignment5\img7.png)

![](\images\CS336_Assignment5\img8.png)

==缺少实验代码，之后补上==
# Group Relative Policy Optimization
既然我们已经通过 prompt 评估了模型的性能，下一步就是通过训练来提升这一性能。具体来说，我们要优化模型的准确率，或者说，优化其预期奖励：

![](\images\CS336_Assignment5\img9.png)

![](\images\CS336_Assignment5\img10.png)

## Deriving on-policy GRPO
我们将首先探究 RL 背后的数学原理，逐步引出 GRPO (Group Relative Policy Optimization) 算法。
### Language models as policies

首先介绍一下传统的强化学习 RL，强化学习的核心目标是：通过与环境交互，学习一个能够获得更高累积奖励的策略（Policy）。
在传统强化学习中，智能体（Agent）会不断观察当前状态、选择动作、获得奖励，并进入下一个状态。这个过程可以表示为：

$$
s_t \xrightarrow{\pi_{\theta}} a_t \xrightarrow{\text{环境}} (r_t,s_{t+1}) 
$$

其中：

- 状态（State）$s_t$：描述智能体在时间步 $t$ 所处的环境情况。
- 策略（Policy）$\pi_{\theta}$：根据当前状态决定采取什么动作。下标 $\theta$ 表示策略的参数。
- 动作（Action）$a_t$：智能体在当前状态下采取的行为。
- 奖励（Reward）$r_t$：用于评价智能体行为的反馈信号。
- 状态转移（State Transition）：智能体采取动作后，环境根据状态转移规律产生下一个状态 $s_{t+1}$。
强化学习通过不断优化策略参数 $\theta$，使智能体在与环境交互的过程中获得更高的累积奖励。

**策略**描述了智能体在给定状态下如何选择动作，主要分为以下两类。
（1）随机策略（Stochastic Policy）
随机策略输出的是一个动作概率分布，而不是固定的动作。智能体根据该概率分布进行采样，得到实际执行的动作。

$$
a_t \sim \pi_{\theta}(\cdot \mid s_t) \\
$$
例如，在某个状态下，策略输出：

| 动作   | 概率  |
| ---- | --- |
| 向左移动 | 0.2 |
| 向右移动 | 0.7 |
| 保持不动 | 0.1 |

智能体按照上述概率分布采样，最终可能选择向左、向右或保持不动，其中向右的概率最高，但并不意味着每次都会选择向右。
（2）确定性策略（Deterministic Policy）
确定性策略直接根据当前状态输出一个确定的动作：

$$
a_t=\mu_{\theta}(s_t) \\
$$
在相同状态和相同策略参数下，它总是输出相同的动作。

**奖励函数**用于评价智能体采取动作后的结果。为了便于理解，通常可以写成：

$$
r_t=r(s_t,a_t) \\
$$

这表示奖励由当前状态和当前动作决定。

但这是一种简化写法。在实际任务中，奖励也可能依赖于下一个状态：

$$
r_t=r(s_t,a_t,s_{t+1}) \\
$$
例如，机器人从当前位置移动到下一个位置后，奖励可能取决于是否成功到达目标。此外，有些任务的奖励需要在整个轨迹结束后才能计算。例如，棋类游戏可能只有在对局结束时才获得胜负奖励。
环境则根据状态转移分布产生下一个状态：
$$
s_{t+1}\sim P(\cdot\mid s_t,a_t) \\
$$
因此，智能体与环境的交互可以概括为：

1. 观察当前状态 $s_t$。
2. 根据策略 $\pi_{\theta}$ 选择动作 $a_t$。
3. 环境返回奖励 $r_t$，并产生下一个状态 $s_{t+1}$。
4. 重复上述过程，并根据获得的奖励优化策略。

**强化学习通常关注的是累积奖励。**

因果语言模型根据已经生成的文本，预测下一个 Token 的概率分布，并从中选择或采样一个 Token。例如，给定提示词：

> 今天天气很

语言模型可能预测出以下下一个 Token 的概率分布：

| 下一个 Token | 概率   |
| --------- | ---- |
| 好         | 0.60 |
| 冷         | 0.25 |
| 热         | 0.10 |
| 差         | 0.05 |

模型可以根据该分布采样一个 Token，例如选择“好”，然后将其追加到已有文本中，继续预测下一个 Token。这个过程与强化学习中的决策过程十分相似：

![](\images\CS336_Assignment5\img11.png)

因此，可以将语言模型的生成过程理解为一个连续决策过程：

$$
s_t=(x,y_{<t}) 
$$

$$
a_t=y_t\sim\pi_\theta(\cdot\mid x,y_{<t}) 
$$
每生成一个 Token，模型就完成了一次决策，并将该 Token 纳入后续决策的上下文中。语言模型的下一个 Token 预测，本质上就是根据当前上下文定义一个动作概率分布，再从中选择或采样一个 Token。 这使得强化学习中的策略优化方法可以应用于语言模型。

当我们使用策略梯度（Policy Gradient）方法优化语言模型时，需要反复与当前策略交互，并根据奖励调整模型参数。其中两个基本操作是：
1. 从策略中采样：从上述分类分布中抽取一个动作 $a_t$
2. 计算动作的对数似然值：计算 $log \pi_{\theta}(a_t | s_t)$

为什么不直接使用概率，而要使用对数概率？
- 数学推导更方便。 策略梯度通常涉及对概率求导，而对数概率能够将相关表达式简化。
- 序列概率更容易计算。 一个回答包含很多 Token，整个序列的概率是每一步条件概率的乘积，对数可以将乘积转换成求和。
- 数值计算更稳定。 很长的 Token 序列，其联合概率可能非常小；使用对数通常更容易处理。

### Trajectories
语言模型每生成一个 Token，就相当于智能体执行一次动作；从输入 Prompt 开始，到生成结束为止，所有状态和动作连起来，就构成了一条轨迹。
在强化学习中，智能体从一个初始状态开始，根据策略 $\pi_{\theta}$ 选择动作，然后环境根据状态转移规律产生下一个状态 $s_{t+1} \sim next\_state(s_t, a_t)$ 。这个过程不断重复，直到任务结束。
整个交互过程可以表示为：
$$
s_0 \rightarrow a_0 \rightarrow s_1
\rightarrow a_1 \rightarrow s_2
\rightarrow \cdots
$$
其中：
- $s_t$：时间步 $t$ 的状态。
- $a_t$：智能体在状态 $s_t$ 下采取的动作。
- $s_{t+1}$：执行动作 $a_t$ 后产生的下一个状态。
把整个过程中的状态和动作按时间顺序记录下来，就得到一条轨迹：
$$
\tau=(s_0,a_0,s_1,a_1,\ldots,s_T,a_T)
$$
这里的 $\tau$ 就是 Trajectory，也常被称为 episode 或 rollout。其中的初始状态一般从初始状态分布中采样得到，$s_0 \sim \rho$ 。

在大语言模型的强化学习中，初始状态通常就是输入的 Prompt：$s_0=x$，
这里的 $x$ 是用户提供的提示词，可能包含数学问题、CoT（思维链）提示，或者 Few-shot（少样本示例）。

例如，输入："请计算：$12\times 13$，并解释计算过程。"，这段 Prompt 就构成了模型生成回答时的初始状态 $s_0$。更严格地说，Prompt 通常从某个提示词数据分布中采样。

语言模型中的状态转移可以简洁地写成：$s_{t+1}=(s_t,a_t)$。这里的 $(s_t,a_t)$ 表示将动作 $a_t$ 追加到当前状态 $s_t$ 中，如果把文本拼接操作记为 $\Vert$，则写得更明确一些：$s_{t+1}=s_t\Vert a_t$。

![](\images\CS336_Assignment5\img12.png)

### Rewards and return

在强化学习中，奖励函数用于评价智能体在某个状态下采取的动作所产生的结果。通常写作：$r_t=r(s_t,a_t)$，表示当前动作获得的即时奖励。
#### 数学问题中的奖励设计

假设我们让模型回答一道数学题：

> 计算 $12\times13$。

模型可能生成如下内容：

> 首先，$12\times13=12\times(10+3)$。 展开后得到 $120+36$。 所以答案是 156。

在这个任务中，我们可以采用结果奖励（Outcome Reward）：不对中间推理步骤单独评分，而是在模型生成最终答案后，根据答案是否正确给出奖励。

例如：

| 生成内容                           | 奖励  |
| ------------------------------ | --- |
| 首先，$12\times13=12\times(10+3)$ | 0   |
| 展开后得到 $120+36$                 | 0   |
| 所以答案是 156                      | 1   |
| 如果最终答案是 166                    | 0   |

这里的 0 表示该时间步没有获得奖励，不一定意味着模型的中间推理是错误的。
假设一条完整轨迹为： $\tau=(s_0,a_0,s_1,a_1,\ldots,s_T,a_T)$ 。其中，$a_T$ 是生成最终答案或结束标记的终止动作。那么奖励可以定义为：

$r_T=r(s_T,a_T)= \begin{cases} 1,& \text{最终答案正确}\\\ 0,& \text{最终答案错误} \end{cases}$

对于中间步骤：$r_t=0, t<T$。

因此，这是一种典型的稀疏奖励（Sparse Reward）设置：大部分时间步没有奖励，只有在任务结束时才获得反馈。需要注意，奖励函数并不一定要这样设计。这里的 0/1 奖励是数学问题中一种常见的可验证奖励设计；其他任务也可以使用连续奖励、过程奖励，或者其他奖励形式。

奖励 $r_t$ 衡量的是某一个时间步的即时反馈，而**回报（Return）** 衡量的是一条轨迹中累计获得的奖励。

一般来说，回报可以写成折扣奖励之和：

$R(\tau)=\sum_{t=0}^{T}\gamma^t r_t$

其中，$\gamma$ 是折扣因子，用来控制未来奖励的相对权重。但是，在当前讨论的数学问题任务中，只有最终答案会获得奖励：$r_0=r_1=\cdots=r_{T-1}=0$。所以，整条轨迹的回报就简化为：$R(\tau)=r_T$。也就是说：

$R(\tau)= \begin{cases} 1,& \text{最终答案正确}\\\ 0,& \text{最终答案错误} \end{cases}$

因为中间步骤的奖励全为 0，所以无论采用什么折扣因子，之前的零奖励都不会改变最终回报。

奖励是在轨迹结束时计算的，但我们希望利用这个奖励来优化整个生成策略，而不仅仅是最后一个 Token。

明确了奖励和回报之后，就可以定义强化学习的优化目标。我们希望模型生成的回答尽可能正确，因此目标不是只让某一次回答正确，而是让模型在反复生成时，获得高回报的概率更大。这可以表示为最大化期望回报：

$J_{\theta} = \mathbb{E}_{\tau\sim\pi_{\theta}}[R(\tau)]$

其中：
- $\theta$：语言模型的参数。
- $\pi_\theta$：由参数 $\theta$ 决定的生成策略。
- $\tau$：一次完整的生成轨迹。
- $R(\tau)$：该轨迹获得的总回报。
- $\mathbb{E}$：对可能生成的轨迹求期望。

既然目标是最大化期望回报，那么我们就需要寻找一组参数 $\theta$，使得目标函数 $J(\theta)$ 尽可能大。

 $\theta^*=\arg\max_\theta J(\theta)$

其中：

- $\theta^*$：最优参数。
- $\arg\max_\theta$：寻找能够使目标函数取得最大值的参数。
- $J(\theta)$：当前策略的期望回报。

这就是当前数学问题强化学习任务的优化目标。

注意：这个公式定义的是“要优化什么”，但还没有说明“具体如何优化”。 后续的策略梯度、PPO 或 GRPO 等方法，才会进一步解决如何根据采样结果和奖励信号来更新参数 $\theta$。

最关键的一点是：虽然奖励只在最终答案处产生，但优化目标是整条轨迹的期望回报，而生成轨迹的概率又由模型在每个时间步的 Token 概率共同决定。 因此，即使中间 Token 没有直接奖励，它们仍然可能通过策略梯度参与整个模型的优化。

### Policy gradients
前面我们已经定义了奖励、轨迹和期望回报，那么如何利用这个奖励信号，更新模型参数，让模型以后更有可能生成正确的回答？核心方法是策略梯度（Policy Gradient），这里介绍的是其中最基础的算法 REINFORCE。后续的 GRPO 就是在这一基础上进一步改进得到的。

REINFORCE 的核心逻辑是：
1. 从数据集中采样数学问题。
2. 使用当前模型生成回答。
3. 根据回答是否正确获得奖励。
4. 计算回答的对数概率及其梯度。
5. 根据奖励调整模型参数，提高高奖励回答的生成概率。

强化学习的目标是最大化模型的期望回报， $J_{\theta} = \mathbb{E}_{\tau\sim\pi_{\theta}}[R(\tau)]$，下面我们将沿用语言模型中的符号 $(x, y)$ 来表示提示和响应，而非强化学习中的状态 $s_t$ 和动作 $a_t$。因此，期望回报公式可表示为：
$$J_{\theta} = \mathbb{E}_{x \sim \rho}\mathbb{E}_{y \sim \pi_\theta(\cdot | x)} [r(y|x)]$$
其中 $r(y | x)$ 表示回答的奖励，由于这里采用 0/1 奖励，所以可理解为对于提示 $x$，响应 $y$ 是否正确。
为了找到一组函数，最大化目标函数， $\theta^*=\arg\max_\theta J(\theta)$，采用梯度上升的方法：
$$
\theta_{k+1}=\theta_k + \alpha \nabla_\theta J(\theta_k)
$$
最常用的方法，因为梯度指向目标函数局部增长最快的方向，所以沿梯度方向更新参数，通常可以最大化目标函数。
**REINFORCE** 策略梯度用于求 $\nabla_\theta J(\theta_k)$：
$$
\nabla_\theta J(\theta)=\mathbb{E}_{x \sim \rho}\mathbb{E}_{y \sim \pi_\theta(y | x)}[r(y|x)\nabla_\theta log \pi_\theta (y|x)]
$$
该公式中被期望项是 $r(y|x)\nabla_\theta log \pi_\theta (y|x)$，由两部分组成：
1. 奖励 $r(y|x)$：用于衡量当前回答的表现
2. 对数概率的梯度 $\nabla_\theta log \pi_\theta (y|x)$：描述了参数更新的方向

![](\images\CS336_Assignment5\img13.png)

通过表达式 $r(y|x)\nabla_\theta log \pi_\theta (y|x)$，可以提高高奖励回答的生成概率，让低奖励回答在训练中获得更少的正向强化。如果重复此过程并执行多个梯度更新步骤，我们就得到了一个基本的强化学习训练循环。

接下来是该公式的**推导过程**：
期望回报公式为：
$$J_{\theta} = \mathbb{E}_{x \sim \rho}\mathbb{E}_{y \sim \pi_\theta(\cdot | x)} [r(y|x)]$$
暂时固定输入 $prompt \quad x$，期望汇报为：
$$J_{x}(\theta) = \mathbb{E}_{y \sim \pi_\theta(\cdot | x)} [r(y|x)]$$
由于回答 y 从策略分布中采样（$\pi_\theta(y|x)$ 表示参数为 $\theta$ 的模型生成回答 y 的概率），所以期望可进一步写为：
$$J_{x}(\theta) = \sum_y\pi_\theta(y|x) r(y|x)$$
然后对 $\theta$ 求梯度：
$$\nabla_\theta J_{x}(\theta) = \sum_y \nabla_\theta \pi_\theta(y|x) r(y|x)$$
由于 $\frac{d}{dx} \log f(x) = \frac{f'(x)}{f(x)}$ ，所以有 $\nabla_\theta log \pi_\theta(y|x) = \frac{\nabla_\theta \pi_\theta(y|x)}{\pi_\theta(y|x)}$，即：
$$
\nabla_\theta \pi_\theta(y|x) = \pi_\theta(y|x) \nabla_\theta log \pi_\theta(y|x)
$$
将该式子代入上式得到：
$$\nabla_\theta J_{x}(\theta) = \sum_y \pi_\theta(y|x) \nabla_\theta log \pi_\theta(y|x) r(y|x)$$
根据期望的定义，该公式可进一步写为：
$$
\nabla_\theta J_{x}(\theta) = \mathbb{E}_{y \sim \pi_\theta(\cdot | x)}[r(y|x) \nabla_\theta log \pi_\theta(y|x)]
$$

最后，再对 prompt 的分布 $x \sim \rho$ 求期望，就得到

$$
\nabla_\theta J(\theta)=\mathbb{E}_{x \sim \rho}\mathbb{E}_{y \sim \pi_\theta(\cdot | x)}[r(y|x)\nabla_\theta log \pi_\theta (y|x)]
$$
由于我们无法枚举所有的可能回答，所以使用蒙特卡洛采样（Monte Carlo Sampling），用有限数量的样本近似期望。

从 prompt 分布 $\rho$ 中采样 B 个问题 $x$，然后对于每个问题，使用当前模型独立采样 $G$ 个回答，一共得到 B * G 个样本。$x^{(1)}, ..., x^{(B)} \overset{\mathrm{iid}}{\sim} \rho$，$y^{(i, j)} \overset{\mathrm{iid}}{\sim} \pi_\theta (y|x^{(i)})$

将上诉 REINFORCE 梯度公式带入具体样本，得到梯度公式（$\hat g$ 是对真实策略梯度 $\nabla_\theta J(\theta)$ 的样本估计）：
$$
\hat{g} = \frac{1}{BG} \sum_{i=1}^{B} \sum_{j=1}^{G} r(y^{(i,j)} \mid x^{(i)}) \nabla_\theta \log \pi_\theta(y^{(i,j)} \mid x^{(i)})
$$
最后梯度上升：
$$
\theta_{k+1}=\theta_k + \alpha \hat{g}
$$
### Baselines
REINFORCE 虽然能够正确估计策略梯度，但梯度的方差可能很大，导致强化学习训练不稳定。通过在奖励中减去一个基线，可以在不改变梯度期望的前提下，改变梯度估计的方差。

首先解释 REINFORCE 为什么会导致梯度的方差大。这里举一个比较极端的例子：

![](\images\CS336_Assignment5\img14.png)

![](\images\CS336_Assignment5\img15.png)

而且在 REINFORCE 中，当答对时，奖励为 1，提供正向信号，在后续训练中会增加该回答的概率。但当答错时，奖励为0，此时并不会提供直接的负面信号，降低该错误回答的概率，该错误回答对应的梯度权重为 0，REINFORCE 对错误回答只“不奖励”而不“惩罚”。

所以最简单的 basline 是从原先的奖励 r 中减去一个常数 b，
$$
\mathbb{E}_{x \sim \rho}\mathbb{E}_{y \sim \pi_\theta(y | x)}[r(y|x)\nabla_\theta log \pi_\theta (y|x)] \to \mathbb{E}_{x \sim \rho}\mathbb{E}_{y \sim \pi_\theta(y | x)}[(r(y|x)-b)\nabla_\theta log \pi_\theta (y|x)]
$$

![](\images\CS336_Assignment5\img16.png)

baseline 方法相对 REINFORCE，从模型中采样响应，对正确响应会增加权重，对错误响应会降低权重。同时，baseline 的核心目的不是简单地惩罚错误回答，而是在保持梯度期望不变的情况下，调整不同样本的梯度贡献，尽可能降低估计的方差。

$$
\mathbb{E}_{x, y}[r(y|x)\nabla_\theta log \pi_\theta (y|x)] = \mathbb{E}_{x, y}[(r(y|x)-b)\nabla_\theta log \pi_\theta (y|x)]
$$
下面从数学角度证明为什么减去常数 b 后，期望不变。为了简化符号，暂时固定 Prompt $x$：
$$
\begin{aligned} 
g_b(x)
&=\mathbb{E}_y[(r(y|x)-b)\nabla_\theta log \pi_\theta (y|x)] \\
&=\mathbb{E}_y[r(y|x)\nabla_\theta log \pi_\theta (y|x)-b\nabla_\theta log \pi_\theta (y|x)] \\
&= \mathbb{E}_y[r(y|x)\nabla_\theta log \pi_\theta (y|x)]-b\mathbb{E}_y[\nabla_\theta log \pi_\theta (y|x)]\\\\

\mathbb{E}_y[\nabla_\theta log \pi_\theta (y|x)]
&=\sum_{y}\pi_\theta(y|x)\nabla_\theta log \pi_\theta (y|x)
\\&=\sum_{y}\pi_\theta(y|x) \frac{\nabla_\theta \pi_\theta(y|x)}{\pi_\theta(y|x)}
\\&=\sum_{y}\nabla_\theta \pi_\theta(y|x)
\\&=\nabla_\theta \sum_{y}\pi_\theta(y|x)
\\&= \nabla_\theta 1 = 0
\\
\\
g_b(x) &= \mathbb{E}_y[r(y|x)\nabla_\theta log \pi_\theta (y|x)]-b\mathbb{E}_y[\nabla_\theta log \pi_\theta (y|x)]
\\&= \mathbb{E}_y[r(y|x)\nabla_\theta log \pi_\theta (y|x)]-b * 0 
\\ &= \mathbb{E}_y[r(y|x)\nabla_\theta log \pi_\theta (y|x)]
\end{aligned}
$$
所以当 baseline b 不依赖于当前采样的回答 y 时，减去 b 梯度的期望不会改变。下面解释 baseline 为什么能降低方差。
首先准确说法是 Baseline 可以降低方差，也可能增加方差，具体取决于基线的取值。
对于方差有 $Var(X) = E(X^2) - [E(x)]^2$，我们有 $\mathbb{E}_y[(r(y|x)-b)\nabla_\theta log \pi_\theta (y|x)] = \mathbb{E}_y[r(y|x)\nabla_\theta log \pi_\theta (y|x)]$，令 $\nabla_\theta log \pi_\theta (y|x) = Z, r(y|x)=r$，此时 $\mathbb{E}_y[((r-b)Z)^2]=\mathbb{E}_y[(r-b)^2 Z^2]=\mathbb{E}_y[(r^2-2br+b^2) Z^2]$，与 $\mathbb{E}_y[r^2 Z)^2]$ 一般情况下并不相等，且此时方差的变化就取决与 b 的选择。

总结：Baseline 通过从奖励中减去一个与当前回答无关的参考值，在不改变策略梯度期望的前提下，重新分配不同样本的梯度权重，从而有机会降低训练中的梯度方差。

### Baselines in GRPO
GRPO 的第一步是使用 Group Mean Baseline（组平均奖励基线）：
$$
\begin{aligned}
\hat{g} 
&= \frac{1}{BG} \sum_{i=1}^{B} \sum_{j=1}^{G} (r(y^{(i,j)} \mid x^{(i)})-\mu_i) \nabla_\theta \log \pi_\theta(y^{(i,j)} \mid x^{(i)})
\\ 
\mu_i &= \frac{1}{G} \sum_{j=1}^G r(y^{(i, j)} | x^{(i)})
\end{aligned}
$$
这些经过群体均值调整的奖励通常被称为“优势”，即 $r(y^{(i,j)} \mid x^{(i)})-\mu_i$ 被称为 Advantage（优势）：与绝对奖励不同，该术语代表的是该滚动策略相对于平均奖励所具有的“优势”。
在该策略下，奖励高于组平均水平的回答，获得正优势，倾向于提高其生成概率。奖励低于组平均水平的回答，获得负优势，倾向于降低其生成概率。奖励恰好等于组平均水平的回答，优势为零。

回顾 Baseline，普通 Baseline 通常要求基线 b 不依赖当前采样的回答，才能直接证明期望梯度完全不变。而 GRPO 使用的 Group Mean Baseline $\mu_i$ 本身依赖这一组回答，因此它并非严格保持原始期望梯度，而是会引入一个 $\frac{G-1}{G}$ 的缩放因子。下面给出具体的数学过程：

首先还是先固定同一个 Prompt x，并对符号进行一定程度的简写：
$$
r_j=r(y^{(j)}|x)
\\ Z_j=\nabla_\theta log \pi_\theta(y^{(j)}|x)
\\ \mu = \frac{1}{G} \sum_{k=1}^G r_k
$$
我们要计算的是梯度的期望，即：
$$
\mathbb{E}[\hat{g}]=\mathbb{E}[\frac{1}{G} \sum_{j=1}^G (r(y^{(j)} \mid x)-\mu) \nabla_\theta \log \pi_\theta(y^{(j)} \mid x)] = \mathbb{E}[\frac{1}{G} \sum_{j=1}^G (r_j -\mu) Z_j]
$$
这里有一点进行解释，我们对比的基准是 REINFORCE 中的梯度期望 $\mathbb{E}[\frac{1}{G} \sum_{j=1}^G r_j Z_j]=\frac{1}{G} \sum_{j=1}^G \mathbb{E}[r_j Z_j]$，因为 G 个回答都是从同一个策略 $\pi_\theta(\cdot\mid x)$ 独立采样的，那么每个回答的分布相同，因此：$\mathbb E[r_1Z_1] =\mathbb E[r_2Z_2] =\cdots =\mathbb E[r_GZ_G]$。所以 $\frac{1}{G} \sum_{j=1}^G \mathbb{E}[r_j Z_j]=\mathbb{E}[r Z]$。所以在后面我们使用 $\mathbb{E}[r Z]$ 来代替 $\frac{1}{G} \sum_{j=1}^G \mathbb{E}[r_j Z_j]$。
$$
\mathbb{E}[\hat{g}]= \mathbb{E}[\frac{1}{G} \sum_{j=1}^G (r_j -\mu) Z_j]
\\= \frac{1}{G} \sum_{j=1}^G \mathbb{E}[(r_j -\mu) Z_j]
$$
我们先看单个回答的期望贡献：

$$
\begin{aligned}
\mathbb{E}[(r_j - \mu) Z_j]
&= \mathbb{E}\left[\left(r_j - \frac{1}{G}\sum_{k=1}^{G}r_k\right)Z_j\right] \\
&= \mathbb{E}[r_j Z_j]
- \frac{1}{G}\sum_{k=1}^{G}\mathbb{E}[r_k Z_j]
\end{aligned}
$$

当 $k \neq j$ 时：由于不同回答是独立采样的，$r_k$ 与 $Z_j$ 独立，因此：

$$
\begin{aligned}
\mathbb{E}[r_k Z_j]
&= \mathbb{E}[r_k]\mathbb{E}[Z_j]
\\ \\
\mathbb{E}[Z_j]
&= \mathbb{E}[\nabla_\theta \log \pi_\theta(y^{(j)}|x)] \\
&= \sum_{j=1}^{G}\pi_\theta(y^{(j)}|x)
\nabla_\theta \log \pi_\theta(y^{(j)}|x) \\
&= \sum_{j=1}^{G}\pi_\theta(y^{(j)}|x)
\frac{\nabla_\theta \pi_\theta(y^{(j)}|x)}
{\pi_\theta(y^{(j)}|x)} \\
&= \sum_{j=1}^{G}\nabla_\theta \pi_\theta(y^{(j)}|x) \\
&= \nabla_\theta \sum_{j=1}^{G}\pi_\theta(y^{(j)}|x) \\
&= \nabla_\theta 1 = 0
\end{aligned}
$$

$$
\mathbb{E}[r_k Z_j]
= \mathbb{E}[r_k]\mathbb{E}[Z_j] = 0
$$

综上所述：

$$
\begin{aligned}
&\mathbb{E}[r_j Z_j]
- \frac{1}{G}\sum_{k=1}^{G}\mathbb{E}[r_k Z_j] \\
&= \mathbb{E}[r_j Z_j]
- \frac{1}{G}\mathbb{E}[r_j Z_j]
- \frac{1}{G}\sum_{\substack{k=1 \\ k\neq j}}^{G}
\mathbb{E}[r_k Z_j] \\
&= \frac{G-1}{G}\mathbb{E}[r_j Z_j]
\end{aligned}
$$

对 G 个回答取平均：
$$
\begin{aligned}
\mathbb{E}[\hat{g}]
&= \frac{1}{G} \sum_{j=1}^G \mathbb{E}[(r_j -\mu) Z_j]
\\&= \frac{1}{G} \sum_{j=1}^G \frac{G-1}{G} \mathbb{E}[r_j Z_j]
\\&= \frac{G-1}{G} \frac{1}{G} \sum_{j=1}^G \mathbb{E}[r_j Z_j]
\\&= \frac{G-1}{G} \mathbb{E}[r Z]
\end{aligned}
$$
GRPO 利用同一 Prompt 下的多个回答构造相对奖励信号；尽管组均值依赖采样回答，在独立采样等条件下，其期望梯度仍等于原始策略梯度乘以 $\frac{G-1}{G}$。
#### Advantage normalization
在GRPO中，扣除组均值后的下一步调整，将是除以组标准差：
$$
\begin{aligned}
\hat{g} &= \frac{1}{BG} \sum_{i=1}^{B} \sum_{j=1}^{G} \frac{r(y^{(i,j)} \mid x^{(i)})-\mu_i}{std_i} \nabla_\theta \log \pi_\theta(y^{(i,j)} \mid x^{(i)})
\\ 
\mu_i &= \frac{1}{G} \sum_{j=1}^G r(y^{(i, j)} | x^{(i)})
\\
std_i &= \sqrt{\frac{1}{G}\sum_{j=1}^G(r(y^{(i, j)}|x^{(i)})-\mu_i)^2}
\end{aligned}
$$
这被称为  Advantage Normalization（优势归一化），先减去同组奖励的平均值，再除以同组奖励的标准差，让不同 prompt 产生的梯度更新更加稳定。

为什么要除以标准差？

![](\images\CS336_Assignment5\img17.png)

Advantage Normalization 通过将每组 Advantage 减去组内均值并除以组内标准差，使其均值为 0、标准差为 1，从而消除不同组之间奖励尺度和波动程度的差异，使梯度更新的尺度更加稳定，提升训练过程的稳定性。需要注意的是，该操作并不保证最终梯度向量的范数相同，因为梯度还受到模型对数概率梯度的方向和大小等因素影响。

**但是**，在 GRPO 中，只减去组内均值时，由于组内均值包含当前回答的奖励，在独立采样等条件下，梯度估计的期望会缩小为原始梯度的 $\frac{G-1}{G}$，因此虽然是有偏估计，但只改变梯度大小，不改变其期望方向，仍可用于优化原始目标。而进一步除以组内标准差时，由于标准差依赖当前组的奖励，可能改变不同梯度贡献的相对权重，**因此期望梯度的大小和方向都可能发生变化**，不再保证是原始期望奖励目标的无偏梯度估计。尽管如此，Advantage Normalization 能够调整不同组的优势值尺度，减轻奖励波动差异对梯度更新的影响，从而提升训练稳定性。

如果假设每个梯度向量都是独立同分布的高斯分布，那么除以组标准差可确保每个组的梯度更新在范数上大致相等。我们将在下一节中更深入地探讨优势归一化。

#### Sequence normalization 
GRPO 中还有另一个细节，似乎是继承自早期的 LLM-PPO 实现，称为序列归一化。回想一下，我们（到目前为止）的策略梯度估计器如下：
$$
\hat{g} = \frac{1}{BG} \sum_{i=1}^{B} \sum_{j=1}^{G} \frac{r(y^{(i,j)} \mid x^{(i)})-\mu_i}{std_i} \nabla_\theta \log \pi_\theta(y^{(i,j)} \mid x^{(i)})
$$
我们将 $log \pi_\theta (y^{(i,j)} | x^{(i)})$ 展开为对时间步的求和，首先解释一下为什么可以展开：

假设输入 prompt 是 $x$，模型生成的回答是 $y=(y_1,y_2,\ldots,y_T)$，根据概率的链式法则，整个回答的概率可以写作：

$$
\pi_\theta(y|x)
= \prod_{t=1}^{T}
\pi_\theta(y_t|x,y_{<t})
$$

其中 $y_{<t}=(y_1,y_2,\ldots,y_{t-1})$，表示当前 token 之前已经生成的所有 token。

两边取对数，得到：

$$
\log \pi_\theta(y|x)
= \sum_{t=1}^{T}
\log \pi_\theta(y_t|x,y_{<t})
$$

再对模型参数 $\theta$ 求梯度，得到：

$$
\nabla_\theta \log \pi_\theta(y|x)
= \nabla_\theta \sum_{t=1}^{T}
\log \pi_\theta(y_t|x,y_{<t})
$$

所以，梯度策略估计器为：
$$
\hat{g} = \frac{1}{BG} \sum_{i=1}^{B} \sum_{j=1}^{G} \frac{r(y^{(i,j)} \mid x^{(i)})-\mu_i}{std_i} \nabla_\theta \sum_{t=1}^{len(y^{(i,j)})}\log \pi_\theta(y_t^{(i,j)} \mid x^{(i)}, y_{<t}^{(i,j)})
$$
在该梯度策略估计器下，一个回答的梯度贡献是该序列所有 token 梯度的累加，再乘以序列级优势值。因此长回答包含更多 token 梯度项，其总梯度贡献可能更大（最终大小还取决于各 token 梯度的方向与相互抵消，并不严格正比于长度）
因此 GRPO 增加了一个额外的序列长度归一化项：
$$
\hat{g} = \frac{1}{BG} \sum_{i=1}^{B} \sum_{j=1}^{G} \frac{r(y^{(i,j)} \mid x^{(i)})-\mu_i}{std_i} \nabla_\theta (\frac{1}{len(y^{(i,j)})}\sum_{t=1}^{len(y^{(i,j)})}\log \pi_\theta(y_t^{(i,j)} \mid x^{(i)}, y_{<t}^{(i,j)}))
$$
这一修改进一步改变了期望值：它不再对不同序列中的所有 token 赋予相等的权重，而是相对于较短序列中的 token，对较长序列中的 token 降低了权重。

归一化前，所有 token 权重相同，序列总贡献大致正比于长度；归一化后，每个序列的总权重被拉平为 1，长序列中每个 token 的权重 $\frac{1}{L}$ 更小、短序列中更大，从而消除了"长序列因 token 多而主导更新"的倾向。同时，由于 $L$ 是依赖策略采样的随机变量，序列归一化与 std 归一化一样会进一步引入偏差，但不同长度序列间的更新尺度会更均衡。

在下一节中，我们将讨论这一选择，并进行消融实验以验证该修改是否合理。但目前，我们将仅实现原始论文中提出的标准 GRPO 算法，因此会包含序列归一化步骤。

#### Putting everything together
GRPO还有一个尚未提及的组成部分，即“截断重要性重新加权”（clipped importance reweighting）。Clipped Importance Reweighting 是为了解决 Off-policy RL 中使用过期样本的问题，我们将在本作业后面的部分讨论离策略强化学习（off-policy RL）时，再详细说明这一组成部分。在这里简单介绍一下什么是 on-policy RL、什么是 off-policy RL。
- On-policy RL 的核心思想是：用当前模型生成的数据，更新当前的模型。
	- 假设我们有一个语言模型 $\pi_\theta$。训练过程如下：
		1. 使用当前模型 $\pi_{\theta_0}$ 生成一批回答 
		2. 根据这些回答计算奖励、梯度，更新模型参数，得到 $\pi_{\theta_1}$
		3. 再用更新后的模型生成下一批数据，训练得到 $\pi_{\theta_2}$
	
	![](\images\CS336_Assignment5\img55.png)

- Off-policy RL 的核心思想是：一批数据可以重复使用多次，进行多次梯度更新，而不需要每次都重新生成数据。
	- 假设我们有一个语言模型 $\pi_\theta$。训练过程如下：
		1. 使用当前模型 $\pi_{\theta_0}$ 生成一批回答
		2. 根据这些回答计算奖励、梯度，更新模型参数，得到 $\pi_{\theta_1}$
		3. 继续使用原来的数据更新模型，训练得到 $\pi_{\theta_2}$
	
	![](\images\CS336_Assignment5\img56.png)

在 Off-policy RL 中，后续更新时，训练数据与当前模型不再匹配，这些数据就变成了 Stale Samples（过期样本）。所以需要 Clipped Importance Reweighting，通过调整样本权重，并限制权重范围，帮助模型利用这些过期数据进行训练。

![](\images\CS336_Assignment5\img18.png)
## Implementing on-policy GRPO
### Using Hugging Face models
在本次作业中，我们将直接使用 Hugging Face 的 transformers 库来加载预训练的基础模型。要加载 Hugging Face 模型和分词器（采用 bfloat16 格式并使用 FlashAttention-2 以节省内存），可以使用 cs336_alignment/checkpoint.py 中的以下示例代码：

注：模型参数通常使用 FP32，这里使用的是 BF16。FP32：32bit，BF16：16bit，使用 BF16 可以减少模型权重占用的显存。eager：普通 Attention 实现。flash_attention_2：
FlashAttention-2，一种更节省显存、通常也更高效的 Attention 实现。

![](\images\CS336_Assignment5\img19.png)

保存训练好的模型：
```
model.save_pretrained(save_directory=output_dir)
tokenizer.save_pretrained(save_directory=output_dir)
```
首先，让我们实现一个辅助函数，该函数使用预训练的分词器对 prompts 和 responses 分别进行分词，不添加特殊令牌，然后将它们拼接为 prompt_ids + response_ids，其间不包含 EOS、BOS 或分隔符令牌。

在训练时，我们只对 response tokens 计算 loss。因为 prompt 是用户给模型的输入，而 response 才是模型需要学习生成的输出。我们希望模型学习：给定 prompt，应该如何生成正确的 response。而不是让模型去学习重复生成 prompt。所以我们还需要构建一个 response_mask：这是一个布尔掩码，与偏移后的标签对齐，当标签令牌来自 response 时，该位置的掩码值为 True；当位于 prompt 或 padding position时，则为 False。我们将在训练循环中使用此掩码，以确保仅对 response tokens 计算损失。

![](\images\CS336_Assignment5\img20.png)

![](\images\CS336_Assignment5\img21.png)

![](\images\CS336_Assignment5\img22.png)

经过上述 tokenize 操作之后，我们可以通过下面的代码将其传入模型：
```
input_ids = train_batch["input_ids"].to(device) 
labels = train_batch["labels"].to(device) 
logits = model(input_ids).logits
```

![](\images\CS336_Assignment5\img23.png)

![](\images\CS336_Assignment5\img24.png)

![](\images\CS336_Assignment5\img25.png)


### Using vLLM in a reinforcement learning loop
（具体训练时，用于优化）
这一部分主要是介绍在强化学习的训练循环中使用 vLLM 加速模型生成，并且在训练 GPU 和推理 GPU 之间同步模型权重。

我们在强化学习的训练循环中使用两块 GPU，将 Hugging Face 模型和优化器部署在 GPU 0 上进行训练，将 vLLM 部署在 GPU 1 上负责生成回答，每次 vLLM 生成回答之前需要将 GPU 0 上训练得到的最近权重同步给 vLLM，确保生成使用的是最新策略。

不直接用 Hugging Face 生成，是因为 vLLM 专门针对 LLM 推理进行了优化，把训练和推理分开，可以减少两者之间的资源竞争。

权重同步代码位于 cs336_alignment/vllm_utils.py：

![](\images\CS336_Assignment5\img26.png)

### GRPO components
![](\images\CS336_Assignment5\img27.png)

![](\images\CS336_Assignment5\img28.png)

![](\images\CS336_Assignment5\img29.png)


我们现在得到的 `raw_rewards` 是一个扁平的一维张量，形状是 `(rollout_batch_size,)`。接下来，将它按 group 重新 reshape，然后在组内做归一化，得到 GRPO 算法真正需要的东西——**advantages（优势值）**。

![](\images\CS336_Assignment5\img30.png)

![](\images\CS336_Assignment5\img31.png)

![](\images\CS336_Assignment5\img32.png)

接下来利用每条回答的 Advantage 和模型生成每个 token 的 log probability，计算 GRPO 的 per-token 策略梯度损失。**由于 PyTorch 优化器执行的是梯度下降而非上升，所以函数应返回该目标的负值作为 loss**。

![](\images\CS336_Assignment5\img33.png)


注：因为

$$
\begin{aligned}
\nabla_\theta J^{\mathrm{GRPO\text{-}on\text{-}policy}}
&= \hat{g} \\
&= \frac{1}{BG}
\sum_{i=1}^{B}
\sum_{j=1}^{G}
\frac{1}{\operatorname{len}(y^{(i,j)})}
\sum_{t=1}^{\operatorname{len}(y^{(i,j)})}
\frac{r(y^{(i,j)} \mid x^{(i)})-\mu_i}
{\mathrm{std}_i}
\nabla_\theta
\log \pi_\theta
\left(
y_t^{(i,j)}
\mid x^{(i)},y_{<t}^{(i,j)}
\right)
\end{aligned}
$$

所以 $J_\theta^{\mathrm{GRPO\text{-}on\text{-}policy}}$ 就是对应的损失函数。

**注意这一部分都是序列归一化（sequence），在后面会介绍常数归一化（constant）**
compute_policy_gradient_loss 将计算每个令牌和每个序列的损失项，而 aggregate_loss_across_microbatch_sequence 则会对所有令牌和序列的每令牌损失进行求平均。

![](\images\CS336_Assignment5\img34.png)

当前方案实现比较简单：

![](\images\CS336_Assignment5\img35.png)

![](\images\CS336_Assignment5\img36.png)

![](\images\CS336_Assignment5\img37.png)

![](\images\CS336_Assignment5\img38.png)

![](\images\CS336_Assignment5\img39.png)


### GRPO training step
这部分将前面的 GRPO 组件组合为完整的训练步骤
#### Gradient accumulation
为了在推理阶段实现良好的资源利用率，我们需要较大的 batch size 大小。因此，在 On-policy RL 中，由于我们将训练的 batch size 设为与推理的 batch size 相同，GPU 将没有足够的内存来一次性计算整个批次的梯度。因此，我们需要将批次拆分为一系列 Microbatch，并在这些 Microbatch 之间累积梯度。

![](\images\CS336_Assignment5\img40.png)

首先回顾一下在 PyTorch 中实现梯度累积：
```
# Forward
logits = model(inputs)
loss = loss_fn(logits, labels)

# Backward
# 计算梯度，并累积到参数的 .grad 中
loss.backward()

# Update parameters
# 根据梯度更新模型参数
optimizer.step()

# Clear gradients
# 清空之前累积的梯度
optimizer.zero_grad()
```
这里注意 **PyTorch 的 `loss.backward()` 默认不会覆盖之前的梯度，而是累加梯度**。所以我们每次需要通过 optimizer.zero_grad() 清除梯度。这一特性也是 Gradient Accumulation 的基础。

为了实现梯度累积，我们将一个 batch 划分为 k 个 microbatches，并仅调用一次 optimizer.step() 和 optimizer.zero_grad()。假设采用序列归一化方法（sequence normalization, 即先对每个序列计算损失的平均值，再对所有序列进行平均）在计算出微批量的平均损失后，我们还需要根据序列数量重新加权。
```
gradient_accumulation_steps = 4

microbatch_size = (
    len(inputs) // gradient_accumulation_steps
)

for i in range(0, len(inputs), microbatch_size):

    inputs_microbatch = inputs[i:i+microbatch_size]
    labels_microbatch = labels[i:i+microbatch_size]

    # Forward pass
    logits = model(inputs_microbatch)

    loss = loss_fn(logits, labels_microbatch)

    # Important: reweight the loss
    loss = loss * (
        len(inputs_microbatch) / len(inputs)
    )

    # Backward pass
    loss.backward()

# Update only once
optimizer.step()

# Clear gradients
optimizer.zero_grad()
```

![](\images\CS336_Assignment5\img41.png)

请注意，稍后当我们改用常数归一化（constant）代替序列归一化（sequence）时，微批次代码会略作调整，以适应不同的归一化方式。
#### Implement the train step

![](\images\CS336_Assignment5\img42.png)

测试出错，不知道问题在哪，先跳过
## Experiments
之后补充
# RL algorithm variants

## Dr. GRPO
在理解 GRPO 的推导过程后，我们可以发现，传统 GRPO 在计算 Policy Gradient 时引入了两种归一化操作：Advantage Standard Deviation Normalization（优势标准差归一化）和 Sequence Length Normalization（序列长度归一化）。这两种操作虽然有助于稳定训练，但也使梯度估计器不再具有原本期望的形式。针对这一问题，Dr. GRPO 提出了相应的改进方法，主要是取消这两种归一化操作。
1. Length Normalization（序列长度归一化）
	首先来看序列长度归一化。传统 GRPO 会将每条 sequence 的 loss 除以其实际长度，使不同长度的回答具有大致相同的总权重。这种方法的优点是避免长回答因为包含更多 token 而对训练产生过大的影响。但是，它也会导致短回答中每个 token 的权重更大，而长回答中每个 token 的权重更小，从而可能使模型过度关注短回答。
	
	相比之下，Dr. GRPO 使用统一的常数 Z 对所有 token 的 loss 进行归一化，而不是分别除以每条 sequence 的长度。其中，Z 通常设置为 B × G × L，分别代表 batch size、每个 prompt 生成的回答数量以及最大生成长度。这样可以保证每个 token 使用相同的归一化权重，但长回答由于包含更多 token，其对整体 loss 的贡献也会更大。
	
	因此，如果希望不同回答具有相近的总权重，Sequence Length Normalization 更合适；如果希望避免短回答中的 token 被额外放大，则 Constant Normalization 更合适。

2. Group Normalized Rewards（组奖励归一化）

	除了序列长度归一化之外，Dr. GRPO 还对 Advantage 的计算方式进行了修改。
	
	在传统 GRPO 中，首先计算同一个 prompt 下多个回答的平均 reward，作为 baseline，然后计算每个回答的 advantage：$A_i = r_i - μ$。随后，传统 GRPO 会进一步除以组内 reward 的标准差：$Â_i = \frac{r_i - μ}{\sigma}$，这种标准差归一化可以调整 advantage 的数值范围，但也可能引入额外的梯度权重变化。
	
	Dr. GRPO 则取消了标准差归一化，只保留减去组内平均 reward 的操作，即 $A_i = r_i - μ$。

![](\images\CS336_Assignment5\img43.png)

![](\images\CS336_Assignment5\img44.png)

这部分代码之前就已经实现了。
## Rejection fine tuning

这一部分介绍的是另一种比 GRPO 更简单的强化学习训练方法：Rejection Fine-Tuning（RFT，拒绝法微调），也叫 Expert Iteration（EI，专家迭代）。这种方法不需要计算复杂的 policy gradient，也不需要对 reward 进行组内归一化，而是通过模型生成多个回答，筛选出正确的回答，再使用监督学习（SFT）训练模型，让模型更倾向于生成正确答案。

![](\images\CS336_Assignment5\img45.png)

![](\images\CS336_Assignment5\img46.png)

其中 Z 是一个常数归一化因子（类似于 Dr. GRPO）, $\mathbb{1}{r(y(i,j) | x(i)) = 1}$ 是一个仅保留正确答案的指示函数。
## MaxRL
最后，我们将探讨一种名为“最大似然强化学习”（Maximum Likelihood Reinforcement Learning，简称 MaxRL）的最新方法 。与 GRPO 中将优势值除以组标准差进行归一化，或 Dr. GRPO 中不进行任何归一化处理不同，MaxRL 提出将优势值除以组均值。这一选择得出了如下梯度估计量：
$$
\hat g= \frac{1}{\sum_{i=1}^{B} \sum_{j=1}^{G}len(y^{(i, j)}} \sum_{i=1}^{B} \sum_{j=1}^{G} \sum_{t=1}^{T_{ij}} \frac{r(y^{(i,j)}|x^{(i)})-\mu_i}{\mu_i} \nabla_\theta\log\pi_\theta(y_t^{(i,j)}|x^{(i)},y_{<t}^{(i,j)}) 
$$
# Off-policy RL
前面实验使用的是 On-policy RL，即每生成一批训练数据，就进行一次模型更新，然后使用更新后的模型重新生成数据。这种方式能够保证训练数据来自当前策略，但需要频繁进行模型推理。下面我们介绍 Off-policy RL。
Off-policy RL 允许对同一批生成数据进行多次训练更新，从而提高数据利用率，减少 Inference 的计算开销，并有可能加快训练速度。不过，随着模型不断更新，训练数据可能逐渐偏离当前策略，造成 Policy Mismatch 和训练不稳定，因此需要额外的机制来控制这种偏差。
## Importance reweighting
在 off-policy RL 中，由于我们使用旧策略生成的数据来训练当前策略，所以会产生误差。假设模型最初使用 $\pi_0$ 生成了一批回答。第一次更新之后，模型参数发生变化，当前策略变成了 $\pi_\theta$。此时，如果继续使用原来的回答进行训练，就出现了一个问题：这些回答是旧策略 $\pi_0$ 生成的，而我们真正想计算的是当前策略 $\pi_\theta$ 的 policy gradient。
对于当前策略，理论上 policy gradient 的期望形式是：
$$
\mathbb{E}_{x\sim\rho}
\mathbb{E}_{y\sim\pi_\theta(y|x)}
\left[
r(y|x)\nabla_\theta\log\pi_\theta(y|x)
\right]
$$

但是，我们训练使用的数据来自 $\pi_0$，因此实际计算出来的朴素估计量是：
$$
\mathbb{E}_{x\sim\rho}
\mathbb{E}_{y\sim\pi_0(y|x)}
\left[
r(y|x)\nabla_\theta\log\pi_\theta(y|x)
\right]
$$

这两个期望通常不相等：
$$
\mathbb{E}_{y\sim\pi_0}[\cdots]
\neq
\mathbb{E}_{y\sim\pi_\theta}[\cdots]
$$
如果我们直接把旧策略生成的数据当作当前策略生成的数据来计算梯度，就会产生估计偏差。这就是文中所说的 naive off-policy estimator（朴素的离策略估计量）。
纠正这种偏差的一种方法是使用 importance reweighting（重要性加权）。它引入一个重要性权重：
$$
w(y|x)= \frac{\pi_\theta(y|x)} {\pi_0(y|x)}
$$ 
然后将原来的估计量修改为：
$$
\mathbb{E}_{x\sim\rho} \mathbb{E}_{y\sim\pi_0(y|x)} \left[ \frac{\pi_\theta(y|x)}{\pi_0(y|x)} r(y|x)\nabla_\theta\log\pi_\theta(y|x) \right]
$$
这个权重的直观含义是：
- 如果某个回答在当前策略下的概率比在旧策略下高，那么它的权重就大于 1，训练时会增加它的贡献。
- 如果某个回答在当前策略下的概率比在旧策略下低，那么它的权重就小于 1，训练时会降低它的贡献。
- 如果某个回答在当前策略下已经几乎不可能出现，那么它的权重就会很小，对当前策略的梯度贡献也会很小。
换句话说，重要性权重会调整旧策略生成的样本，使它们在加权后能够更好地代表当前策略的样本分布。
从数学角度分析为什么 importance reweighting 可以纠正 off-policy 偏差，还是先固定 x：
$$
\begin{aligned}
\mathbb{E}_{y\sim\pi_0(y|x)} \left[ \frac{\pi_\theta(y|x)}{\pi_0(y|x)} r(y|x)\nabla_\theta\log\pi_\theta(y|x) \right]
&= \sum_y \pi_0(y|x) \frac{\pi_\theta(y|x)}{\pi_0(y|x)} r(y|x)\nabla_\theta\log\pi_\theta(y|x)
\\ &= \sum_y \pi_\theta(y|x) r(y|x)\nabla_\theta\log\pi_\theta(y|x)
\\ &= \mathbb{E}_{y\sim\pi_\theta(y|x)} \left[r(y|x)\nabla_\theta\log\pi_\theta(y|x) \right]
\end{aligned}
$$
但是 importance reweighting 虽然解决了偏差问题，但它引入了另一个问题：**高方差**。
假设某个样本的重要性权重为：
$$
w(y|x)=\frac{\pi_\theta(y|x)}{\pi_0(y|x)}
$$
如果这个权重非常大，那么这个样本对最终梯度的贡献也可能非常大。这样一来，少数样本就可能主导整个梯度估计，使不同 batch 之间的梯度变化剧烈，训练变得不稳定。

对于语言模型，这个问题尤其严重，因为一个回答通常由很多 token 组成。一个完整回答的概率可以写成：
$$
\pi_\theta(y|x)=\prod_{t=1}^{T}\pi_\theta(y_t|x,y_{<t})
$$
因此，重要性权重为：
$$
\frac{\pi_\theta(y|x)}{\pi_0(y|x)}=\prod_{t=1}^{T}
\frac{
\pi_\theta(y_t|x,y_{<t})
}{
\pi_0(y_t|x,y_{<t})
}
$$
也就是说，整条回答的重要性权重，是每个 token 的概率比值连乘得到的。随着回答长度增加，即使每个 token 的概率比值只有轻微变化，最终的权重也可能变得非常大，从而显著放大梯度估计的方差。

举个简单例子：假设每个 token 的概率比值都是 1.1，那么：
$$
1.1^{100}\approx 13780.6
$$
单个 token 看起来只有 10% 的变化，但 100 个 token 连乘后，整条回答的重要性权重就已经非常大了。

因此，朴素的 Off-policy estimator 虽然有偏，但没有额外的权重放大问题；完整 Importance Reweighting 虽然在理想条件下无偏，却可能具有很高的方差。后续方法的核心，就是在 Bias（偏差）和 Variance（方差）之间进行权衡，在提高数据利用率的同时保持训练稳定性。
## PPO/GRPO-style importance reweighting and clipping
### Token-level reweighting
上一节提到的 importance reweighting 方法是 sequence-level importance：
$$
\mathbb E_{x\sim\rho} \mathbb E_{y\sim\pi_0} \left[ r(y|x) \left( \prod_{t=1}^{T} \frac{\pi_\theta(y_t|x,y_{<t})} {\pi_0(y_t|x,y_{<t})} \right) \sum_{t=1}^{T} \nabla_\theta\log\pi_\theta(y_t|x,y_{<t}) \right]
$$
Sequence-level Reweighting 使用整条回答的概率比值，即所有 token 概率比值的乘积；而 在 PPO 和 GRPO 中采用的是 Token-level Reweighting，为每个 token 单独计算概率比值。Token-level Reweighting 可以避免整条回答的概率比值连乘所带来的高方差问题。
Token-level Reweighting 的公式为：
$$
\mathbb E_{x\sim\rho} \mathbb E_{y\sim\pi_0} \left[ r(y|x)\sum_{t=1}^{T}\frac{\pi_\theta(y_t|x,y_{<t})} {\pi_0(y_t|x,y_{<t})} \nabla_\theta\log\pi_\theta(y_t|x,y_{<t}) \right]
$$
注意，Token-level Reweighting 并不是 Sequence-level Reweighting 的简单改写，而是改变了原本的梯度估计方式。

![](\images\CS336_Assignment5\img47.png)
 
Token-level Reweighting 实际优化的目标不是当前策略 $\pi_\theta$ 下的期望 reward，而是所有代理策略的期望 reward 之和。定义代理策略 $\tilde{\pi}_t$ 为一种混合策略：生成回答时，只有第 $t$ 个 token 使用当前策略 $\pi_\theta$，其余 token 都使用旧策略 $\pi_0$。具体表示为：

$$\tilde{\pi}_t(y|x)
=
\left(
\prod_{s=1}^{t-1}
\pi_0(y_s|x,y_{<s})
\right)
\pi_\theta(y_t|x,y_{<t})
\left(
\prod_{s=t+1}^{T}
\pi_0(y_s|x,y_{<s})
\right)

$$
例如一条长度为 4 的序列：
$$
y=(y_1,y_2,y_3,y_4)
$$
如果 $t=2$，那么：
$$
\tilde\pi_2(y)
=
\pi_0(y_1)
\pi_\theta(y_2\mid y_1)
\pi_0(y_3\mid y_1,y_2)
\pi_0(y_4\mid y_1,y_2,y_3)
$$
注意这里一个非常容易忽略的细节：虽然 $y_3,y_4$ 是由旧策略生成的，但它们的条件上下文包含新策略生成的 $y_2$。所以混合策略并不是把几段独立文本直接拼起来。它仍然是一个合法的自回归生成过程。
下面进行数学推导，我们先固定某个 t，定义：


$$
J_t(\theta)
= \mathbb{E}_{y\sim\tilde{\pi}_t}[r(y)]
$$

在旧策略下生成完整序列的概率：

$$
\begin{aligned}
\pi_0(y)
&= \prod_{s=1}^{L}\pi_0(y_s\mid y_{<s}) \\
&= \left(\prod_{s<t}\pi_0(y_s\mid y_{<s})\right)
\pi_0(y_t\mid y_{<t})
\left(\prod_{s>t}^{L}\pi_0(y_s\mid y_{<s})\right)
\end{aligned}
$$

而混合策略下：

$$
\tilde{\pi}_t(y)
=
\left(\prod_{s<t}\pi_0(y_s\mid y_{<s})\right)
\pi_\theta(y_t\mid y_{<t})
\left(\prod_{s>t}^{L}\pi_0(y_s\mid y_{<s})\right)
$$

两者相除，可以得到：

$$
\frac{\pi_0(y)}{\tilde{\pi}_t(y)}
=
\frac{\pi_0(y_t\mid y_{<t})}
{\pi_\theta(y_t\mid y_{<t})}
$$

所以：

$$
\tilde{\pi}_t(y)
=
\pi_0(y)
\frac{\pi_\theta(y_t\mid y_{<t})}
{\pi_0(y_t\mid y_{<t})}
$$

$$
\begin{aligned}
J_t(\theta)
&= \mathbb{E}_{y\sim\tilde{\pi}_t}[r(y)] \\
&= \sum_y \tilde{\pi}_t(y)r(y) \\
&= \sum_y \pi_0(y)
\frac{\pi_\theta(y_t\mid y_{<t})}
{\pi_0(y_t\mid y_{<t})}r(y) \\
&= \mathbb{E}_{y\sim\pi_0}
\left[
\frac{\pi_\theta(y_t\mid y_{<t})}
{\pi_0(y_t\mid y_{<t})}r(y)
\right]
\end{aligned}
$$

求梯度，由于采样分布 $\pi_0$ 是固定的，与 $\theta$ 无关，可以把梯度放入期望：

$$
\nabla_\theta J_t(\theta)
=
\mathbb{E}_{y\sim\pi_0}
\left[
\frac{\nabla_\theta\pi_\theta(y_t\mid y_{<t})}
{\pi_0(y_t\mid y_{<t})}r(y)
\right]
$$

又因为：

$$
\nabla_\theta\pi_\theta
=
\pi_\theta\nabla_\theta\log\pi_\theta
$$

所以：

$$
\nabla_\theta J_t(\theta)
=
\mathbb{E}_{y\sim\pi_0}
\left[
r(y)
\frac{\pi_\theta(y_t\mid y_{<t})}
{\pi_0(y_t\mid y_{<t})}
\nabla_\theta\log\pi_\theta(y_t\mid y_{<t})
\right]
$$

定义整体的混合策略：

$$
J_{\text{token}}(\theta)
=
\sum_{t=1}^{L}J_t(\theta)
$$

$$
\begin{aligned}
\nabla_\theta J_{\text{token}}(\theta)
&= \sum_{t=1}^{L}\nabla_\theta J_t(\theta) \\
&= \sum_{t=1}^{L}
\mathbb{E}_{y\sim\pi_0}
\left[
r(y)
\frac{\pi_\theta(y_t\mid y_{<t})}
{\pi_0(y_t\mid y_{<t})}
\nabla_\theta\log\pi_\theta(y_t\mid y_{<t})
\right] \\
&= \mathbb{E}_{y\sim\pi_0}
\left[
r(y)\sum_{t=1}^{L}
\frac{\pi_\theta(y_t\mid y_{<t})}
{\pi_0(y_t\mid y_{<t})}
\nabla_\theta\log\pi_\theta(y_t\mid y_{<t})
\right]
\end{aligned}
$$

虽然 Token-level Reweighting 降低了方差，但它优化的surrogate objective 与真实策略的优化目标并不一致。 真实目标是最大化完整序列由当前策略 $\pi_\theta$ 生成时的期望奖励，而 surrogate objective 则假设每次只有第 $t$ 个 token 由当前策略 $\pi_\theta$ 生成，其余 token 均由旧策略 $\pi_0$ 生成。因此，虽然 token-level importance sampling 能够正确估计 surrogate objective 的梯度，但这个梯度通常并不等于真实目标的梯度。

偏差主要有以下两种：

(1) Prefix Bias（前缀偏差）

在计算第 $t$ 个 token 的梯度时，其前缀 $y_{<t}$ 仍然由旧策略 $\pi_0$ 生成，而非当前策略 $\pi_\theta$。随着模型参数不断更新，新旧策略的生成分布可能逐渐产生差异，使得旧策略生成的前缀无法准确代表当前策略实际会遇到的上下文。因此，Token-level Reweighting 实际上是在旧策略的前缀分布下计算当前策略的梯度，导致训练时使用的上下文分布与当前策略的真实状态分布不一致，从而引入偏差。

(2) Suffix Bias（后缀偏差）

在第 $t$ 个 token 由当前策略 $\pi_\theta$ 生成后，后续 token $y_{>t}$ 仍然由旧策略 $\pi_0$ 生成。这意味着，在评估当前 token 对最终奖励的影响时，使用的是旧策略的后续生成行为，而非当前策略的后续生成行为。随着策略更新，当前策略可能已经具备更强的后续生成能力，但旧策略生成的后缀无法反映这一变化。因此，Token-level Reweighting 实际上是基于旧策略的后续生成结果来评估当前 token 的价值，可能导致对其真实贡献的估计产生偏差。
这两种偏差都会随着当前策略 $\pi_\theta$ 与旧策略 $\pi_0$ 之间的差距增大而变得更加严重。

Token-level Importance Reweighting 是一种在偏差和方差之间进行权衡的方法。它通过将整条回答的概率比值拆分为每个 token 独立的概率比值，降低了重要性权重随回答长度指数增长的风险，但也因此无法完整校正旧策略与当前策略之间的分布差异。PPO 和 GRPO 采用这种方法，体现的正是 Off-policy RL 中 Bias-Variance Tradeoff 的思想。
### Clipping
除了 token-level reweighting，PPO 和 GRPO 引入的另一个启发式方法是 importance weight clipping。如上所述，当 $\pi_\theta$ 和 $\pi_0$ 差距增大时，$\frac{\pi_\theta(y_t|y_{<t})}{\pi_0(y_t|y_{<t})}$增大，由于$\mathbb{E}_{y\sim\pi_0}[r(y) \sum_{t=1}^{L} \frac{\pi_\theta(y_t|y_{<t})}{\pi_0(y_t|y_{<t})}\nabla_\theta log\pi_\theta(y_t|y_{<t})]$，那么 importance reweighting 的方差可能变得很大，从而导致训练不稳定。因此，为了保证 Off-policy RL 的稳定性，我们需要通过 clipping 来确保 $\pi_\theta$ 始终保持在 $\pi_0$ 附近。

近期论文对截断的具体实现方式存在分歧，但在本作业中，我们将实现 PPO 中的截断方法，该方法一直沿用至 GRPO，且似乎已成为主流做法。首先，将 token-level reweighting 与标准的 GRPO 目标函数相结合，可得到以下损失函数：
$$
J_\theta^{GRPO-off-policy-noclip} = \frac{1}{BG} \sum_{i=1}^{B}\sum_{j=1}^{G} \frac{1}{len(y^{(i, j)})} \frac{r(y^{(i, j)}|x^{(i)})-\mu_i}{std_i}\sum_{t=1}^{len(y^{(i,j)})} \frac{\pi_\theta(y_t|x, y_{<t})}{\pi_0(y_t|x, y_{<t})}
$$
解释一下为什么是这个公式：

$$
\begin{aligned}
\nabla_\theta J_\theta^{\mathrm{GRPO\text{-}off\text{-}policy\text{-}noclip}}
&= \frac{1}{BG}
\sum_{i=1}^{B}\sum_{j=1}^{G}
\frac{1}{\operatorname{len}(y^{(i,j)})}
\frac{r(y^{(i,j)}\mid x^{(i)})-\mu_i}{\mathrm{std}_i}
\sum_{t=1}^{\operatorname{len}(y^{(i,j)})}
\frac{
\nabla_\theta\pi_\theta(y_t\mid x,y_{<t})
}{
\pi_0(y_t\mid x,y_{<t})
}
\\[8pt]
&= \frac{1}{BG}
\sum_{i=1}^{B}\sum_{j=1}^{G}
\frac{1}{\operatorname{len}(y^{(i,j)})}
\frac{r(y^{(i,j)}\mid x^{(i)})-\mu_i}{\mathrm{std}_i}
\sum_{t=1}^{\operatorname{len}(y^{(i,j)})}
\frac{
\pi_\theta(y_t\mid x,y_{<t})
\nabla_\theta\log\pi_\theta(y_t\mid x,y_{<t})
}{
\pi_0(y_t\mid x,y_{<t})
}
\\[8pt]
&= \frac{1}{BG}
\sum_{i=1}^{B}\sum_{j=1}^{G}
\frac{1}{\operatorname{len}(y^{(i,j)})}
\frac{r(y^{(i,j)}\mid x^{(i)})-\mu_i}{\mathrm{std}_i}
\sum_{t=1}^{\operatorname{len}(y^{(i,j)})}
\frac{
\pi_\theta(y_t\mid x,y_{<t})
}{
\pi_0(y_t\mid x,y_{<t})
}
\nabla_\theta\log\pi_\theta(y_t\mid x,y_{<t})
\\[8pt]
&= \hat{g}
\end{aligned}
$$

现在我们简写：

令：

$$
\begin{aligned}
A^{(i,j)}
&= \frac{r(y^{(i,j)}\mid x^{(i)})-\mu_i}{\mathrm{std}_i}, \\
w_t^{(i,j)}
&= \frac{\pi_\theta(y_t\mid x,y_{<t})}
{\pi_0(y_t\mid x,y_{<t})}
\end{aligned}
$$

所以：

$$
J_\theta^{\mathrm{GRPO\text{-}off\text{-}policy\text{-}noclip}}
=
\frac{1}{BG}
\sum_{i=1}^{B}\sum_{j=1}^{G}
\frac{1}{\operatorname{len}(y^{(i,j)})}
\sum_{t=1}^{\operatorname{len}(y^{(i,j)})}
A^{(i,j)}w_t^{(i,j)}
$$

现在对该目标添加裁剪项：

$$
\begin{aligned}
&J_\theta^{\mathrm{GRPO\text{-}off\text{-}policy\text{-}clip}} 
= \frac{1}{BG}
\sum_{i=1}^{B}\sum_{j=1}^{G}
\frac{1}{\operatorname{len}(y^{(i,j)})}
\sum_{t=1}^{\operatorname{len}(y^{(i,j)})}
\min\Big(
A^{(i,j)}w_t^{(i,j)}, 
A^{(i,j)}
\operatorname{clip}\big(
w_t^{(i,j)},[1-\epsilon,1+\epsilon]
\big)
\Big)
\end{aligned}
$$

其中：

$$
\begin{aligned}
\operatorname{clip}
\left(w_t^{(i,j)},[1-\epsilon,1+\epsilon]\right)= \min\left(
\max\left(w_t^{(i,j)},1-\epsilon\right),
1+\epsilon
\right)
\end{aligned}
$$

更直观的表示就是：

$$
\begin{aligned}
J_\theta^{\mathrm{GRPO\text{-}off\text{-}policy\text{-}clip}} &= \frac{1}{BG}
\sum_{i=1}^{B}\sum_{j=1}^{G}
\frac{1}{\operatorname{len}(y^{(i,j)})}
\sum_{t=1}^{\operatorname{len}(y^{(i,j)})}
\begin{cases}
\min(w_t^{(i,j)},1+\epsilon)A^{(i,j)},
& A^{(i,j)}\geq 0, \\[6pt]
\max(w_t^{(i,j)},1-\epsilon)A^{(i,j)},
& A^{(i,j)}<0
\end{cases}
\end{aligned}
$$

解释一下为什么这样设置：因为 $w=\frac{\pi_\theta}{\pi_0}$，当 $A \geq 0$ 时，代表我们希望增加该 token 被生成的概率，此时若 $\pi_\theta \geq \pi_0$，那么代表已经将该 token 的概率增加了，所以我们设置上限 $1+\epsilon$，当增加概率超过 $1+\epsilon$ 时，我们就认为 $w$ 朝着 Advantage 希望的方向变化过多，停止提供梯度。同理，当 $A < 0$ 时，代表我们希望降低该 token 被生成的概率，此时若 $\pi_\theta < \pi_0$，那么代表已经将该 token 的概率降低了，且 $\pi_\theta$ 越降低，$w$ 越小 所以我们设置 $1-\epsilon$，当 $w$ 小于 $1-\epsilon$ 时，我们就认为 $w$ 朝着降低的方向变化过多，停止提供梯度。

对 $J_\theta^{GRPO-off-policy-clip}$ 求导得到：

当 $A^{(i,j)}\geq 0$ 时：

$$
\nabla_\theta J_\theta^{
\mathrm{GRPO\text{-}off\text{-}policy\text{-}clip}
}
=
\begin{cases}
\frac{1}{BG}
\sum_{i=1}^{B}\sum_{j=1}^{G}
\frac{1}{\operatorname{len}(y^{(i,j)})}
\sum_{t=1}^{\operatorname{len}(y^{(i,j)})}
w_t^{(i,j)}A^{(i,j)}
\nabla_\theta
\log\pi_\theta(y_t\mid x,y_{<t}),
& w_t^{(i,j)}<1+\epsilon,\\[6pt]
0,
& w_t^{(i,j)}>1+\epsilon
\end{cases}
$$

当 $A^{(i,j)}<0$ 时：

$$
\nabla_\theta J_\theta^{
\mathrm{GRPO\text{-}off\text{-}policy\text{-}clip}
}
=
\begin{cases}
\frac{1}{BG}
\sum_{i=1}^{B}\sum_{j=1}^{G}
\frac{1}{\operatorname{len}(y^{(i,j)})}
\sum_{t=1}^{\operatorname{len}(y^{(i,j)})}
w_t^{(i,j)}A^{(i,j)}
\nabla_\theta
\log\pi_\theta(y_t\mid x,y_{<t}),
& w_t^{(i,j)}>1-\epsilon,\\[6pt]
0,
& w_t^{(i,j)}<1-\epsilon
\end{cases}
$$

所以可以统一写为：

$$
\begin{aligned}
\nabla_\theta J_\theta^{
\mathrm{GRPO\text{-}off\text{-}policy\text{-}clip}
}=
\frac{1}{BG}
\sum_{i=1}^{B}\sum_{j=1}^{G}
\frac{1}{\operatorname{len}(y^{(i,j)})}
\sum_{t=1}^{\operatorname{len}(y^{(i,j)})}
\operatorname{mask}_t^{(i,j)}
w_t^{(i,j)}A^{(i,j)}
\nabla_\theta
\log\pi_\theta(y_t\mid x,y_{<t})
\end{aligned}
$$

其中：

$$
\operatorname{mask}_t^{(i,j)}
=
\begin{cases}
\mathbb{1}[w_t^{(i,j)}<1+\epsilon],
& A^{(i,j)}\geq 0,\\[6pt]
\mathbb{1}[w_t^{(i,j)}>1-\epsilon],
& A^{(i,j)}<0
\end{cases}
$$

这里 $\mathbb1[\cdot]$ 是指示函数，条件成立为 1，否则为 0。恰好等于裁剪边界的位置存在不可导点，实际自动微分框架会采用相应分支的梯度约定。

![](\images\CS336_Assignment5\img48.png)

![](\images\CS336_Assignment5\img49.png)

![](\images\CS336_Assignment5\img50.png)

noclip 和 grpo 测试通过，失败是 gspo

下面补充 GSPO 代码后，测试通过

![](\images\CS336_Assignment5\img51.png)

## GSPO
为了改进 GRPO 中存在的偏差（前缀偏差和后缀偏差），提出了GSPO。
GSPO 不采用 GRPO 中提出的 token-level，而是重新回到 sequence-level，但为了避免 ratio 相乘带来的高方差，使用了几何平均：
$$
s^{(i,j)}=(\prod_{t=1}^{len(y^{(i,j)})}\frac{\pi_\theta(y_t|x,y_{<t})}{\pi_0(y_t|x,y_{<t})})^{\frac{1}{len(y^{(i,j)})}}
$$
但因为 GSPO 的几何平均不是严格的 sequence-level importance sampling 校正，所以它降低方差的同时也还是会引入 bias。

先在无 clipping 时，考虑 GSPO 的公式：
$$
\begin{aligned}
J_\theta^{GSPO-off-policy-noclip} &= \frac{1}{BG} \sum_{i=1}^{B}\sum_{j=1}^{G} A^{(i,j)}s^{(i,j)}
\\\\&= \frac{1}{BG} \sum_{i=1}^{B}\sum_{j=1}^{G} \frac{r(y^{(i, j)}|x^{(i)})-\mu_i}{std_i}(\prod_{t=1}^{len(y^{(i,j)})}\frac{\pi_\theta(y_t|x,y_{<t})}{\pi_0(y_t|x,y_{<t})})^{\frac{1}{len(y^{(i,j)})}}
\end{aligned}
$$
数学推导为什么这个公式可以：

首先，对序列级重要性权重取对数：

$$
\begin{aligned}
\log s^{(i,j)}
&= \frac{1}{\operatorname{len}(y^{(i,j)})}
\log\left(
\prod_{t=1}^{\operatorname{len}(y^{(i,j)})}
\frac{\pi_\theta(y_t\mid x,y_{<t})}
{\pi_0(y_t\mid x,y_{<t})}
\right) \\
&= \frac{1}{\operatorname{len}(y^{(i,j)})}
\sum_{t=1}^{\operatorname{len}(y^{(i,j)})}
\left(
\log\pi_\theta(y_t\mid x,y_{<t})
-
\log\pi_0(y_t\mid x,y_{<t})
\right)
\end{aligned}
$$

然后，对 $\theta$ 求梯度：

$$
\nabla_\theta\log s^{(i,j)}
=
\frac{1}{\operatorname{len}(y^{(i,j)})}
\sum_{t=1}^{\operatorname{len}(y^{(i,j)})}
\nabla_\theta\log\pi_\theta(y_t\mid x,y_{<t})
$$

利用 $\nabla_\theta s=s\nabla_\theta\log s$，得到：

$$
\begin{aligned}
\nabla_\theta s^{(i,j)}
&= s^{(i,j)}\nabla_\theta\log s^{(i,j)} \\
&= s^{(i,j)}
\frac{1}{\operatorname{len}(y^{(i,j)})}
\sum_{t=1}^{\operatorname{len}(y^{(i,j)})}
\nabla_\theta\log\pi_\theta(y_t\mid x,y_{<t})
\end{aligned}
$$

因此，GSPO 的梯度可以写为：

$$
\begin{aligned}
&\nabla_\theta J_\theta^{
\mathrm{GSPO\text{-}off\text{-}policy\text{-}noclip}
} \\
&= \frac{1}{BG}
\sum_{i=1}^{B}\sum_{j=1}^{G}
\frac{r(y^{(i,j)}\mid x^{(i)})-\mu_i}
{\mathrm{std}_i}
\nabla_\theta
\left(
\prod_{t=1}^{\operatorname{len}(y^{(i,j)})}
\frac{\pi_\theta(y_t\mid x,y_{<t})}
{\pi_0(y_t\mid x,y_{<t})}
\right)^{
\frac{1}{\operatorname{len}(y^{(i,j)})}
}
\\[8pt]
&= \frac{1}{BG}
\sum_{i=1}^{B}\sum_{j=1}^{G}
\frac{r(y^{(i,j)}\mid x^{(i)})-\mu_i}
{\mathrm{std}_i}
s^{(i,j)}
\frac{1}{\operatorname{len}(y^{(i,j)})}
\sum_{t=1}^{\operatorname{len}(y^{(i,j)})}
\nabla_\theta\log\pi_\theta(y_t\mid x,y_{<t})
\end{aligned}
$$

与我们之前的 sequence-level reweighting 基本一致：

$$
\mathbb{E}_{x\sim\rho}
\mathbb{E}_{y\sim\pi_0}
\Bigg[
r(y\mid x)
\left(
\prod_{t=1}^{T}
\frac{\pi_\theta(y_t\mid x,y_{<t})}
{\pi_0(y_t\mid x,y_{<t})}
\right) 
\sum_{t=1}^{T}
\nabla_\theta\log\pi_\theta(y_t\mid x,y_{<t})
\Bigg]
$$

由上述推导可知，GSPO 的梯度与标准 sequence-level importance-reweighted policy gradient 具有相似的结构：两者都使用一个共享的序列级 importance weight，对整条回答的 token-level log-probability gradients 进行统一加权。但两者并不完全等价。
标准 sequence-level importance sampling 使用完整序列的概率比：
$$
\prod_{t=1}^{L}w_t  
$$
这种权重能够校正新旧策略之间的序列分布差异，但由于多个 token 的概率比连续相乘，在长序列中可能产生较大的方差。
GSPO 则将完整序列权重替换为其几何平均：
$$
\prod_{t=1}^{L}w_t^{\frac{1}{L}}  
$$
从而缓解序列长度导致的权重尺度问题。同时，由于几何平均中的指数 $\frac{1}{L}$，对 GSPO objective 求梯度时，会自然出现序列长度归一化因子：
$$
\frac1{BG}\sum_{i,j}  
A^{(i,j)}s^{(i,j)}  
\frac1{L_{ij}}  
\sum_t\nabla_\theta\log\pi_\theta(y_t)  
$$
因此，GSPO 可以理解为一种**使用几何平均序列权重的、带长度归一化的 off-policy policy gradient surrogate**。它保留了序列级共享权重的特点，但不再是严格无偏的 importance sampling 校正，而是通过引入一定的偏差来降低方差、改善训练稳定性。

下面给出经过 clipping 的 GSPO，与 GRPO clipping 规则一致：
$$
J_\theta^{GSPO-off-policy-clip} = \frac{1}{BG} \sum_{i=1}^{B}\sum_{j=1}^{G} min(A^{(i,j)}s^{(i,j)}, A^{(i,j)}clip(s^{(i,j)},[1-\epsilon,1+\epsilon]))
$$

具体展开不再赘述。

![](\images\CS336_Assignment5\img52.png)

![](\images\CS336_Assignment5\img53.png)

![](\images\CS336_Assignment5\img54.png)
