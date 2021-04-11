---
title: "Deep RL 3 Intro to RL"
date: 2021-04-10
categories:
  - DeepRL
tags:
  - RL
  - notes
---

The is an introduction to reinforcement learning, including core concepts, the general goal, the general framework, introduction and comparison of different types of approaches. This section contains nighty percent of the materials that are going to be covered by this course.

## 1 Notations and terminologies
We have introduced some of the notations in [lecture 2](https://jasonppy.github.io/deeprl/deeprl-2-imitation/). Here we give a more comprehansive introduction of the core concepts in reinforcement learning。

Reinforcement learning centers around the Markov decision processes $$\mathcal{M} = \{ \mathcal{S}, \mathcal{A}, \mathcal{T}, r \}$$ with the four terms defined as:

$$\begin{eqnarray*}
&&\mathcal{S} \text{ is state space, state } s \in \mathcal{S} \text{ can be discrete or continuous} \\
&&\mathcal{A} \text{ is action space, state } a \in \mathcal{A} \text{ can be discrete or continuous} \\
&&\mathcal{T} \text{ is the transition operator } \\
&&r: \mathcal{S}\times\mathcal{A} \to \mathbb{R} \text{ is the reward function, specify how good the state and action is}
\end{eqnarray*}$$

More on the transition operator $$\mathcal{T}$$, denote
$$\begin{eqnarray*}
&&\mu_{j} := p(s = j) \\
&&\xi_{k} := p(a = k) \\
&&\mathcal{T}_{i,j,k} := p(s' = i\mid s=j, a=k)\\
&&\text{we have } \mu_{i} = \sum_{j,k}\mathcal{T}_{i,j,k}\mu_{j}\xi_{k} \\
\end{eqnarray*}$$

For simplicity, we illustrate the transition operator using discrete state and action space. I didn't find any illustration for the continuous case. Actually trainsition operator is rarely used in this course, most of the time we just use the transition probability $$p(s'\mid s,a)$$ (more often writen as $$p(s_{t+1}\mid s_t, a_t)$$). To avoid confusion, the community also refer to the transition probability as *model*, *dynamics*, or *transition dynamics*.





