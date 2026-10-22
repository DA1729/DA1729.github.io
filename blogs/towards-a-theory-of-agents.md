---
title: "Towards a Theory of Agents"
date: "2026-10-22"
description: ""
---

# Towards a Theory of Agents

Say we have two systems $G$ (2 agents) and $H$ (3 agents), and we want to 'combine' them. For now, let's work with what we could intuitively infer from the term 'combine,' since I have no idea what 'combine' could formally mean here. But still, we can go ahead and ask more intuitively formed, interesting questions about it. Here is an interesting one:

> Can we find compact mathematical descriptions of $G$ and $H$ that capture everything we need to study the combined system $G + H$?

But...

## What is an agent?

Well, here is a neat part (in the wrong way, perhaps). There is no single formal definition of an agent. In particular, there is no single definition of an agent across game theory, decision theory, and now AI. But I can present an intuitive definition:

> An agent is an entity that can make decisions and act within some environment.

What exactly are decisions? What is an action? Again, their meanings change as we move across fields. Now, if you really think about it, or maybe think hard and try to break my following assertion, you should be able to see that I can describe an agent in the context of the entire system, rather than how it functions internally, using the following three 'quantities':

- Information
- Action
- Decision.

I am deliberately not giving even rough, single-line definitions for them, because it's impossible for what I am trying to describe here. For instance, the information a human being has exists in many forms: through senses, memory, logical inferences, etc. Similarly, decisions and actions have their own nuances and troubles when we try to classify each even in its own right. For instance, I am writing this essay: the decision is that I choose to create this, and the action is that I am typing. I am also breathing as I type, so is that my decision or action? Well, I choose to breathe because I have the information that I don't know what happens after death; therefore, in the best interest of my consciousness, I decide to breathe and take the necessary actions. Note that it doesn't matter whether my conscious mind is taking the action/decision or my subconscious. The important thing is that we can, in the end, observe an agent entirely using the three terms above.

We do have enough to start off with some schematic symbols and move towards apparent formalism, but let's not do that for now, and just try some introspection and perhaps some formal dealings with other fields.

Let's start with a formal sketch of a field whose name should become apparent in a while.

Let agent $i$ have an observation space $\mathcal{O}_i$ and an action space $\mathcal{A}_i$. A deterministic decision rule is a function:

$$
\pi_i : \mathcal{O}_i \rightarrow \mathcal{A}_i.
$$

Simply put, given an observation, the agent will take a certain well-defined, deterministic action. We can easily increase the complexity by letting the agent take 'randomized' decisions and write $\pi_i: \mathcal{O}_i \rightarrow \Delta(\mathcal{A}_i)$, where $\Delta(\mathcal{A}_i)$ denotes the set of probability distributions over $\mathcal{A}_i$.

Now suppose we have $n$ agents. We can write their collection as $N = \\{1, 2, \cdots, n\\}$. Each agent $i$ has its own action space $\mathcal{A}_i$.

The set of all possible combinations of actions is

$$
\mathcal{A} = \mathcal{A}_1 \times \cdots \times \mathcal{A}_n.
$$

An element $(a_1, \cdots, a_n) \in \mathcal{A}$ is called an action profile.

This simply specifies what every agent does. Just putting it out there, and please note that for now, there is no need to fixate on certain assertions and descriptions of any term. I am just glossing over stuff as intuitively (with just enough rigour) as possible. Yeah, certain actions may influence other actions in complex ways.

We also define utility functions $u_i$ that take an action profile as input. Think of it this way: the higher the value of this function for a certain agent, the more preferred the outcome is for that agent.

Now with this, we could define a mathematical object:

$$
G = (N, (\mathcal{A}_i)_{i\in N}, (u_i)_{i\in N}).
$$

This is called a **normal-form game**, which is a foundational model in game theory.

---

Let's end it here.
