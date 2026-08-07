---
title: "When Secure Messages Still Leak Information"
date: 2026-08-07
tags:
    - AI Safety
    - Information Theory
    - Multi-Agent Systems
    - Security
---

> *This post is based on general ideas I have been thinking about during my research internship at COSIC, KU Leuven. The actual research is ongoing, so I am deliberately avoiding unpublished results, concrete constructions, experimental numbers, and theorem statements.*

When we talk about securing communication between software systems, we usually reach for familiar tools: authentication, encryption, access control, signatures, capability systems, and so on.

These tools answer questions like:

- Who sent this message?
- Was it modified in transit?
- Was the sender authorized to send it?
- Can an outsider read it?

For ordinary distributed systems, these are already difficult and important problems.

But LLM-based multi-agent systems introduce a slightly stranger question:

> **Even if every message is legitimate, authenticated, and syntactically harmless, what information is actually flowing through the system?**

This distinction between **secure communication** and **secure information flow** is what I have been spending a lot of time thinking about recently.

---

## 1. The Basic Setup: An Agent Knows Something

Consider two agents.

The first agent, which I will call the **sender**, has access to some private state

$$S.$$

This could represent anything:

- private user information,
- a hidden instruction,
- privileged tool output,
- a secret system state,
- some task-specific record.

The sender generates a message

$$Z$$

which is passed to another agent.

At the most abstract level, the sender is just a probabilistic channel:

$$P(Z \mid S).$$

If the distribution of the message changes depending on the private state, then observing the message can reveal something about that state.

Information theory gives us a natural way of expressing this:

$$I(S;Z).$$

This is the **mutual information** between the private state and the emitted message.

If

$$I(S;Z)=0,$$

then seeing the message tells us nothing about the private state.

If

$$I(S;Z)>0,$$

then some information is flowing.

This seems straightforward.

The interesting part begins when we add a monitor.

---

## 2. What Does the Monitor Actually See?

Suppose a security monitor watches every message emitted by the agent.

It may know what normal behaviour looks like. Maybe we collected thousands of benign conversations and learned an expected message distribution

$$P_0(Z).$$

A very tempting security condition would then be:

> Ensure that the deployed agent continues to produce messages following the expected distribution.

That is,

$$Q(Z) \approx P_0(Z).$$

This sounds quite strong. If the output distribution has not changed, perhaps nothing suspicious is happening.

But there is an important difference between

$$P(Z)$$

and

$$P(Z \mid S).$$

The first tells us how messages look **in aggregate**.

The second tells us how messages depend on the agent's **private state**.

Those are very different objects.

---

## 3. Same Output Distribution, Different Dependence

Here is a tiny toy example.

Suppose the private state is a fair bit:

$$S \in \{0,1\}.$$

The sender is also supposed to output a fair bit:

$$Z \in \{0,1\}.$$

Imagine the honest sender chooses $Z$ independently:

$$P(Z=0)=P(Z=1)=\frac12.$$

Then

$$I(S;Z)=0.$$

Now consider a different sender:

$$Z=S.$$

The output distribution is still

$$P(Z=0)=P(Z=1)=\frac12.$$

So somebody looking only at the frequency of zeros and ones sees **exactly the same distribution**.

But now

$$I(S;Z)=1 \text{ bit}.$$

The entire private state is being transmitted.

Nothing changed in the marginal distribution of the output.

What changed was the **coupling** between $S$ and $Z$.

This is a very simple example, but I find the underlying idea important:

> **A distribution can look perfectly ordinary while the dependence structure underneath it changes completely.**

---

## 4. Couplings: The Hidden Degree of Freedom

Suppose we know the distribution of the private state

$$P(S)$$

and the distribution of the message

$$P(Z).$$

These two distributions do **not** uniquely determine the joint distribution

$$P(S,Z).$$

There can be many possible joint distributions having exactly the same marginals.

Mathematically, these are called **couplings**.

For example, two binary variables can have identical marginals while being:

- independent,
- perfectly correlated,
- perfectly anti-correlated,
- or somewhere in between.

The marginals tell us what each variable looks like individually.

The coupling tells us **how they move together**.

And information leakage lives in that relationship.

This is one reason information theory has turned out to be such a natural language for thinking about agent security.

The question is not merely:

> "Does the output look normal?"

but rather:

> "How much dependence between hidden state and observable output is compatible with what the monitor can see?"

That is a much more structural question.

---

## 5. Why LLMs Make This Particularly Weird

For a traditional network protocol, a message usually has a relatively explicit semantic role.

A field might contain:

```text
user_id = 4127
```

or

```text
permission = READ_ONLY
```

With language models, the communication space is enormous.

Consider two perfectly ordinary sentences:

> "The application seems broadly acceptable, although I would double-check the financial history."

and

> "The application looks acceptable overall; the financial history may deserve another review."

Semantically, these may accomplish essentially the same task.

But from the perspective of a probabilistic language model, there were many choices:

- word choice,
- syntax,
- punctuation,
- verbosity,
- ordering,
- formatting,
- token selection.

Every degree of freedom potentially creates room for information to be encoded.

This does **not** mean that every LLM message secretly contains a covert channel.

It means that natural language has a large amount of conditional variability, and security analysis should be careful about what exactly is being constrained.

---

## 6. Predictability and Capacity

A useful intuition comes from entropy.

Suppose an agent has many equally acceptable ways to phrase the same answer.

Then its output has relatively high conditional entropy:

$$H(Z \mid \text{task}).$$

There is flexibility.

If instead the system forces the agent to emit something extremely rigid, such as

```json
{"decision":"approve"}
```

then much of that flexibility disappears.

Intuitively, fewer degrees of freedom means fewer places in which unrelated information could hide.

This gives a nice connection between **predictability** and **information-flow capacity**.

A system that says

> "communicate anything you want, as long as the meaning is correct"

leaves much more freedom than one that says

> "emit exactly one value from this small canonical schema."

Of course, strict formatting is not automatically a complete security solution.

But it illustrates the deeper principle:

> **Communication freedom is itself a security resource.**

---

## 7. Authentication Does Not Solve This

This is also why cryptographic security and information-flow security should not be conflated.

Suppose two agents communicate over:

- TLS,
- mutually authenticated identities,
- signed messages,
- tightly scoped API tokens,
- perfectly implemented access control.

Excellent.

We have established that the right agent sent an untampered message through an authorized channel.

But none of those guarantees tells us:

$$I(S;Z)$$

for some sensitive state $S$.

Cryptography protects the **channel from outsiders**.

Information-flow analysis asks what the **legitimate participants themselves are communicating**.

These are complementary security layers, not substitutes for each other.

My background is mostly in cryptography, so this distinction took me a little while to appreciate properly.

In cryptanalysis, we often ask:

> What hidden structure remains even after the obvious attack surface has been removed?

The same instinct turns out to be useful here.

---

## 8. Autoregressive Models Complicate Everything Further

For an LLM, the message $Z$ is not generated all at once.

It is produced token by token:

$$Z=(Z_1,Z_2,\dots,Z_T).$$

The model samples according to

$$P(Z_t \mid Z_{<t}, S, C),$$

where $C$ is the public context.

This means that after every emitted token, the observer's knowledge changes.

For example, if the first few tokens make one private state much more likely than another, then the relevant uncertainty before generating the next token is no longer the original prior.

Conceptually, we should think about a sequence of posterior updates:

$$P(S) \rightarrow P(S\mid Z_1) \rightarrow P(S\mid Z_1,Z_2) \rightarrow \cdots$$

This makes language-model communication feel much closer to a sequential probabilistic process than to a traditional packet protocol.

And once you start thinking this way, concepts from Bayesian inference, information theory, stochastic processes, and coding theory become very tempting.

---

## 9. The Connection to Multi-Agent Systems

Now imagine not two agents but an entire graph:

$$A_1 \rightarrow A_2 \rightarrow A_3 \rightarrow \cdots$$

Different agents may see different pieces of private information.

Some have tool access.

Some make decisions.

Some summarize information for others.

Some act as monitors.

This creates a distributed information-flow problem.

A message from $A_1$ might reveal almost nothing by itself but change how $A_2$ behaves later.

Or multiple individually harmless messages may jointly reveal something.

Or an intermediate agent may transform information into a representation that a downstream agent can exploit even though a human monitor sees nothing suspicious.

This is one reason I think multi-agent AI security is interesting.

The relevant object is no longer just the model.

It is the **joint stochastic system formed by models, private states, tools, communication channels, and observers**.

---

## 10. A Different Way to Think About Monitoring

A lot of security thinking begins with:

1. enumerate attacks,
2. detect them,
3. block them.

That is useful, but generative systems make exhaustive enumeration difficult.

An alternative style of question is:

> What information does the observer have, and what guarantees are even possible from that information?

This turns monitoring into an inference problem.

If two different internal behaviours induce exactly the same observable distribution, then no monitor restricted to those observations can distinguish them perfectly.

That is not a weakness of the classifier.

It is a limitation of the **observation model**.

I find this perspective useful because it forces us to specify what the monitor actually knows before claiming that something is detectable.

---

## 11. Why I Like the Information-Theoretic View

The thing I like most about information theory is that it removes a lot of semantic ambiguity.

Instead of beginning with:

> "Does this sentence look suspicious?"

we can begin with:

- What are the random variables?
- What is hidden?
- What is observable?
- What distribution is known?
- What dependence is allowed?
- What quantity measures information transmission?
- Which constraints reduce that transmission?

That does not magically solve AI security.

But it gives us a much cleaner mathematical object to argue about.

And perhaps more importantly, it tells us when a monitoring strategy is trying to infer something that simply is not determined by its observations.

---

## 12. Where This Seems Useful

Even at a very general level, I think this viewpoint connects to several problems:

### Synthetic-data training

If one model generates training data for another, what information about the teacher survives through apparently neutral outputs?

### Model distillation

Can traits or preferences propagate even when they are not explicitly present in the semantic content?

### Agent communication

Can private information influence downstream behaviour through channels that look ordinary to a transcript monitor?

### Interpretability

Can internal representations reveal information flow that is invisible from behaviour alone?

### Security policy

Should we constrain only *what* agents are allowed to say, or also how much freedom they have in representing the same task-relevant information?

All of these feel like variations of the same underlying question:

> **What can be transmitted through a constrained stochastic channel?**

That is an old information-theory question appearing inside a very new kind of system.

---

## 13. The Bigger Lesson

The main conceptual lesson I have taken from working on this problem is fairly simple:

> **Observing the output distribution is not the same as observing the information flow that generated it.**

A system can preserve many visible properties while changing hidden dependencies.

This phenomenon is not unique to language models. It is basic probability theory.

But LLMs make it operationally important because their output spaces are enormous, their communication is stochastic, and their semantics are flexible.

That makes the boundary between

$$\text{normal variation}$$

and

$$\text{information-bearing variation}$$

a surprisingly deep security problem.

I came into this area mostly from algebra and cryptography.

Somehow I ended up thinking about entropy, conditional distributions, Bayesian observers, and stochastic channels.

Not the worst detour.

peace. da1729

---

### References / Further Reading

1. **Shannon, C. E. (1948).** *A Mathematical Theory of Communication.* Bell System Technical Journal.
2. **Cover, T. M., & Thomas, J. A.** *Elements of Information Theory.* Wiley.
3. **Schroeder de Witt, C. et al. (2023).** *Perfectly Secure Steganography Using Minimum Entropy Coupling.* ICLR.
4. **Motwani, S. et al. (2024).** *Secret Collusion among AI Agents: Multi-Agent Deception via Steganography.* NeurIPS.
5. **Lampson, B. W. (1973).** *A Note on the Confinement Problem.* Communications of the ACM.
