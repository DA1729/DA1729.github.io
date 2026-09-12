---
title: "On Origins of Understanding"
date: "2026-09-12"
description: ""
---

# On Origins of Understanding

2 days back, I wrote a short note -- [A Note on Machine Generated Proofs](https://da1729.github.io/post.html?post=a-note-on-machine-generated-proofs&kind=blog). Today, I sat down and developed something seeded by the core of that note.

I am at this moment (perhaps this week) feeling more pushed to think more about this, following an internal OpenAI model finding a solution to the Navier-Stokes equation. I am in no way in an existential crisis as a person on a road to becoming a mathematician, and the following reasonings should explain why.

I am really bothered by the fact that everyone seems to be in an existential crisis or feeling vulnerable, based on the question, "Can AI prove difficult things better than humans?" The question, as with most in this regime, is highly subjective. We must settle on a few things we can all reasonably agree on (perhaps axioms) and then reason further to better understand AI developments and, more importantly, how slowly our society's structure could transform.

---
## Proof Production and Mathematical Superiority

This argument is short and clean, and it's better settled early on before I get to the core of my theory. It follows nicely from incompleteness and undecidability.

For a sufficiently expressive, consistent, effectively axiomatized theory $T$, there are statements that $T$ cannot decide. There is also, in general, no algorithm that will always terminate and correctly decide whether an arbitrary statement has a proof in $T$.

If a statement has a proof, proofs can in principle be enumerated until one is found. If it has no proof, simply continuing the search is an absurd venture.

With this, we get a distinction between the following two: $$\text{being an extraordinary proof generator}$$ $$\text{and}$$ $$\text{having unrestricted access to mathematical truth}.$$

Note that I am not claiming superiority of humans over AI, and vice versa; this argument applies to a human mathematician as well. It just settles that proof-generating abilities are an irrelevant metric when asked about mathematical superiority.

---

## The Clerk Thought Experiment

Suppose an AI system produces a proof of some highly nontrivial theorem. Perhaps, there is no need to suppose it now.

The entire computation performed by the model running on a GPU could be specified finitely. This includes things like: 
- the architecture;
- exact model weights;
- inputs;
- intermediate states;
- finite-precision arithmetic;
- external tool outputs;
- multi-agent interactions; etc.

Now imagine that, instead of letting these steps run on a computer, a human clerk is given unlimited paper, a pen, time, and the exact instructions for carrying out every operation. The clerk then produces the computation manually. After an absurdly long time, the same proof appears on the paper.

Schematically: $$C + A + P \longrightarrow Q,$$ where:

- $C$ is the clerk,
- $A$ is the algorithm (includes every spec),
- $P$ is the external physical tools, ergo, pen and paper,
- $Q$ is the proof.

The procedure may be unimaginably long, but this is irrelevant for the inferences out of it.

Now, the resulting proof is identical and establishes one kind of equivalence: $$\text{same specified computation} \Rightarrow \text{same mathematical output}.$$

It does not yet establish any equivalence between still-subjectively-assumed objects—understanding, consciousness, intuition, reasoning, etc. The clerk could very well have produced the proof without understanding it at all. I will strengthen this argument in the following section.

---

## What About Accidental Understanding?

Continuing from the previous section, suppose the clerk does not speak English. Furthermore, their own culture has a completely different symbolic-mathematical tradition. In other words, their symbols for numbers, operations and everything are different from the standard symbols of modern mathematics. With this, they may not know whether the symbol represents addition, subtraction, integration, a matrix, or something else. Now, the algorithm is described to them entirely in their native language. An even stronger case would be losing semantic instructions altogether. Therefore, instead of: $$\text{Add the values in these cells.}$$ we say: $$\text{if the current state is symbol }q_7\text{ and the next mark is }\Delta\cdots.$$

Now, it still does not justify the claim that it's impossible for the clerk—being aware of some human language—to figure out the meaning behind these symbols. It's still possible. But now there is a choice for them: whether to decode. Either way, we shall get the same proof from the steps executed by the clerk, and we can safely assume that the clerk does not choose to perform the translation and decoding.

---
## On Composition of Understanding

With the entire setup of my thought experiment and my rolling inferences so far, it seems like I am about to assert what follows. If we observe the system—clerk's mechanical capacity + paper + pen + instructions—and further, saying that each component does not possess the "mathematical understanding," thus the system composed by them also does not possess the same.

It sounds clean, but here, I am making an implicit assumption that understanding composes. It's a type of assumption, which cannot be taken for granted. A really nice functionalist's argument against composability would be, on observing individual neurons or brain cells, it would not make sense to say that they possess understanding, yet we would be more than comfortable to say that human being as a whole can understand and does understand some things.

I constructed another argument against such conclusion (inspired from Searle's Chinese Room thought experiment), of the clerk + paper + pen + instructions system not possessing the mathematical understanding. Let's say that I put the clerk in a box, in which people can send in their texts (prompts), and I go ahead and ask a series of questions, maybe related to mathematics, like: 

> Does theorem $T$ survive if assumption $A$ is weakened?

It gives a useful answer. The question's followed by,

> What's the geometric intuition?

And yet again, it provides me some useful answer.

Given that I do not know whether inside, is a human or a machine, from my perspective, whatever's inside, possesses a mathematical understanding.

---

## Clerk Thought Experiment in Interstellar Space

Now, that I have modeled the thought experiment, and addressed, I think are the three fundamental and critical arguments against concluding that the seemingly mechanical system does not possess mathematical understanding, I can go ahead and start extracting possible inferences out of this premise and later try to develop them in a formal manner.

First, imagine that the clerk, who knows a native human language, and further, also aware of the mechanical instructions given to them. Now a mapping may very well exist, between their traditional mathematics and the mathematics conveyed to them for the algorithm, my entire argument depends on the assumption that the clerk has a choice between using the mapping or not. And dealing with the case that they choose not to do that.

Now, the last objection from the previous section would suggest that understanding is an observational property, and it depends on the observer whether the box understands mathematics or not. And I think it could become a valid and I would really like to hold on to this thought and pursue it later, but what follows does not encourage keeping this one of the primary views.

The reason why I am not able to carry understanding being an observational quantity further is as follows. One can reason the fact living in human society, we are always observed, even if we are alone in our room (we are being observed on social media). So we send our trained clerk to interstellar space, completely isolated with absolutely 0 communication between any possible observer. The assumption I made earlier that clerk is not entirely mechanical and knows at least one natural human language in which they can keep thinking in. When that clerk is left alone, being able to think in their native language has a notion that they understand some things and also maths in whatever symbolic system they have studied that in. The clerk still knows the set of instructions provided to them, and has a notion of understanding that they understand how to execute them on a pen and a paper (which we can assume they still have). Assuming that clerk never chooses to utilize the mapping between the two symbolic systems and even execute the algorithm, the system is still there, and also an observer (one could maybe argue that clerk is not a valid observer), yet the clerk has a notion of understanding for themselves and 0 notion of understanding for the mechanical part of their brain + instructions + paper + pen.

With the argument above, what I am really trying to establish is that understanding is not purely an observational fact, but rather a partially observed fact. The clerk was once observed by humans who taught them their native language and the understanding carried on later when they were isolated. But it doesn't look like the same thing holds for the "artificial intelligent" system we constructed.

---
## The Causal Chain

There is no guarantee that the instructions given to the clerk were produced by some fundamentally superior entity. There could be some other similar clerk, given different sets of instructions and prompted to produce a set of instructions for the next one.

We can write this chain schematically as: $$C_n \rightarrow C_{n-1} \rightarrow \cdots \rightarrow C_1 \rightarrow Q,$$
here, $Q$ is the eventual mathematical output. The fact that one system supplies instructions to another does not imply that mathematical understanding existed in the supplier. The causal origin of instruction and the semantic understanding of that instruction are separate questions. 

This removes our temptation to locate understanding merely by walking backward through the chain until one finds whoever produced the previous procedure. But the chain cannot remain completely ungrounded. The instructions themselves consist of symbols. The proof consists of symbols. The prompts, tables, weights, encodings, etc all belong to some symbolic system.

Those symbols came from somewhere, especially given that humans have existed for only a finite amount of time, we cannot extend the chain backwards infinitely. The causal model therefore has to extend beyond the components performing immediate computation.

---
## Origin of the Symbolic System

The human mathematical and in fact any notation did not exist eternally. Human beings have existed for a finite amount of time, and mathematical symbolic traditions have existed for a still shorter period.

A simplified history can therefore be represented as: $$H_0 \rightarrow \cdots \rightarrow H_n.$$
This is not meant as a literal historical structure. Human symbolic development is distributed, nonlinear, culturally fragmented, collaborative, and recursive. The linear representation preserves only the relevant dependence: later symbolic users inherit systems developed by the earlier ones. Eventually, however, the lineage terminates.

At some point there were human or proto-human cognitive systems that did not themselves receive the relevant symbolic system from earlier users of that same system. The symbols emerged from the interactions between those cognitive systems and the world. A better representation is therefore: $$W \leftrightarrow H_0 \rightarrow \Sigma_H,$$
where:
- $W$ is the world,
- $\Sigma_H$ is the human symbolic system.

Humans did not need to create the structures described by mathematics. The world already exhibits quantity, repetition, order, spatial structure, motion, symmetry, etc. What emerged within the human lineage was a system of symbolic representations capable of expressing and manipulating structures.

---

I am ending this part here.
