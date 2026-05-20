---
title: First Isomorphism Theorem
topic: Algebra
date: 2026-05-20
tags:
    - Algebra
    - Set Theory
---

The first isomorphism theorem is usually stated separately for groups, rings, vector spaces, modules, and so on: a homomorphism factors through a quotient as an isomorphism onto its image. The statement below is the underlying set-theoretic core, with no algebraic structure assumed. Each of the algebraic versions follows by checking that the construction respects the relevant structure (operations, scalar multiplication, etc.).

## Setup

Given $f : A \to B$, define
$$ a' \sim a'' \iff f(a') = f(a''). $$
Reflexivity, symmetry, and transitivity follow directly from those of equality on $B$, so $\sim$ is an equivalence relation on $A$. Write $A/{\sim}$ for the set of equivalence classes and $[a]\_\sim$ for the class of $a$.

We claim $f$ decomposes as
$$ A \;\twoheadrightarrow\; A/{\sim} \;\xrightarrow{\,\sim\,}\; \operatorname{im} f \;\hookrightarrow\; B, $$
where the first map is the canonical surjection $\pi \colon a \mapsto [a]\_\sim$, the last is the inclusion $\operatorname{im} f \subseteq B$, and the middle map is
$$ \tilde{f} \colon [a]\_\sim \mapsto f(a). $$

There are three things to check: that the two outer maps behave as advertised, that $\tilde{f}$ is a well-defined bijection, and that composing the three maps recovers $f$.

## The outer maps

$\pi$ is surjective by construction: every class $[a]\_\sim$ has any of its representatives $a$ as a preimage. The inclusion $\operatorname{im} f \hookrightarrow B$ is trivially injective. All the real content sits in the middle map.

## $\tilde{f}$ is well-defined

The recipe $\tilde{f}([a]\_\sim) := f(a)$ picks a particular representative $a$ of the class, so we need the output not to depend on that choice. Suppose $a' \sim a''$, i.e., $[a']\_\sim = [a'']\_\sim$. Then by the very definition of $\sim$, $f(a') = f(a'')$. So the value $\tilde{f}([a]\_\sim)$ is unambiguous, and $\tilde{f}$ is a genuine function.

## $\tilde{f}$ is injective

Suppose $\tilde{f}([a']\_\sim) = \tilde{f}([a'']\_\sim)$. Unfolding the definition, this is $f(a') = f(a'')$, which is precisely the condition $a' \sim a''$, i.e., $[a']\_\sim = [a'']\_\sim$.

## $\tilde{f}$ is surjective onto $\operatorname{im} f$

Every $b \in \operatorname{im} f$ has, by definition, some $a \in A$ with $f(a) = b$. Then $\tilde{f}([a]\_\sim) = f(a) = b$. Since $b$ was arbitrary, $\tilde{f}$ surjects onto $\operatorname{im} f$.

## The composite is $f$

Tracking an arbitrary $a \in A$ through the three maps,
$$ a \;\xmapsto{\;\pi\;}\; [a]\_\sim \;\xmapsto{\;\tilde{f}\;}\; f(a) \;\xmapsto{\;\iota\;}\; f(a), $$
the composition agrees with $f$ on every input. So $f$ is indeed the composite of a canonical surjection, a bijection, and an inclusion.
