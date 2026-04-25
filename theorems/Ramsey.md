---
title: Ramsey
topic: Graph Theory
date: 2026-04-26
tags:
    - Graph Theory
    - Ramsey Theory
    - Combinatorics
---

Write $R(p\_1, \dots, p\_n; r)$ for the least integer $N$ that satisfies the conclusion; we are showing this number is finite. Call a set $X$ *$i$-homogeneous* under a coloring $f$ if every $r$-subset of $X$ has $f$-color $i$. We proceed by induction on $r$.

## Base case: $r = 1$

For $r = 1$, an $n$-coloring of $\binom{[N]}{1}$ is just an $n$-coloring of the elements of $[N]$. Taking
$$ N = \sum\_{i=1}^{n} p\_i - n + 1 $$
and applying the pigeonhole principle, some color $i$ appears on at least $p\_i$ elements, giving an $i$-homogeneous set of size $p\_i$.

## Inductive step: $r \geq 2$

Assume the claim holds for every $n$-coloring of $(r - 1)$-subsets, for **every** parameter tuple. Call this the *outer induction hypothesis*. Within this hypothesis we run a second induction on $\sum\_i p\_i$.

### Inner base case

Suppose some $p\_i \leq r - 1$. Then any set of $p\_i$ elements has no $r$-subset, so its (empty) collection of $r$-subsets is *vacuously* $i$-homogeneous. Hence taking $N = \min\\{p\_1, \dots, p\_n\\}$ already works.

### Inner inductive step

Assume now that $p\_i \geq r$ for every $i$. For each $i \in \\{1, \dots, n\\}$, define
$$ p'\_i = R(p\_1, \dots, p\_i - 1, \dots, p\_n; r), $$
i.e., the Ramsey number with the $i$-th argument decremented by one. By the inner induction hypothesis (the parameter sum has dropped), all the $p'\_i$ exist. Now let
$$ N = 1 + R(p'\_1, \dots, p'\_n; r - 1), $$
which exists by the **outer** induction hypothesis (subset size dropped from $r$ to $r - 1$).

We claim this $N$ works. Let $S$ be a set of $N$ elements and let $f \colon \binom{S}{r} \to \\{1, \dots, n\\}$ be any $n$-coloring.

#### Step 1. Set up an auxiliary $(r - 1)$-coloring.

Pick any $x \in S$ and let $S' = S \setminus \\{x\\}$, so $|S'| = R(p'\_1, \dots, p'\_n; r - 1)$. Define a coloring $f'$ on $\binom{S'}{r - 1}$ by
$$ f'(A) = f\left(A \cup \\{x\\}\right). $$
By the **outer** induction hypothesis applied to $f'$, there exist some $i$ and some $T \subseteq S'$ with $|T| = p'\_i$ such that $T$ is $i$-homogeneous under $f'$. Equivalently, every $r$-subset of $T \cup \\{x\\}$ that contains $x$ has $f$-color $i$.

#### Step 2. Recurse inside $T$.

The set $T$ has size $p'\_i = R(p\_1, \dots, p\_i - 1, \dots, p\_n; r)$. Apply the **inner** induction hypothesis to the restriction of $f$ to $\binom{T}{r}$. There are two cases:

- there is a $j$-homogeneous set of size $p\_j$ in $T$ for some $j \neq i$ — done immediately, or
- there is an $i$-homogeneous set $P \subseteq T$ of size $p\_i - 1$.

Suppose the second case holds.

#### Step 3. Glue $P$ and $x$.

Consider $P \cup \\{x\\}$, a set of size $p\_i$. Each $r$-subset of $P \cup \\{x\\}$ is one of two kinds:

- An $r$-subset *not* containing $x$. This lies inside $P$, which is $i$-homogeneous under $f$, so it has $f$-color $i$.
- An $r$-subset containing $x$. It has the form $A \cup \\{x\\}$ for some $(r - 1)$-subset $A \subseteq P \subseteq T$. By Step 1, $A$ has $f'$-color $i$, and $f'(A) = f(A \cup \\{x\\})$, so $A \cup \\{x\\}$ has $f$-color $i$.

In either case the $r$-subset has color $i$. Hence $P \cup \\{x\\}$ is an $i$-homogeneous set of size $p\_i$, completing the proof.
