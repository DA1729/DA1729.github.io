---
title: Graham-Entringer-Szekely Theorem
date: 2026-04-23 22:00:47
tags:
    - Graph Theory
---

Here is the Theorem Statement out of the DB West book (Theorem 8.3.2)
# Theorem
If $T$ is a spanning tree of the $k$-dimensional cube $Q\_k$, then there is an edge of $Q\_k$ outside of $T$ whose addition to $T$ creates a cycle of length at least $2k$.

# Proof
## Setup
In $Q\_k$, every vertex $v$ (a binary $k$-tuple) has a complement $v'$ (flip all the bits). These two vertices are as far apart as possible, i.e., distance exactly $k$, since we must flip every bit.

Since $T$ is a spanning tree, there is a unique path in $T$ between any two vertices. So for every $v$, there's a unique $v \rightarrow v'$ path in $T$.

## The Claiming Argument
We define a rule: every vertex $v$ **claims** the first edge on its unique tree-path toward $v'$.

Now count:
- Vertices doing the claiming: $2^k$
- Edges in $T$ available to be claimed: $2^k - 1$.

By Pigeonhole Principle, some edge $\{u, v\}$  in $T$ gets claimed by two different vertices.

## Double-Claiming
The only way both $u$ and $v$ claim the edge $\{u, v\}$ is if: 
- $v$ lies on the $u$-to-$u'$ path in $T$, and is immediate neighbour of $u$ on that path.
- $v$'s parth towards $v'$ starts by stepping to $u$ so $u$ lies on the $v$-to-$v'$ path in $T$.

Now if we remove this $\{u, v\}$ edge, we are essentially splitting $T$ into two components: $T\_u$ (containing $u$) and $T\_v$ (containing $v$).

From above, we know:
- $u'$ is in $T\_v$
- $v'$ is in $T\_u$

## Building the Cycle
We know for a fact that all such paths between complements, have length atleast $k$, therefore $u-v'$ and $v-u'$ paths both have lengths atleast $k-1$.

Since $u$ and $v$ are neighbours, their complements are also neighbours for each other and they must not have any edge between them, since they both lie in different components and we know that those components were obtained by deleting $uv$ edge.

So we can stitch together a cycle using this edge not in $T$ with total length $L$: $$ L \geq (k - 1) + 1 + (k - 1) + 1 = 2k$$ 

QED
