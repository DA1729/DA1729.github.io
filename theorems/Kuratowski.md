---
title: Kuratowski
topic: Graph Theory
date: 2026-05-04
tags:
    - Graph Theory
    - Planarity
    - Combinatorics
---

## Setup

The "if it has a Kuratowski subgraph then it's nonplanar" direction is direct: any subdivision of $K\_5$ or $K\_{3,3}$ is itself nonplanar, and embeddability is monotone under taking subgraphs.

The hard direction is what we prove. Suppose it fails. Among all counterexamples, take $G$ with the fewest edges. So $G$ is nonplanar, contains no Kuratowski subgraph, and every proper subgraph of $G$ is planar. We will show no such $G$ exists.

The plan has two phases:

1. $G$ must be 3-connected.
2. Every 3-connected graph with no Kuratowski subgraph admits a *convex* embedding (every face boundary is a convex polygon), and is therefore planar.

We will use one geometric fact repeatedly without re-proving it. Call it the **face-rotation lemma**: in any planar embedding, we can choose any face to be the outer face by inverting through a point of that face. So whenever we have a planar embedding of a piece, we can put any of its faces on the outside.

## Phase 1: $G$ is 3-connected

### $G$ is connected

If $G$ is disconnected, every component is a proper subgraph of $G$, hence planar. Embed one component, then drop the others into distinct faces of its embedding. The result is a planar embedding of $G$, contradicting nonplanarity.

### $G$ is 2-connected

Suppose $v$ is a cut vertex, and let $G\_1, \dots, G\_k$ be the $\\{v\\}$-lobes (each lobe is one piece of $G - v$ together with $v$ reattached). Each $G\_i$ is a proper subgraph of $G$, hence planar. By the face-rotation lemma, embed each $G\_i$ with $v$ on its outer face. Now glue the embeddings together at $v$, fitting each lobe into a wedge of angle less than $360^\circ / k$. No edges cross, and we have a planar embedding of $G$. Contradiction.

### Adding $xy$ to some lobe stays nonplanar

Now assume $G$ is 2-connected. Let $S = \\{x, y\\}$ be a separating 2-set, and let $G\_1, \dots, G\_k$ be the $S$-lobes.

*Claim.* For some $i$, the graph $H\_i = G\_i \cup xy$ (the $i$-th lobe with edge $xy$ added, in case it was not already there) is nonplanar.

Suppose not: every $H\_i$ is planar. Embed each $H\_i$ in the plane and use the face-rotation lemma to put $xy$ on the outer face. Embed $H\_1$, then place $H\_2$ inside a face of $H\_1$ whose boundary contains $xy$, gluing along the shared edge. Repeat for $H\_3, \dots, H\_k$. Finally delete $xy$ if it was not originally an edge of $G$. The result is a planar embedding of $G$. Contradiction.

### $G$ is 3-connected

Suppose $G$ has a separating 2-set $S = \\{x, y\\}$. Use the previous claim to pick a lobe $G\_i$ such that $H = G\_i \cup xy$ is nonplanar. Since $H$ has fewer edges than $G$, and $G$ is a *minimum*-edge counterexample, $H$ must contain a Kuratowski subgraph $F$.

$F$ must use the edge $xy$, otherwise $F$ would already lie inside $G$, contradicting that $G$ has no Kuratowski subgraph.

Now, $S$ is a minimal vertex cut, so each of $x, y$ has a neighbor in every $S$-lobe. Pick a different lobe $G\_j$ (with $j \neq i$). Inside $G\_j$ there is an $x$-$y$ path $P$, going through the connected lobe via neighbors of $x$ and $y$. Replace the edge $xy$ in $F$ with this path $P$. The result is a subdivision of the same Kuratowski graph, sitting inside $G$ itself. This contradicts $G$ being Kuratowski-free.

Therefore $G$ has no separating 2-set, i.e., $G$ is 3-connected.

## Phase 2: every 3-connected Kuratowski-free graph has a convex embedding

We strengthen the conclusion from "planar" to "convex" because the induction needs the geometric structure, not just planarity. Induct on $n = |V(G)|$.

### Base case $n \leq 4$

The only 3-connected graph on at most 4 vertices is $K\_4$, which has a convex embedding: a triangle with a vertex placed inside, edges drawn straight.

### Induction step $n \geq 5$

The strategy is to contract an edge $e = xy$ to a single vertex $z$, get a smaller graph $G \cdot e$, apply the induction hypothesis, and expand $z$ back to recover an embedding of $G$. Three sub-arguments:

- **(I) Safe edge.** Some edge $e \in E(G)$ has $G \cdot e$ still 3-connected.
- **(II) Contraction is Kuratowski-safe.** $G$ Kuratowski-free implies $G \cdot e$ Kuratowski-free.
- **(III) Expansion is convex-safe.** A convex embedding of $G \cdot e$ extends to a convex embedding of $G$.

By (I), (II), and induction, $G \cdot e$ has a convex embedding; (III) lifts it to one of $G$.

#### (I) A safe edge exists

Suppose not: every edge $xy \in E(G)$ has $G \cdot xy$ with a separating 2-set. Since $G$ is 3-connected, this 2-set must include the merged vertex $z$, so it has the form $\\{z, w\\}$. Equivalently, $\\{x, y, w\\}$ is a separating 3-set in $G$ itself.

Among all such triples $(xy, w)$, choose one where the largest component of $G - \\{x, y, w\\}$ has the most vertices. Call this component $H$, and let $H'$ be any other component.

Since $\\{x, y, w\\}$ is a *minimal* separating set, $w$ has a neighbor $u$ in $H'$. Apply the standing assumption to the edge $wu$: there is a vertex $v$ such that $\\{w, u, v\\}$ separates $G$.

Look at the subgraph induced by $V(H) \cup \\{x, y\\}$ in $G$. Its vertex set:

- avoids $w, x, y$ (by definition of $H$, and since $x, y$ are added but not in any component of $G - \\{x, y, w\\}$),
- avoids $u$ (since $u \in H'$, not $H$),
- may contain $v$, but if so we drop it and still have at least $|V(H)| + 1$ vertices.

This subgraph is connected without using any of $w, u, v$: $H$ is connected internally, and $x, y$ each have a neighbor in $H$ (again by minimality of the cut $\\{x, y, w\\}$). So $V(H) \cup \\{x, y\\}$, minus $v$ if needed, lies inside a single component of $G - \\{w, u, v\\}$, which therefore has at least $|V(H)| + 1$ vertices.

But by the extremal choice, the largest component of $G - \\{w, u, v\\}$ has at most $|V(H)|$ vertices. Contradiction.

#### (II) Contraction preserves Kuratowski-freeness

Contrapositive: if $G \cdot e$ contains a Kuratowski subgraph $K$, then so does $G$.

Let $z$ be the merged vertex from $e = xy$. Recall that *branch vertices* of a subdivision are the vertices of degree $\geq 3$ (the originals of $K\_5$ or $K\_{3,3}$); the rest are degree-2 vertices in the middle of subdivided paths. Cases on the role of $z$ in $K$.

*Case 1: $z \notin V(K)$.* $K$ already lives inside $G$. Done.

*Case 2: $z$ is a non-branch vertex of $K$.* Then $z$ has degree 2 in $K$, lying inside a single subdivided path. Each of its two edges in $K$ came from $x$ or from $y$ in $G$. If both came from the same side, replace $z$ in $K$ by that vertex. Otherwise, replace $z$ by the pair $x, y$ joined by the edge $xy$. In either case the path is at most one edge longer, and $K$ lifts to a subdivision in $G$.

*Case 3: $z$ is a branch vertex of $K$, and $K$ is a $K\_{3,3}$ subdivision.* $z$ has three edges in $K$, splitting between $x$-side and $y$-side as $(3, 0)$ or $(2, 1)$. Replace $z$ by $x$ (say), and route the at-most-one $y$-side edge by appending the edge $xy$ to the start of its path. We get a $K\_{3,3}$ subdivision in $G$.

*Case 4: $z$ is a branch vertex of $K$, $K$ is a $K\_5$ subdivision, split $(4, 0)$ or $(3, 1)$.* Same as Case 3: replace $z$ by $x$, push the at-most-one $y$-side edge through $xy$.

*Case 5: $z$ is a branch vertex of $K$, $K$ is a $K\_5$ subdivision, split $(2, 2)$.* The interesting case. Let $u\_1, u\_2$ be the branch vertices of $K$ whose paths to $z$ entered through $x$-side edges, and $v\_1, v\_2$ those entering through $y$-side edges. So in $K$, the five branch vertices are $\\{z, u\_1, u\_2, v\_1, v\_2\\}$ with internally disjoint paths between every pair.

In $G$, expand $z$ into $x$ and $y$ joined by the edge $xy$, with $x$ inheriting the paths to $u\_1, u\_2$ and $y$ inheriting the paths to $v\_1, v\_2$. Now consider the subgraph consisting of:

- the partition $\\{x, v\_1, v\_2\\}$ versus $\\{y, u\_1, u\_2\\}$,
- the edge $xy$,
- the inherited $z$-$u\_i$ paths (now $x$-$u\_i$),
- the inherited $z$-$v\_j$ paths (now $y$-$v\_j$),
- the original $u\_i$-$v\_j$ paths from $K$.

That is exactly nine internally disjoint paths between the parts, i.e., a subdivision of $K\_{3,3}$ in $G$. The unused $u\_1$-$u\_2$ and $v\_1$-$v\_2$ paths from $K$ are simply discarded.

In every case, $G$ contains a Kuratowski subgraph.

#### (III) Expansion: from $G \cdot e$ convex to $G$ convex

By (I) and (II) and the induction hypothesis, $G \cdot e$ has a convex embedding. Let $z$ be the merged vertex. Delete the edges of $G \cdot e$ incident to $z$. Since $G \cdot e$ is 3-connected, $(G \cdot e) - z$ is 2-connected, and the face that now contains $z$ has a boundary cycle $C$ on which all neighbors of $z$ lie, in some cyclic order.

In $G$, each neighbor of $z$ corresponds to a neighbor of $x$ or of $y$ (a vertex adjacent to both $x$ and $y$ may be assigned either role). Call them *$x$-neighbors* and *$y$-neighbors*.

*Good case.* All $y$-neighbors lie in a single arc of $C$ between two consecutive $x$-neighbors $x\_i$ and $x\_{i+1}$.

Place $x$ at $z$'s old position. Place $y$ very close to $x$, inside the wedge formed by the edges $x x\_i$ and $x x\_{i+1}$. Connect $x$ to its neighbors and $y$ to its neighbors (those edges live inside the wedge), and add the edge $xy$. No crossings, all faces remain convex.

*Bad cases.* Suppose the good case fails. Then the $x$-neighbors and $y$-neighbors interleave around $C$ in one of two ways:

- $y$ has three neighbors $a, b, c$ on $C$ that interleave with two $x$-neighbors $x\_i, x\_{i+1}$;
- $y$ has two neighbors $a, b$ on $C$ that interleave with two consecutive $x$-neighbors $x\_i, x\_{i+1}$.

In the first situation, the cycle $C$, the edge $xy$, and the edges from $\\{x, y\\}$ to $\\{a, b, c, x\_i, x\_{i+1}\\}$ contain a $K\_5$ subdivision.

In the second, the paths $a$-$y$-$b$, $x\_i$-$x$-$x\_{i+1}$, the edge $xy$, and the cycle $C$ contain a $K\_{3,3}$ subdivision on the parts $\\{x, a, b\\}$ and $\\{y, x\_i, x\_{i+1}\\}$.

Either is a Kuratowski subgraph of $G$, contradicting our hypothesis. So the good case must occur, and $G$ inherits a convex embedding.

## Putting it together

If a Kuratowski-free nonplanar graph existed, the smallest one $G$ would, by Phase 1, be 3-connected, and by Phase 2 would have a convex embedding. So $G$ is planar, contradicting nonplanarity. No counterexample exists.
