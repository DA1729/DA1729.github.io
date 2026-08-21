---
title: "Representational Multiplicity and Abstraction -- 2"
date: "2026-08-19"
---

# Representational Multiplicity and Abstraction -- 2

These are some notes, enjoy.

## 1. Starting intuition

A single underlying thing may admit many representations:

$$A_X \leftarrow X \rightarrow B_X.$$

Originally, $A_X$ and $B_X$ were different linguistic encodings of the same concept $X$.

The broader question is:

$$\boxed{\text{representation}\neq\text{represented structure}}$$

Language is only one example. Representations could also be symbolic, geometric, computational, diagrammatic, algebraic, etc.

---

## 2. First abstraction intuition

A possible view of abstraction is:

$$\boxed{\text{abstraction} \sim \text{recognizing what survives changes of representation}}$$

If $r$ is a representation and $T$ changes its form, then something structural might satisfy

$$F(r)=F(T(r)).$$

Here $F$ extracts something insensitive to that representational change.

But this should **not** yet be taken as a definition of abstraction.

---

## 3. "Representation" may mean several different things

The word representation may hide several distinct relationships.

### Encoding

An already existing object $X$ is encoded as $r$:

$$X\xrightarrow{E}r.$$

Example intuition: writing the same number in decimal or binary.

### Presentation

The representation $r$ gives a particular presentation of $X$:

$$r\twoheadrightarrow X.$$

The presentation may contain arbitrary or redundant choices.

### Observation

A representation may reveal only some aspect of an object:

$$X\xrightarrow{o}r.$$

Here $r$ is not a full encoding of $X$.

### Constitutive representation

Perhaps $X$ is not independently available at all.

Instead,

$$\\{r_i,\text{relations between }r_i\\}\rightsquigarrow X.$$

The abstract object is reconstructed from the network of representations.

This possibility should remain open.

---

## 4. Two opposing pictures of the abstract object

### Object-first picture

There is an underlying $X$, and representations are generated from it:

$$X\longrightarrow P_i(X).$$

Then abstraction means recovering properties of $X$ from its presentations.

### Representation-first picture

We begin only with presentations and relations:

$$\\{P_i,T_{ij},\ldots\\}\longrightarrow X.$$

Then $X$ is something reconstructed or inferred.

These are conceptually different theories.

We should not assume beforehand that the first one is correct.

---

## 5. Observation-relative sameness

Suppose $\mathcal O$ is a family of observations.

Define

$$r\equiv_{\mathcal O}s \iff \forall o\in\mathcal O,\quad o(r)=o(s).$$

Then $r$ and $s$ are indistinguishable to the observations available in $\mathcal O$.

Important:

$$r\equiv_{\mathcal O}s$$

does **not automatically mean**

$$r\text{ and }s\text{ are the same underlying thing}.$$

It only means that the chosen observational system cannot distinguish them.

---

## 6. Several notions of sameness may coexist

We may need to distinguish:

$$r\equiv_{\mathcal O}s$$

observational sameness,

$$r\sim_{\mathrm{str}}s$$

structural sameness,

and perhaps

$$r\sim_{\mathrm{gen}}s$$

sameness of generative origin.

These need not coincide.

For weak observations,

$$r\equiv_{\mathcal O}s$$

can happen even if the representations are structurally very different.

Conversely, two structurally equivalent objects may be distinguishable by representation-sensitive observations.

---

## 7. Abstraction depends on what is allowed to matter

Suppose

$$F(r)=F(T(r)).$$

Then $F$ ignores the distinction introduced by $T$.

But that distinction may be irrelevant for one purpose and important for another.

For example, two programs may compute the same function:

$$\operatorname{Sem}(P)=\operatorname{Sem}(P'),$$

while having different costs:

$$\operatorname{Cost}(P)\neq\operatorname{Cost}(P').$$

So there may be no absolute notion of "irrelevant representational detail."

Rather,

$$\boxed{\text{irrelevance is relative to what must be preserved}.}$$

---

## 8. Invariance is weaker than irrelevance

If

$$F(r)=F(T(r)),$$

we know only that $F$ is invariant under $T$.

We do **not** yet know that $T$ changes nothing important.

There may exist another observation $G$ such that

$$G(r)\neq G(T(r)).$$

Therefore:

$$\text{invariant under }F \not\Rightarrow \text{universally irrelevant}.$$

This distinction seems fundamental.

---

## 9. Increasing observational power changes equivalence

If

$$\mathcal O_1\subseteq\mathcal O_2,$$

then generally

$$\equiv_{\mathcal O_2} \subseteq \equiv_{\mathcal O_1}.$$

A stronger observer can distinguish more cases.

So abstraction may depend on **resolution**.

At low resolution:

$$r_1\sim r_2\sim r_3.$$

At higher resolution:

$$r_1\sim r_2, \qquad r_3\not\sim r_1.$$

---

## 10. There may not be one linear hierarchy of abstraction

It is tempting to imagine:

$$\mathcal O_1 \subseteq \mathcal O_2 \subseteq \mathcal O_3 \subseteq\cdots$$

giving increasingly fine abstractions.

But different observational systems may reveal different kinds of distinctions.

For example,

$$\mathcal O_A\not\subseteq\mathcal O_B, \qquad \mathcal O_B\not\subseteq\mathcal O_A.$$

Then abstraction levels may form a branching or partially ordered structure rather than one scale:

$$\begin{array}{ccc}
& \mathcal O_{\mathrm{fine}} & \\\\
/ & & \backslash \\\\
\mathcal O_A & & \mathcal O_B \\\\
\backslash & & / \\\\
& \mathcal O_{\mathrm{coarse}} &
\end{array}$$

So "more abstract" may not always be a one-dimensional notion.

---

## 11. Representational multiplicity alone is not enough

A naive hypothesis would be:

$$\text{more representations} \Rightarrow \text{better abstraction}.$$

This is false.

Suppose we have

$$r_1,\ldots,r_{1000},$$

but no information connecting them.

Then multiplicity may provide little useful structure.

The stronger idea is:

$$\boxed{\text{representational multiplicity} + \text{cross-representational relations}}$$

or

$$\boxed{\text{multiple views} + \text{constraints between views}.}$$

The relations may matter more than the number of representations.

---

## 12. The network may be more important than the individual representations

Instead of merely having

$$\\{r_1,r_2,\ldots,r_n\\},$$

we might have

$$\\{r_i,T_{ij}\\}.$$

For example,

$$r_1\xrightarrow{T_{12}}r_2, \qquad r_2\xrightarrow{T_{23}}r_3.$$

Different transformation paths may themselves contain information:

$$r_1\to r_2\to r_3$$

versus

$$r_1\to r_4\to r_3.$$

Therefore the abstract object may not simply be an equivalence class

$$[r].$$

Some information may live in the **pattern of transformations**.

---

## 13. A central unresolved question: where do the transformations come from?

Suppose we decide

$$r\sim s$$

whenever some allowed transformation connects them.

Then we have introduced a family

$$\mathcal T.$$

But:

$$\boxed{\text{Who chooses }\mathcal T?}$$

If $\mathcal T$ is supplied beforehand, much of the abstraction may already have been encoded into the system.

For genuine abstraction discovery, perhaps the system must infer

$$\mathcal T^\ast$$

itself.

---

## 14. Too many transformations destroy structure

If every transformation is allowed, then everything may become equivalent:

$$\mathcal T=\\{\text{all transformations}\\}$$

could give

$$r\sim s \qquad \forall r,s.$$

Then all distinctions disappear.

At the opposite extreme,

$$\mathcal T=\\{\operatorname{id}\\}$$

preserves every distinction and produces no abstraction.

So a useful abstraction lies somewhere between:

$$\boxed{\text{collapse too little} \quad\longleftrightarrow\quad \text{collapse too much}.}$$

---

## 15. There is a circularity between transformations and invariants

To decide whether $T$ is an acceptable representational transformation, we might say:

$$F(T(r))=F(r).$$

But then we must already know which $F$ matters.

Conversely, perhaps we discover $F$ by noticing what remains unchanged under transformations.

So:

$$\boxed{F\longleftrightarrow\mathcal T}$$

There may be no natural starting point.

This circularity might be important rather than problematic.

---

## 16. A similar circularity exists between observations and equivalence

We define:

$$r\equiv_{\mathcal O}s$$

using observations $\mathcal O$.

But perhaps we choose useful observations because they distinguish the equivalence classes we care about.

Thus:

$$\boxed{\mathcal O\longleftrightarrow\sim}$$

Again, neither side may be primitive.

---

## 17. The object itself may participate in the same circularity

We may end up with something like

$$X \longleftrightarrow \mathcal T \longleftrightarrow \mathcal O \longleftrightarrow \sim.$$

More completely:

$$\boxed{(\text{representations}, \text{transformations}, \text{observations}, \text{invariants}, \text{objects})}$$

may have to be determined together.

This is much stronger than:

$$\text{given }X,\text{ find its invariants}.$$

---

## 18. A useful minimal mathematical toy world

Start only with representations:

$$R=\\{r_1,\ldots,r_n\\}.$$

Add some partial transformations:

$$T_{ij}:r_i\dashrightarrow r_j.$$

Add some observations:

$$o_k:R\to Y_k.$$

Possibly also record transformation composition:

$$T_{jk}\circ T_{ij}.$$

Then ask:

$$\boxed{\text{What abstract objects are justified by this information?}}$$

Not:

> What abstract object did we secretly assume generated it?

That distinction seems important.

---

## 19. Candidate answers for what an "object" might be

Given such a system, an object could possibly be:

an equivalence class,

$$[r]_\sim;$$

an orbit of transformations,

$$\\{T(r):T\in\mathcal T\\};$$

an observational profile,

$$(o_1(r),o_2(r),\ldots);$$

a connected region of the representation network;

a pattern of compatible transformations;

or something not reducible to any individual representation at all.

We should keep all of these possibilities open.

---

## 20. Computation gives a stronger test of representational sameness

Suppose computation occurs in two representations:

$$\begin{array}{ccc}
r & \xrightarrow{C_r} & r' \\\\
\downarrow T & & \downarrow T \\\\
s & \xrightarrow{C_s} & s'
\end{array}$$

A natural compatibility test is

$$T\circ C_r \sim C_s\circ T.$$

If this holds, then the change of representation may preserve the computation in question.

---

## 21. Failure of compatibility may be informative

Usually we focus on when the diagram commutes.

But failure may be equally important:

$$T\circ C_r \not\sim C_s\circ T.$$

This means some distinction we attempted to ignore actually matters for the computation.

So abstraction can be corrected by **obstructions**.

A possible process is:

$$\text{propose abstraction}$$

$$\downarrow$$

$$\text{test it under operations}$$

$$\downarrow$$

$$\text{discover failure}$$

$$\downarrow$$

$$\text{restore a distinction}.$$

This gives abstraction a dynamic character.

---

## 22. Abstraction may involve both forgetting and recovering distinctions

A common picture is:

$$\text{concrete}\rightarrow\text{abstract}$$

by forgetting details.

But intelligent abstraction may require movement in both directions:

$$\text{distinctions} \xrightarrow{\text{ignore}} \text{abstraction} \xrightarrow{\text{failure}} \text{refined distinctions}.$$

So abstraction is not necessarily monotonic.

One may first identify:

$$r_1\sim r_2,$$

and later discover a context in which

$$r_1\not\sim r_2.$$

---

## 23. This suggests context-sensitive abstraction

Instead of one global relation

$$r\sim s,$$

perhaps we need

$$r\sim_C s,$$

where $C$ describes a context, task, collection of operations, or observational regime.

Then:

$$r\sim_{C_1}s$$

but

$$r\not\sim_{C_2}s.$$

The abstract object itself may therefore depend on context.

---

## 24. Compression and abstraction should not be identified

Suppose many representations can be encoded compactly:

$$r_1,\ldots,r_n \longrightarrow z.$$

Even if

$$L(z)\ll\sum_i L(r_i),$$

this does not prove that $z$ captures meaningful abstract structure.

Compression may exploit accidental regularity.

Therefore keep separate:

$$\text{compression},$$

$$\text{invariance},$$

$$\text{abstraction},$$

$$\text{generalization},$$

$$\text{explanation}.$$

Their relationships are questions, not assumptions.

---

## 25. The data-efficiency intuition

A system that discovers a reusable structural law $L$ may obtain:

$$L\Rightarrow \\{x_1,x_2,\ldots,x_N\\}$$

without independently learning every $x_i$.

Thus:

$$\boxed{\text{one structural discovery} \Rightarrow \text{many implied cases}.}$$

This could reduce dependence on large numbers of superficially different examples.

But the missing information must come from somewhere:

$$\text{less data} \Rightarrow \text{more structure/inference/computation/interaction}.$$

---

## 26. The more interesting learning problem

The goal is therefore not simply to give a learner useful invariances.

That would amount to supplying the abstraction ourselves.

The more ambitious problem is:

$$\boxed{\text{Can the learner discover which invariances should exist?}}$$

And further:

$$\boxed{\text{Can it later revise those invariances?}}$$

That seems closer to genuine structural learning.

---

## 27. Possible recursive process

A hypothetical system might proceed roughly as:

$$\text{examples}$$

$$\downarrow$$

$$\text{candidate correspondences}$$

$$\downarrow$$

$$\text{candidate transformations}$$

$$\downarrow$$

$$\text{candidate invariants}$$

$$\downarrow$$

$$\text{candidate equivalences}$$

$$\downarrow$$

$$\text{test under new observations/operations}$$

$$\downarrow$$

$$\text{refine}.$$

The word **candidate** matters. None of these should be treated as final immediately.

---

## 28. A deeper formulation of the project

The interesting question may not be:

$$\text{How do we find invariants?}$$

It may be:

$$\boxed{\text{How are the right invariants, transformations, observations, and objects jointly discovered?}}$$

In particular, suppose initially none of

$$X, \qquad \mathcal T, \qquad \mathcal O, \qquad \sim$$

is given.

Can they emerge together from structured interaction among representations?

---

## 29. A useful guiding distinction

Whenever we construct a theory, ask:

$$\boxed{\text{Did we explain the abstraction, or did we secretly put it in the assumptions?}}$$

For example, if we give a system exactly the correct equivalence relation,

$$\sim,$$

then forming

$$R/{\sim}$$

is easy.

The difficult part is explaining why **that** equivalence relation is justified.

Likewise, finding invariants of a given transformation family may be straightforward once

$$\mathcal T$$

has already been chosen.

The difficult question is where $\mathcal T$ came from.

---

## 30. Current conceptual nucleus

At the moment, the project seems to orbit the following problem:

$$\boxed{\begin{array}{c}
\text{Given multiple presentations of experience,} \\\\
\text{how can a system discover} \\\\
\\\\
\text{which differences are representational,} \\\\
\text{which differences are structural,} \\\\
\text{which transformations are meaningful,} \\\\
\text{what must remain invariant,} \\\\
\text{what counts as the same object,} \\\\
\text{and when an ignored distinction must return?}
\end{array}}$$

The important feature is that these notions may have to be **co-determined**, rather than one being fixed first and the others derived from it.

---

### Short phrases worth keeping

$$\boxed{\text{Multiplicity without correspondence may teach nothing.}}$$

$$\boxed{\text{Sameness is often relative to observational power.}}$$

$$\boxed{\text{Invariance does not automatically imply irrelevance.}}$$

$$\boxed{\text{Abstraction may be context-dependent.}}$$

$$\boxed{\text{Abstraction may require restoring distinctions, not only removing them.}}$$

$$\boxed{\text{The network of representations may be more fundamental than any one representation.}}$$

$$\boxed{\text{The hard problem is discovering the quotient, not taking the quotient.}}$$

$$\boxed{\text{The hard problem is discovering the transformations, not merely finding their invariants.}}$$

$$\boxed{\text{An abstract object might be reconstructed rather than presupposed.}}$$

$$\boxed{\text{Intelligence may involve discovering both what does not matter and when it matters again.}}$$
