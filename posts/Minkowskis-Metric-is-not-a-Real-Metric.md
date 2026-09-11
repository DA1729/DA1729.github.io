---
title: "Minkowski's Metric is not a Real Metric"
date: 2026-09-10
---

I had my Analysis exam today. It was a lot of measure theory and point-set topology. It went well, and now I have General Relativity tomorrow, and in that we have what is called a Minkowski Metric. This is not a post explaining what's that and measure theory stuff, but on a very niche and well seemingly not interesting fact that Minkowski metric does not really satisfy conditions of a real metric. I am not really surprised, having studied both math and physics these past 3 years, but still IDK, it's interesting to me. Mostly because I am finding myself naturally being able to form connections between abstract math structures and abstract physics structures, and being able to point out such stuff by myself.

Take $1+1$-dimensional Minkowski space with

$$
\eta=\begin{pmatrix}-1&0\\\\0&1\end{pmatrix},
\qquad
\eta(v,v)=-t^2+x^2.
$$

A real metric $d$ must satisfy

$$
d(p,q)\ge 0,\qquad
d(p,q)=0\iff p=q,
$$

as well as symmetry and the triangle inequality.

Now choose

$$
p=(0,0),\qquad q=(1,0).
$$

Then

$$
\eta(q-p,q-p)=-(1)^2+0^2=-1<0.
$$

So **positivity fails**.

Next choose

$$
p=(0,0),\qquad q=(1,1).
$$

Then

$$
\eta(q-p,q-p)=-(1)^2+(1)^2=0,
$$

but

$$
p\neq q.
$$

So **positive definiteness / identity of indiscernibles fails**.

Hence

$$
\boxed{\eta \text{ is not a metric-space metric.}}
$$

It is instead a **pseudo-Riemannian metric**, because it is symmetric and non-degenerate but **not positive definite**.
