---
title: "Review of Quantum Mechanics"
subject: "Quantum Theory of Solids"
date: "2026-08-11"
---

## 1. Quantum States

- A quantum system is described by a **state vector** $|\psi\rangle$ in a complex Hilbert space $\mathcal H$.
- More precisely, a pure physical state is a **ray**:
  $$
  |\psi\rangle \sim c|\psi\rangle,\qquad c\neq 0
  $$
  so multiplication by an overall nonzero complex number does not change the physical state.
- We normally choose a normalized representative:
  $$
  \langle\psi|\psi\rangle=1.
  $$

### Two-state example

Choose basis states

$$
|L\rangle=
\begin{pmatrix}
1\\\\
0
\end{pmatrix},
\qquad
|R\rangle=
\begin{pmatrix}
0\\\\
1
\end{pmatrix}.
$$

A general state is

$$
|\psi\rangle
=
\alpha|L\rangle+\beta|R\rangle
=
\begin{pmatrix}
\alpha\\\\
\beta
\end{pmatrix},
$$

where $\alpha,\beta\in\mathbb C$.

Normalization:

$$
|\alpha|^2+|\beta|^2=1.
$$

Measurement probabilities:

$$
P(L)=|\alpha|^2,
\qquad
P(R)=|\beta|^2.
$$

- $\alpha,\beta$ are **probability amplitudes**, not probabilities.
- Amplitudes can interfere before their absolute squares are taken.

---

## 2. Bras and Inner Products

For

$$
|\psi\rangle=
\begin{pmatrix}
\alpha\\\\
\beta
\end{pmatrix},
$$

the corresponding **bra** is

$$
\langle\psi|
=
\begin{pmatrix}
\alpha^*&\beta^*
\end{pmatrix}.
$$

For

$$
|\phi\rangle=
\begin{pmatrix}
a\\\\
b
\end{pmatrix},
$$

the inner product is

$$
\langle\phi|\psi\rangle
=
a^*\alpha+b^*\beta.
$$

Norm:

$$
\|\psi\|^2=\langle\psi|\psi\rangle.
$$

Normalized state:

$$
\langle\psi|\psi\rangle=1.
$$

---

## 3. Orthonormal Bases

Two vectors are **orthogonal** if

$$
\langle\phi|\psi\rangle=0.
$$

A basis $\\{|e_n\rangle\\}$ is **orthonormal** if

$$
\langle e_m|e_n\rangle=\delta_{mn}.
$$

Any state can be expanded as

$$
|\psi\rangle=\sum_n c_n|e_n\rangle.
$$

The coefficients are extracted using the inner product:

$$
c_n=\langle e_n|\psi\rangle.
$$

Therefore,

$$
|\psi\rangle
=
\sum_n|e_n\rangle\langle e_n|\psi\rangle.
$$

Completeness relation:

$$
\sum_n|e_n\rangle\langle e_n|=I.
$$

---

## 4. Operators

- An operator is a map
  $$
  \hat A:\mathcal H\rightarrow\mathcal H.
  $$
- In finite dimensions, linear operators are represented by matrices.

Example:

$$
A=
\begin{pmatrix}
a&b\\\\
c&d
\end{pmatrix},
\qquad
|\psi\rangle=
\begin{pmatrix}
\alpha\\\\
\beta
\end{pmatrix}.
$$

Then

$$
A|\psi\rangle
=
\begin{pmatrix}
a\alpha+b\beta\\\\
c\alpha+d\beta
\end{pmatrix}.
$$

### Linearity

$$
A(\alpha|\psi\rangle+\beta|\phi\rangle)
=
\alpha A|\psi\rangle+\beta A|\phi\rangle.
$$

Hence knowing the action of $A$ on a basis determines its action on every state.

---

## 5. Eigenvalues and Eigenvectors

An eigenvector satisfies

$$
A|a\rangle=a|a\rangle.
$$

- $|a\rangle$ = eigenvector/eigenstate
- $a$ = eigenvalue

The operator does not change the Hilbert-space direction of an eigenvector; it only multiplies it by a scalar.

### Quantum interpretation

If $A$ represents an observable:

- its eigenvalues are the possible measurement outcomes;
- if the system is in $|a_n\rangle$, measuring $A$ gives $a_n$ with certainty.

If

$$
|\psi\rangle=\sum_n c_n|a_n\rangle,
$$

then

$$
P(a_n)=|c_n|^2
=
|\langle a_n|\psi\rangle|^2.
$$

---

## 6. Hermitian / Self-Adjoint Operators

For a matrix $A$,

$$
A^\dagger=(A^*)^T.
$$

$A$ is Hermitian if

$$
A^\dagger=A.
$$

Quantum observables are represented by **self-adjoint operators**.

### Important finite-dimensional properties

For Hermitian $A$:

- all eigenvalues are real;
- eigenvectors corresponding to distinct eigenvalues are orthogonal;
- there exists a complete orthonormal eigenbasis.

This is a consequence of the **spectral theorem**.

> An arbitrary operator does **not** necessarily possess a complete eigenbasis.

In infinite-dimensional spaces, continuous spectra require generalized eigenvectors.

---

## 7. Measurement

Suppose

$$
A|a_n\rangle=a_n|a_n\rangle
$$

and

$$
|\psi\rangle=\sum_n c_n|a_n\rangle.
$$

Then measuring $A$:

- can return only an eigenvalue $a_n$;
- returns $a_n$ with probability
  $$
  P(a_n)=|c_n|^2.
  $$

If the result is $a_n$, an ideal measurement updates the state to the corresponding eigenstate/eigenspace.

Therefore, immediately repeating the same measurement gives the same result with certainty.

---

## 8. Expectation Value

The expectation value is the average result over many identically prepared systems:

$$
\boxed{
\langle A\rangle
=
\langle\psi|A|\psi\rangle
}
$$

In the eigenbasis of $A$:

$$
\langle A\rangle
=
\sum_n a_nP(a_n)
=
\sum_n a_n|\langle a_n|\psi\rangle|^2.
$$

> $\langle A\rangle$ does not need to be a possible outcome of a single measurement.

---

## 9. Variance and Uncertainty

Variance:

$$
(\Delta A)^2
=
\langle(A-\langle A\rangle)^2\rangle.
$$

Equivalent form:

$$
\boxed{
(\Delta A)^2
=
\langle A^2\rangle-\langle A\rangle^2
}
$$

Uncertainty / standard deviation:

$$
\boxed{
\Delta A
=
\sqrt{\langle A^2\rangle-\langle A\rangle^2}
}
$$

If $|\psi\rangle$ is an eigenstate of $A$,

$$
\Delta A=0.
$$

---

## 10. Operator Ordering and Commutators

For operators $A$ and $B$,

$$
AB|\psi\rangle
$$

means **apply $B$ first, then $A$**.

In general,

$$
AB\neq BA.
$$

Define the commutator:

$$
\boxed{
[A,B]=AB-BA
}
$$

If

$$
[A,B]=0,
$$

the operators commute.

### Simultaneously definite observables

$A$ and $B$ are simultaneously definite in $|\psi\rangle$ if

$$
A|\psi\rangle=a|\psi\rangle,
$$

and

$$
B|\psi\rangle=b|\psi\rangle.
$$

For finite-dimensional Hermitian operators,

$$
[A,B]=0
$$

means that one can choose a complete orthonormal basis of **common eigenvectors**.

---

## 11. Uncertainty Relation

For observables $A$ and $B$,

$$
\boxed{
\Delta A\,\Delta B
\ge
\frac12
\left|
\langle[A,B]\rangle
\right|
}
$$

where

$$
\langle[A,B]\rangle
=
\langle\psi|[A,B]|\psi\rangle.
$$

This is the **Robertson uncertainty relation**.

- Noncommutativity is a property of the operators.
- The numerical lower bound can depend on the state.

For position and momentum,

$$
[x,p]=i\hbar I,
$$

so

$$
\boxed{
\Delta x\,\Delta p\ge\frac{\hbar}{2}
}
$$

This uncertainty is intrinsic to the quantum state, not merely experimental imprecision.

---

# Position-Space Quantum Mechanics

## 12. Abstract State vs Wavefunction

For one spinless particle on a line,

$$
\mathcal H=L^2(\mathbb R).
$$

The abstract state is

$$
|\psi\rangle.
$$

After choosing the **position representation**, its components are

$$
\boxed{
\psi(x)=\langle x|\psi\rangle.
}
$$

Finite-dimensional analogy:

$$
\psi_n=\langle e_n|\psi\rangle
$$

becomes

$$
\psi(x)=\langle x|\psi\rangle.
$$

> The wavefunction is a representation of the state, not the abstract state itself.

---

## 13. Probability in Position Space

$|\psi(x)|^2$ is a **probability density**.

Probability of finding the particle in an interval $I$:

$$
\boxed{
P(x\in I)
=
\int_I|\psi(x)|^2\,dx
}
$$

Normalization:

$$
\int_{-\infty}^{\infty}|\psi(x)|^2\,dx=1.
$$

For a small interval $dx$,

$$
P(x\rightarrow x+dx)
\approx
|\psi(x)|^2dx.
$$

The probability of finding the particle at one exact point is zero for a continuous distribution.

---

## 14. Position Eigenstates

Formally,

$$
\hat x|x_0\rangle=x_0|x_0\rangle.
$$

In position representation,

$$
\langle x|x_0\rangle
=
\delta(x-x_0).
$$

The states $|x_0\rangle$ are not ordinary elements of $L^2(\mathbb R)$ because the Dirac delta is not square-integrable.

They are **generalized eigenvectors**.

Formally,

$$
\langle x|x'\rangle
=
\delta(x-x').
$$

---

## 15. Position Operator

In the position representation,

$$
\boxed{
(\hat x\psi)(x)=x\psi(x)
}
$$

so $\hat x$ acts by multiplication by $x$.

---

## 16. Momentum Operator

Momentum is the **generator of spatial translations**.

A translation by $a$ acts as

$$
(T(a)\psi)(x)=\psi(x-a).
$$

For small $a$,

$$
\psi(x-a)
=
\psi(x)-a\frac{d\psi}{dx}+O(a^2).
$$

Quantum translations are written

$$
T(a)=e^{-ia\hat p/\hbar}.
$$

For small $a$,

$$
T(a)
\approx
I-\frac{ia}{\hbar}\hat p.
$$

Comparing gives

$$
\boxed{
\hat p=-i\hbar\frac{d}{dx}
}
$$

in the position representation.

Therefore,

$$
\boxed{
\hat x\leftrightarrow x,
\qquad
\hat p\leftrightarrow-i\hbar\frac{d}{dx}
}
$$

---

## 17. Momentum Eigenfunctions

A momentum eigenfunction satisfies

$$
\hat p\psi_p=p\psi_p.
$$

Hence

$$
-i\hbar\frac{d\psi_p}{dx}
=
p\psi_p.
$$

The solution is

$$
\boxed{
\psi_p(x)=Ce^{ipx/\hbar}
}
$$

Writing

$$
p=\hbar k,
$$

we obtain

$$
\psi_k(x)=Ce^{ikx}.
$$

Indeed,

$$
-i\hbar\frac{d}{dx}e^{ikx}
=
\hbar k e^{ikx}.
$$

Therefore,

$$
\boxed{
p=\hbar k
}
$$

where $k$ is the wave number.

---

## 18. Plane Waves and Generalized Eigenstates

For

$$
\psi(x)=e^{ikx},
$$

we have

$$
|\psi(x)|^2=1.
$$

Therefore,

$$
\int_{-\infty}^{\infty}|\psi(x)|^2dx
=
\infty.
$$

So a plane wave cannot be normalized in $L^2(\mathbb R)$.

Momentum eigenstates on the infinite line are therefore **generalized eigenstates** rather than physical normalizable states.

Physical states are wave packets formed from superpositions of plane waves:

$$
\psi(x)
=
\int \phi(k)e^{ikx}\,dk
$$

up to Fourier-transform normalization conventions.

---

# Core Picture

- **Abstract state:** $|\psi\rangle$
- **Position representation:** $\psi(x)=\langle x|\psi\rangle$
- **Observable:** self-adjoint operator $A$
- **Definite value:**
  $$
  A|a\rangle=a|a\rangle
  $$
- **Measurement probability:**
  $$
  P(a_n)=|\langle a_n|\psi\rangle|^2
  $$
- **Expectation value:**
  $$
  \langle A\rangle=\langle\psi|A|\psi\rangle
  $$
- **Uncertainty:**
  $$
  \Delta A=
  \sqrt{\langle A^2\rangle-\langle A\rangle^2}
  $$
- **Position operator:**
  $$
  \hat x=x
  $$
- **Momentum operator:**
  $$
  \hat p=-i\hbar\frac{d}{dx}
  $$
- **Momentum eigenfunction:**
  $$
  e^{ikx},\qquad p=\hbar k
  $$

> **Key viewpoint:** Quantum mechanics is formulated abstractly in Hilbert space. A wavefunction appears only after choosing a representation.
