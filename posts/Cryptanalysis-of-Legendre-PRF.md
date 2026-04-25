---
title: "Cryptanalysis of the Legendre PRF"
date: 2026-04-01 14:00:41
tags:
    - Cryptography
    - Cryptanalysis
---

The search for "MPC-friendly" primitives has led cryptographers back to some very classical number theory. In the world of Secure Multi-Party Computation (MPC), the cost of a protocol is often dominated by its "multiplicative depth," the number of sequential multiplications required. This makes traditional symmetric primitives like AES or SHA-256, which rely heavily on non-linear S-boxes, quite expensive to compute securely. Enter the **Legendre PRF**, a construction so simple it almost feels like a trick, yet it has become a serious candidate for usage in the Ethereum 2.0 blockchain and other MPC-heavy architectures.

However, as we often find in cryptanalysis, extreme simplicity is frequently a double-edged sword. The very algebraic structure that makes the Legendre PRF efficient also opens the door to a variety of "virtual" attacks that exploit the deep tension between addition and multiplication in finite fields.

## The Foundation: Quadratic Residues and the Legendre Symbol

At the heart of this PRF lies the Legendre symbol, $(\frac{a}{p})$. Originally proposed as a Pseudorandom Generator (PRG) by Damgård in 1988, it was later suggested as a Pseudorandom Function (PRF) by Grassi et al. for its efficiency in multiparty settings.

Mathematically, for an odd prime $p$ and an integer $a$, the symbol $(\frac{a}{p})$ checks if $a$ is a "perfect square" modulo $p$:
$$ \left(\frac{a}{p}\right) = \begin{cases} 1 & \text{if } a = b^2 \pmod{p} \text{ for some } b \in \mathbb{F}\_p^\times \\\\ 0 & \text{if } a \equiv 0 \pmod{p} \\\\ -1 & \text{otherwise} \end{cases} $$

The most critical property for our analysis is that the Legendre symbol is **multiplicative**:
$$ \left(\frac{ab}{p}\right) = \left(\frac{a}{p}\right) \left(\frac{b}{p}\right) $$

This property is what makes the symbol a "homomorphism" from the multiplicative group of the field to the set $\\{1, -1\\}$. In the world of MPC, this is a dream, but in the world of cryptanalysis, this is the "mathematical loophole."

## Why Does it Look Random? (The Weil Bound)

Before we trust a sequence for cryptography, we must be convinced it actually behaves like unpredictable noise. If you calculate the Legendre symbol for consecutive integers $1, 2, 3, \dots$, you get a string of $+1$s and $-1$s. The **Weil Bound** provides the theoretical guarantee that this sequence is virtually indistinguishable from a series of fair coin flips.

Specifically, the bound states that the number of occurrences of any fixed pattern of length $l$ among the integers $1, 2, \dots, p-1$ is:
$$ \frac{p}{2^l} + O(\sqrt{p}) $$

To get an intuition for this, imagine you are looking for the pattern $(+1, -1, -1)$. In a perfectly random sequence, the chance of any 3-bit pattern is $1/2^3 = 1/8$. Thus, over a range of $p$ numbers, you expect $p/8$ occurrences. The $O(\sqrt{p})$ term is the "margin of error." Since the square root grows much slower than $p$ itself, as the prime becomes cryptographically large, this error becomes a negligible fraction of the total count.

<p align="center">
  <img src="/images/legendre_plot.png" width="90%">
</p>

As visualized in the plot above, while the error fluctuates pseudo-randomly, it is strictly and aggressively constrained by the $\sqrt{p}$ boundary. This convergence ensures that an attacker cannot easily predict the next bit in the sequence based on a local pattern.

## The PRF Construction: Hidden Shifts

The **Legendre PRF**, $L\_k(x)$, is defined by taking this public sequence and adding a "secret shift" $k$. Since $p$ is public, anyone can calculate the sequence starting from zero. The secret $k$ acts as a hidden starting point on a massive, public ribbon of bits.

To map the mathematical symbols to computer-friendly bits, we use a bit-mapping trick:
$$ L\_k(x) = \left\lfloor \frac{1}{2} \left( 1 - \left(\frac{k+x}{p}\right) \right) \right\rfloor $$

If the symbol is $+1$ (a square), the formula yields $(1-1)/2 = 0$. If the symbol is $-1$ (not a square), it yields $(1-(-1))/2 = 1$. Thus, the PRF outputs a sequence of $0$s and $1$s that inherits the pseudorandomness of the Legendre symbol.

## The Linear Attack: Collision Search

The first major attack on the "linear" (Degree-1) Legendre PRF was proposed by Khovratovich. It relies on a **memoryless collision search**. 

The attacker's goal is to find the secret shift $k$ by looking for a collision between the real PRF and a "dummy" version. By Assumption 1 in the literature, a sequence of bits of length $m = \lceil \log\_2 p \rceil$ acts as a unique fingerprint for the key. The attacker defines:
1. $f\_1(x) = L\_k(x + [m])$ : The real sequence (queried from the oracle).
2. $f\_2(b) = L\_0(b + [m])$ : The public sequence starting at $b$ (computed locally).

A collision $f\_1(a) = f\_2(b)$ implies $L\_k(a + [m]) = L\_0(b + [m])$, which mathematically translates to $a + k = b$. Instantly, the secret key is revealed: $k = b - a$. Due to the Birthday Paradox, this requires roughly $O(\sqrt{p} \log p)$ queries and computations.

## The "Virtual Expansion" Breakthrough

The paper we are examining introduces a massive improvement. The authors realized that we can exploit the multiplicative property to turn $M$ queries into $M^2$ "virtual" fingerprints.

Consider a set of $M$ queries to the PRF oracle. In a standard attack, you have $M$ pieces of data. But the authors use **Arithmetic and Geometric progressions** to expand this. By Lemma 1, for any $b \in \mathbb{F}\_p^\times$:
$$ L\_{k/b}(a/b + [m]) = (l(b), \dots, l(b)) \oplus L\_k(a + b[m]) $$

Think about what this means: by changing our "stride" (the step size $b$) and applying the multiplicative property, we can mathematically predict the output of the PRF for a *different* key $k/b$ using the data we already have for key $k$.

By iterating over many values of $b$, we "squish" and "stretch" the original $M$ outputs to generate $M^2/m$ virtual fingerprints. This exponentially increases the size of our lookup table without costing a single additional query to the oracle. Consequently, the time complexity of finding a collision drops from $O(p/M)$ to $O(p \log^2 p / M^2)$. This attack was used to break several concrete challenges proposed by the Ethereum Foundation, proving that instances with 44 and 54 bits of expected security were much more vulnerable than initially assumed.

## Higher-Degree Generalizations and the Möbius Trap

To increase security, one might consider a **Degree-d** Legendre PRF, where the key $k$ is a vector of $d$ coefficients for a polynomial $P\_k(x)$:
$$ L\_k(x) = l\left( x^d + \sum\_{i=0}^{d-1} k\_{i+1} x^i \right) $$

While this seems harder, it actually introduces a devastating vulnerability to **Möbius Transformations**. A Möbius transformation is a rational function:
$$ f(x) = \frac{cx+d}{ax+b} \pmod{p} $$

As it turns out, the Legendre symbol of such a fraction behaves almost perfectly:
$$ L\_k\left( \frac{cx+d}{ax+b} \right) = \pm L\_{k'}(x) \cdot L\_0(cx+d) $$

where $k'$ is a brand new, transformed key. This creates an interconnected web of virtual keys. For the linear PRF, the "unknown sign" ($\pm 1$) makes this attack difficult to coordinate. However, for degree $d \geq 3$, we can calculate **absolute invariants** of the secret polynomial. 

These invariants are mathematical fingerprints that do not change under Möbius shifts. By calculating these, an attacker can identify equivalent polynomials without ever needing to resolve the unknown $\pm 1$ sign. Research by Kaluderovic et al. has shown that for $d \geq 3$, the key can be recovered in $O(1)$ time after a precomputation phase. This essentially kills the security of any Legendre PRF generalization with a degree higher than 2.

## The Security Verdict: Addition vs. Multiplication

The ultimate security of the Legendre PRF rests on a deep mathematical tension. The symbol is perfectly multiplicative: $(\frac{ab}{p}) = (\frac{a}{p})(\frac{b}{p})$. But the secret key $k$ is applied **additively**: $(\frac{x+k}{p})$. 

Addition and multiplication modulo a prime do not "play nicely" together inside a radical. There is no known general formula to simplify $(\frac{a+b}{p})$. This "algebraic friction" is what protects the secret key. 

Is it insecure? Not exactly. The linear version (Degree 1) remains secure for real-world applications like Ethereum 2.0, provided the prime $p$ is large (e.g., 256 or 384 bits). Even if our "virtual expansion" reduces the security from $2^{256}$ to $2^{128}$, the cost of $2^{128}$ operations is still beyond any current or foreseeable computational capacity. However, the higher-degree versions are effectively dead, and the quantum threat, while currently only theoretical, reminds us that our " MPC-friendly" shortcuts are always under the microscope of cryptanalysts.

## References

1. **Beullens, W., Beyne, T., Udovenko, A., & Vitto, G. (2020).** *Cryptanalysis of the Legendre PRF and generalizations.* IACR Transactions on Symmetric Cryptology, 2020(1), 313–330.
2. **Khovratovich, D. (2019).** *Key recovery attacks on the Legendre PRFs within the birthday bound.* Cryptology ePrint Archive, Report 2019/862.
3. **Kaluđerović, N., Kleinjung, T., & Kostić, D. (2020).** *Improved key recovery on the legendre prf.* Cryptology ePrint Archive, Report 2020/098.
4. **Damgård, I. (1990).** *On the randomness of Legendre and Jacobi sequences.* In Conference on the Theory and Application of Cryptographic Techniques (pp. 163-172). Springer, Berlin, Heidelberg.
5. **Grassi, L., Rechberger, C., Rotaru, D., Scholl, P., & Smart, N. P. (2016).** *MPC-friendly symmetric key primitives.* In Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security (pp. 430-443).
6. **van Dam, W., & Hallgren, S. (2000).** *Efficient quantum algorithms for shifted quadratic character problems.* arXiv preprint quant-ph/0011067.
7. **Weil, A. (1948).** *On some exponential sums.* Proceedings of the National Academy of Sciences, 34(5), 204.

peace. da1729

