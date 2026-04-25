---
title: Gentry-Lee Encoding for Efficient Matrix FHE
date: 2025-12-04 10:15:37
tags:
    - Cryptography
    - Fully Homomorphic Encryption
---

Craig Gentry (yeah the man himself) and Yongwoo Lee proposed a new FHE scheme quite recently (October 2025). The special thing about this new scheme is its algorithmic efficiency towards **Homomorphic Matrix Arithmetic**, which can be groundbreaking in Privacy-Preserving Machine Learning. So let's dive right into it. I am assuming that the reader is already familiar with the LWE problem and its ring variant (RLWE) and how we proceed to build Fully Homomorphic Cryptosystems around it.

In CKKS, we usually pack a long vector of numbers into a polynomial. If we want to multiply two matrices, we have to perform and deal with awkward rotations (automorphisms) and corresponding key-switches. I went through the trouble before October for our team (Aurva) project, you can find [here](https://github.com/MrRoy09/AOHW_327_Aurva_Student), which was selected as one of the winners at AMD Open Hardware Competition 2025. In this project, we accelerated the convolution operation over the CKKS Encryption.

The key innovation proposed by Gentry and Lee is that they change this "container" from a 1D line (polynomial in $X$) to a **multidimensional cube** (polynomial in $X, Y, W$). Here, 
- $X$ axis: represents the **rows** of the matrix. 
- $Y$ axis: represents the **columns** of the matrix. 
- $W$ axis: represents the **batch** (process multiple matrices at once).

By encoding the matrix into $X$ and $Y$ coordinates, matrix multiplication becomes a natural byproduct of polynomial algebra.

## Encoding Matrices
The goal is to turn a matrix $M$ (a grid of numbers) into a polynomial $m(X, Y, W)$. The authors use the **Evaluation Representation** (similar to DFT/FFT). Imagine the polynomial as a wave. You can define this wave by its **coefficients** (numbers in front of $X, Y$) or by **its values at specific points (roots of unity)**.

The paper defines the polynomial such that when evaluated at specific "coordinates" (roots of unity), spits out the entry $M\_{j, k}$ of the matrix $M$.

For this, we use a **special ring** $R^{'}\_q$: $$R^{'}\_q = \mathbb{Z}\_q\[i\]\[X, Y, W\]\langle X^n - i, Y^n - i, \Phi\_p(W) \rangle$$.

Further, we use: 
- $\zeta$: roots of unity. The specific roots used are primitive 4n-th roots such that $\zeta ^ n = i$.

Now, the polynomial $m$ encodes a matrix $M^{(l)}$ if: $$m(\zeta\_j, \zeta\_k, \eta\_l) = M^{l}\_{jk}$$, where: 
- $\zeta\_j$ is the coordinate for row $j$.
- $\zeta\_k$ is the coordinate for row $k$.
- $\eta\_l$ is the coordinate for $l$-th matrix in the batch.

This encoding process should look something like this in the implementation: 

```cpp
poly_XY encode(const std::vector<std::vector<int64_t>>& M, gaussian_int zeta) {
    poly_XY m;
    int64_t n_inv_int = mod_inverse(N * N);
    gaussian_int scale(n_inv_int, 0);

    for (int u = 0; u < N; ++u) {
        for (int v = 0; v < N; ++v) {
            
            gaussian_int sum(0, 0);
            
            for (int j = 0; j < N; ++j) {   
                for (int k = 0; k < N; ++k) {
                    
                    gaussian_int val(M[j][k], 0);
                    
                    gaussian_int z_j = zeta;
                    
                    int pwrX = (j * u); 
                    int pwrY = (k * v);
                    int cycle = 4 * N;
                    int inv_pwrX = (cycle - (pwrX % cycle)) % cycle;
                    int inv_pwrY = (cycle - (pwrY % cycle)) % cycle;

                    gaussian_int zx(1,0), zy(1,0);
                    for(int iter=0; iter<inv_pwrX; ++iter) zx = zx * zeta;
                    for(int iter=0; iter<inv_pwrY; ++iter) zy = zy * zeta;
                    
                    sum = sum + (val * zx * zy);
                }
            }
            m.coeffs[u][v] = sum * scale;
        }
    }
    return m;
}
```

Let's consider a toy example now for better understanding. For simplicity, I am assuming singular batch dimension. Let's say that we want to encode the matrix:

$$
M = \begin{pmatrix}
    1 & 2 \\\\
    3 & 4
\end{pmatrix}
$$

Here, $M\_{0, 0} = 1$, $M\_{0, 1} = 2$, $M\_{1, 0} = 3$, $M\_{1, 1} = 4$. Now, in this case, we have $n = 2$, so we need to find $\zeta$s such that $\zeta^2 = i$. Let's call the two values that we get $\zeta\_0$ and $\zeta\_1$ respectively. $\zeta\_0$ represents column and row 0 and likewise for column and row 1.

With this, we need a polynomial $m(X, Y)$ such that:

$$ m(\zeta\_0, \zeta\_0) = 1 $$

$$ m(\zeta\_0, \zeta\_1) = 2 $$

$$ m(\zeta\_1, \zeta\_0) = 3 $$

$$ m(\zeta\_1, \zeta\_1) = 4 $$

Using the inverse FFT (DFT), or simply by solving the system of linear equations, we solve for the coefficients $c\_{xy}$ in $m(X, Y) = \Sigma c\_{xy}X^xY^y$.

For the roots, $\zeta\_0 = \sqrt{i} = w$ and $\zeta\_1 = -\sqrt{i} = -w$, we end up with the following polynomial which you can verify yourself: 

$$ m(X, Y) = \frac{5}{2} - \frac{3}{2w} X - \frac{1}{2w} Y $$

Conceptually speaking, our matrix is now "smeared" across the coefficients of the polynomial. This allows us to perform operations like rotation by simply multiplying the variable $X$ or $Y$, rather than permuting a vector.

---

**NOTE:** I have not at all cared about the implementation and computational efficiency in the toy example and pseudo-code. I just naively followed the mathematics behind the encoding process. One can definitely look into optimizations like precomputed twiddle factors, FFT, etc.

---

Let's end this part right here, in the next part, I will be covering the encryption process, which is mostly similar to all the other schemes following the RLWE problem, but here we shall see one key innovation, where the sampled secret key does not depend on the $Y$ axis allowing cheap column operations and efficient key-switching.

peace. da1729

PS: I am still an undergraduate student trying to make sense of all the mathematics and implementations of these post-quantum and FHE algorithms, if you find any sorts of errors, please point it out over at [twitter](https://x.com/sp0oky_daksh) or [mail](daksh_p@ph.iitr.ac.in).

## References
@misc{cryptoeprint:2025/1935,
      author = {Craig Gentry and Yongwoo Lee},
      title = {Fully Homomorphic Encryption for Matrix Arithmetic},
      howpublished = {Cryptology {ePrint} Archive, Paper 2025/1935},
      year = {2025},
      url = {https://eprint.iacr.org/2025/1935}
}
