---
title: Need for Gadget Decomposition in LWE Based Cryptosystems
date: 2025-09-27 12:13:36
mathjax: true
tags:
    - Fully Homomorphic Encryption
    - Cryptography
    - Post-Quantum Cryptography
---

Gadget Decomposition is one of the key and essential **Homomorphic Operations** in FHE cryptosystems based on the LWE problem. I have already introduced the LWE problem and how to build a basic cryptosystem around it in my previous blogs. In this blog, we will:

1. **Extend** the basic LWE cryptosystem into more generalized and mathematically robust cryptosystems
2. **Discuss** a basic homomorphic operation: **Ciphertext-Plaintext Multiplication**
3. **Introduce** the **Gadget Decomposition** operation in the context of these generalized cryptosystems

The information for this essential operation is quite scattered and inconsistent across the internet. I found it very confusing at first on how to implement the operation and, more importantly, when and where to apply it. I hope to clear this confusion with this blog. So let's get into it.

---

## LWE Cryptosystem (Quick Refresher)

**Setup**: Secret key $\mathbf{s} \in \{0, 1\}^k$, scaling factor $\Delta = \frac{q}{t}$ where $t \ll q$.

**Encryption**: Given message $m \in \mathbb{Z}\_t$, sample $\mathbf{a} \leftarrow \mathbb{Z}\_q^k$ and $e \leftarrow \chi_\sigma$:
$$\boxed{\text{LWE}\_{\mathbf{s}, \sigma}(\Delta m) = (\mathbf{a}, b) \text{ where } b = \mathbf{a} \cdot \mathbf{s} + \Delta m + e \pmod{q}}$$

**Decryption**: Given ciphertext $(\mathbf{a}, b)$:
$$\boxed{m = \left\lfloor \frac{b - \mathbf{a} \cdot \mathbf{s}}{\Delta} \right\rceil \bmod t}$$

---

### Important Case: When $t$ Does Not Divide $q$

*This is a small detour from the original purpose of the blog, but it is an important and very critical detail of such cryptosystems.*

In our previous analysis, we assumed that $t$ divides $q$. In this case, there is no upper or lower limit on the size of plaintext $m$: its value is allowed to wrap around modulo $t$ indefinitely, yet the decryption works correctly. This is because any $m$ value greater than $t$ will be correctly modulo-reduced by $t$ when we do modulo reduction by $q$ during decryption.

On the other hand, suppose that $t$ does not divide $q$. In such a case, we set the scaling factor as:
$$\Delta = \left\lfloor \frac{q}{t} \right\rfloor$$

Then, provided $q \gg t$, the decryption works correctly even if $m$ is a large value that wraps around $t$. Let's examine why this is so.

We can denote plaintext $m \bmod t$ as $m = m' + kt$, where $m' \in \mathbb{Z}_t$ and $k$ is some integer representing the modulo $t$ wrap-around value portion of $m$. Setting the plaintext scaling factor as $\Delta = \left\lfloor \frac{q}{t} \right\rfloor$, the noise-added scaled plaintext value becomes:

$$\left\lfloor \frac{q}{t} \right\rfloor \cdot m + e = \left\lfloor \frac{q}{t} \right\rfloor \cdot m' + \left\lfloor \frac{q}{t} \right\rfloor \cdot kt + e$$

By applying $m = m' + kt$:

$$= \left\lfloor \frac{q}{t} \right\rfloor \cdot m' + \frac{q}{t} \cdot kt - \left(\frac{q}{t} - \left\lfloor \frac{q}{t} \right\rfloor\right) \cdot kt + e$$

where $0 \leq \frac{q}{t} - \left\lfloor \frac{q}{t} \right\rfloor < 1$:

$$= \left\lfloor \frac{q}{t} \right\rfloor \cdot m' + qk - \left(\frac{q}{t} - \left\lfloor \frac{q}{t} \right\rfloor\right) \cdot kt + e$$

We treat the above noisy scaled ciphertext as:
$$\left\lfloor \frac{q}{t} \right\rfloor \cdot m' + qk - e' + e$$

where $e' = kt$ is the maximum possible value of $\left(\frac{q}{t} - \left\lfloor \frac{q}{t} \right\rfloor\right) \cdot kt$. We overestimate the noise caused by this term to $kt$ because the maximum value this term can become is less than $kt$.

Given the LWE decryption relation $b - \mathbf{a} \cdot \mathbf{s} \bmod q = \Delta m + e$, we can decrypt the above message by performing:

$$\left\lfloor \frac{1}{\left\lfloor \frac{q}{t} \right\rfloor} \cdot \left(\left\lfloor \frac{q}{t} \right\rfloor \cdot m' + qk - kt + e \bmod q\right) \right\rceil \bmod t$$

$$= \left\lfloor \frac{1}{\left\lfloor \frac{q}{t} \right\rfloor} \cdot \left(\left\lfloor \frac{q}{t} \right\rfloor \cdot m' - kt + e\right) \right\rceil \bmod t$$

$$= m' - \left\lfloor \frac{kt + e}{\left\lfloor \frac{q}{t} \right\rfloor} \right\rceil \bmod t$$

$$= m' \quad \text{provided } \left\lfloor \frac{kt + e}{\left\lfloor \frac{q}{t} \right\rfloor} \right\rceil < \frac{1}{2}$$

**Summary**: If we set the plaintext's scaling factor as $\Delta = \left\lfloor \frac{q}{t} \right\rfloor$ where $t$ does not divide $q$, the decryption works correctly as long as the error bound $\left\lfloor \frac{kt + e}{\left\lfloor \frac{q}{t} \right\rfloor} \right\rceil < \frac{1}{2}$ holds.

This error bound can break if:
1. The noise $e$ is too large
2. The plaintext modulus $t$ is too large
3. The plaintext value wraps around $t$ too many times (i.e., $k$ is too large)

A solution to ensure the error bound holds is that the ciphertext modulus $q$ is sufficiently large. In other words, if $q \gg t$, then the error bound will hold.

Therefore, we can generalize the formula for the plaintext's scaling factor as $\Delta = \left\lfloor \frac{q}{t} \right\rfloor$ where $t$ does not necessarily divide $q$.

---

## RLWE Cryptosystem (Quick Refresher)

*I have discussed RLWE in detail in my previous blog, so this serves as a quick refresher.*

**Setup**: Work in polynomial ring $R]\_{\langle n,q \rangle} = \mathbb{Z}\_q[x]/(x^n + 1)$ with secret key $\mathbf{s} \stackrel{\$}{\leftarrow} R_{\langle n,2 \rangle}$ and scaling factor $\Delta = \frac{q}{t}$.

**Encryption**: Given polynomial message $M \in R_{\langle n,t \rangle}$, sample $A \stackrel{\$}{\leftarrow} R_{\langle n,q \rangle}$ and $E \stackrel{\chi_\sigma}{\leftarrow} R_{\langle n,q \rangle}$:
$$\boxed{\text{RLWE}\_{\mathbf{s},\sigma}(\Delta M) = (A, B) \text{ where } B = A \cdot \mathbf{s} + \Delta M + E \pmod{R_{\langle n,q \rangle}}}$$

**Decryption**: Given ciphertext $(A, B)$:
$$\boxed{M = \left\lfloor \frac{B - A \cdot \mathbf{s}}{\Delta} \right\rceil \bmod t \in R_{\langle n,t \rangle}}$$

**Correctness**: Requires noise bound $e_i < \frac{\Delta}{2}$ for all coefficients $e_i$ of $E$.

---

*NOTE: All cryptosystems presented so far are symmetric (same key for encryption and decryption). The following section presents how to make such systems asymmetric.*

## GLWE Cryptosystem (General LWE)

As the name suggests, this is the generalized version of the LWE system that encompasses both LWE and RLWE. If you understand the construction of the two systems above, understanding this is straightforward.

First, we shall see the symmetric version, then I will present the asymmetric system, and this being the general version, one can easily map that system to the two systems above.
 
**Setup**: Work in polynomial ring $R\_{\langle n,q \rangle} = \mathbb{Z}\_q[x]/(x^n + 1)$ with secret key list $\\{S\_i\\}\_{i=0}^{k-1} \stackrel{\$}{\leftarrow} R\_{\langle n,2 \rangle}^k$ and scaling factor $\Delta = \frac{q}{t}$.

**Encryption**: Given polynomial message $M \in R\_{\langle n,t \rangle}$, sample $\\{A\_i\\}\_{i=0}^{k-1} \stackrel{\$}{\leftarrow} R\_{\langle n,q \rangle}^k$ and $E \stackrel{\chi\_\sigma}{\leftarrow} R\_{\langle n,q \rangle}$:
$$\boxed{\text{GLWE}\_{S,\sigma}(\Delta M) = (\\{A\_i\\}\_{i=0}^{k-1}, B) \text{ where } B = \sum\_{i=0}^{k-1}(A\_i \cdot S\_i) + \Delta M + E \pmod{R\_{\langle n,q \rangle}}}$$

**Decryption**: Given ciphertext $(\\{A\_i\\}\_{i=0}^{k-1}, B)$:
$$\boxed{M = \left\lfloor \frac{B - \sum\_{i=0}^{k-1}(A\_i \cdot S\_i)}{\Delta} \right\rceil \bmod t \in R\_{\langle n,t \rangle}}$$

**Correctness**: Requires noise bound $e\_i < \frac{\Delta}{2}$ for all coefficients $e\_i$ of $E$.

**Connection to LWE/RLWE**:
- When $n=1$ (polynomials become scalars), GLWE reduces to LWE
- When $k=1$ (single polynomial), GLWE reduces to RLWE

---

Now, let's see the asymmetric version. 

### Public-Key GLWE
The basic idea here is that a part which is used during the encryption stage is pre-computed during the setup stage and released as the public key. During the encryption stage, the encryptor will have to add some additional noise of their own. Let's see the mathematics of the system to make things clearer. 

#### Setup

- The scaling factor: $ \Delta = \lfloor \frac{q}{t} \rfloor $.
- The secret key: $$\mathbf{S} = \\{S\_i \\}\_{i = 0}^{k - 1} \stackrel{\$}{\leftarrow} R\_{\langle n,2 \rangle}^k$$
- Public key pair $(PK\_1, \mathbf{PK\_2}) \in R\_{\langle n,q \rangle}^{k + 1}$ is to be generated as follows: $$ \mathbf{A} = \\{A\_i\\}\_{i = 0}^{k - 1} \stackrel{\$}{\leftarrow}R\_{\langle n,q \rangle}^{k} \text{,       } E \stackrel{\sigma}{\leftarrow}R\_{\langle n,q \rangle}$$ $$\boxed{PK\_1 = \mathbf{A} \cdot \mathbf{S} + E \in R\_{\langle n,q \rangle}}$$ $$\boxed{\mathbf{PK\_2} = \mathbf{A} \in R\_{\langle n,q \rangle}^k}$$



#### Encryption

- **Input:** $M \in R\_{\langle n,t \rangle},  U \stackrel{\$}{\leftarrow} R\_{\langle n,2 \rangle},  E\_1 \stackrel{\$}{\leftarrow} R\_{\langle n,q \rangle},  \mathbf{E\_2} \stackrel{\$}{\leftarrow} R^{k}\_{\langle n,q \rangle}$


- Scale up the plaintext message: $M \rightarrow \Delta M \in R\_{\langle n,q \rangle}$.

- Perform the following computations: $$B = PK\_1 \cdot U + \Delta M + E\_1 \in R\_{\langle n,q \rangle}$$ $$\mathbf{D} = \mathbf{PK\_2}\cdot U + \mathbf{E\_2} \in R\_{\langle n,q \rangle}^k$$

- With the computations above, we get our final ciphertext: $$\boxed{\text{GLWE}\_{S, \sigma}(\Delta M) = (\mathbf{D}, B) \in R\_{\langle n,q \rangle}^{k+1}}$$


#### Decryption
- **Input:** A GLWE ciphertext $C = (\mathbf{D}, B) \in R_{\langle n,q \rangle}^{k+1}$ and the secret key $\mathbf{S}$.
- Cancel out the mask by computing the inner product with the secret key and subtracting it from $B$:
$$B - \mathbf{D} \cdot \mathbf{S} = \Delta M + E_{all} \in R_{\langle n,q \rangle}$$ 
- Scale the result down by $\Delta$ and round to the nearest integer to remove the scaling factor and noise. This recovers the original plaintext message.$$\boxed{M' = \left\lfloor \frac{B - \mathbf{D} \cdot \mathbf{S}}{\Delta} \right\rceil \pmod t \in R_{\langle n,t \rangle}}$$
- **Correctness Condition:** For the decryption to be successful, every coefficient $e_i$ of the total noise polynomial $E_{all}$ must satisfy the condition $|e_i| < \frac{\Delta}{2}$.


## GLev Cryptosystem

GLev is a "leveled" homomorphic encryption scheme built upon GLWE. A GLev ciphertext isn't a single entity, but rather a **list of several GLWE ciphertexts**. Each of these "level" ciphertexts encrypts the same underlying plaintext message, but uses a different, progressively smaller scaling factor. This structure is key for managing noise in homomorphic computations.

---

* **Setup**: In addition to the standard GLWE parameters ($n, q, t, k, \mathbf{S}$), GLev introduces two new ones:
* **Decomposition Base** $\beta$: An integer used to define the different scaling levels. It should be chosen such that $t < \beta < q$.
* **Number of Levels** $l$: The total number of GLWE ciphertexts that will make up a single GLev ciphertext.

From these, a list of scaling factors is derived for each level $i \in [1, l]$:
$$\Delta_i = \frac{q}{\beta^i}$$

**Encryption**: To encrypt a message $M \in R_{\langle n,t \rangle}$, we generate $l$ separate public-key GLWE ciphertexts. The $i$-th ciphertext, $C_i$, encrypts the message $M$ using the scaling factor $\Delta_i$.

The complete GLev ciphertext is the collection of all these level ciphertexts:
$$\boxed{\text{GLev}\_{S,\sigma}^{\beta,l}(M) = \{ C\_i = \text{GLWE}\_{S,\sigma}(\Delta\_i M) \}\_{i=1}^{l}}$$
where each ciphertext is $C\_i = (\mathbf{D}\_i, B\_i) \in R\_{\langle n,q \rangle}^{k+1}$.

**Decryption**: To decrypt a GLev ciphertext, you can choose to decrypt any specific level $i$. Decryption follows the standard GLWE procedure, but you **must** use the scaling factor $\Delta\_i$ that corresponds to the level you are decrypting.

Given the $i$-th level ciphertext $C\_i = (\mathbf{D}\_i, B\_i)$:
$$\boxed{M' = \left\lfloor \frac{B\_i - \mathbf{D}\_i \cdot \mathbf{S}}{\Delta\_i} \right\rceil \pmod t \in R\_{\langle n,t \rangle}}$$

**Connection to Lev/RLev**: Just like GLWE, GLev is a generalized construction that unifies other schemes:
* When $n=1$ (polynomials are scalars), GLev becomes the **Lev** cryptosystem.
* When $k=1$ (the secret key is a single polynomial), GLev becomes the **RLev** cryptosystem.


*NOTE: Keep this in mind, it's going to be the key concept later when I introduce the need for gadget decomposition.*
*Also, there is another generalization, i.e., the GGSW Cryptosystem, which is nothing but a list of GLev Ciphertexts, similar to how GLev is nothing but a list of GLWE ciphertexts, but GGSW is not required to fulfill the purpose of the blog, so I am going to skip it. Curious people might refer to: [TFHE Deep Dive](https://www.zama.ai/post/tfhe-deep-dive-part-1) written by Ilaria Chillotti*



## Homomorphic Operation (`ct-pt multiplication`)

I have made it clear in my previous blogs, that heart of all the modern FHE schemes is this LWE problem only, hence, the naive system which we built, i.e., GLWE system is obviously Fully Homomorphic. In fact, all of the core **Homomorphic Operations** and properties, can be demonstrated over this system, then be easily mapped over to specific schemes like TFHE, CKKS, etc. For this blog, I am not going over through all the operations. I will only present one operation, i.e., **Ciphertext-Plaintext Multiplication** which will guide us towards introducing **gadget decomposition**. So let's dive right into it.

### Ciphertext-Plaintext Multiplication
Consider the GLWE ciphertext: $$C = \text{GLWE}\_{S, \sigma}(M) = (A\_1, \cdots , A\_{k - 1}, B) \in R\_{\langle n, q \rangle}^{k + 1}$$

Now, consider the following plaintext polynomial $\Phi$ : $$\Phi = \sum\_{i=0}^{n-1}(\Phi\_i \cdot X\_i) \in R\_{\langle n, q \rangle}$$


Now, if we want to homomorphically multiply this plaintext to the ciphertext, such that after decrypting the resulting ciphertext, we should get the original plaintext message multiplied with the second plaintext message ($\Phi$). Note that we are never encrypting the second plaintext. Now it can be easily shown with simple mathematics that the following is true: 

$$ \Phi \cdot \text{GLWE}\_{S, \sigma}(\Delta M) = \Phi \cdot (\\{A\_i\\}\_{i = 0}^{k - 1}, B) = (\\{\Phi \cdot A\_{i}\\}\_{i = 0}^{k - 1}, \Phi \cdot B) = \text{GLWE}\_{S, \sigma}(\Delta(M\cdot \Phi))$$



## Gadget Decomposition for Limiting Noise Growth

If you have done some calculations to verify the equation for (`ct-pt multiplication`), you must have noticed that the **new error** term is **scaled-up** by $|\Phi|$. This can potentially be a huge problem, unless we have $t$ much smaller compared to $q$, which is not always the case, and even if we were to increase $q$, the computational cost would increase too much to even consider that, as we already have to take a large $q$, so increasing $q$ is not a sensible option, practically speaking. 

The way we are going to tackle the problem is that we will decompose our plaintext into a series of smaller moduli terms, then have **separate ct-pt multiplied GLWE encryptions** for each of those smaller moduli terms. Mathematically, the decomposition would look like this: $$\Phi = \Phi\_1\cdot \frac{q}{\beta^1} + \Phi\_2\cdot \frac{q}{\beta^2} + \cdots + \Phi\_l \cdot \frac{q}{\beta^l} \rightarrow \text{decomp}^{\beta, l}(\Phi) = (\Phi\_1, \cdots, \Phi\_l)$$

Now, for the given GLWE ciphertext, we can get the following GLev ciphertext: $$ \text{GLev}^{\beta,l}\_{S,\sigma}(\Delta M) = \\{ \text{GLWE}\_{S,\sigma}(\Delta M \tfrac{q}{\beta^1}), \dots, \text{GLWE}\_{S,\sigma}(\Delta M \tfrac{q}{\beta^l}) \\} $$

Next, consider the operation below: $$ \text{decomp}^{\beta, l}(\Phi)\cdot \text{GLev}\_{S, \sigma}^{\beta, l}(\Delta M)$$ $$= \sum\_{i=1}^{l}(\Phi\_i \cdot \text{GLWE}\_{S, \sigma}(\frac{q}{\beta^i}\Delta M))$$ $$ = \sum\_{i=1}^{l}(\text{GLWE}\_{S, \sigma}(\frac{q}{\beta^i}\Delta M\cdot \Phi\_i))$$ $$= \text{GLWE}\_{S, \sigma}(\sum\_{i=1}^{l}(\frac{q}{\beta^i}\Delta M\cdot \Phi\_i))$$ $$ = \text{GLWE}\_{S, \sigma}(\Delta M\cdot\sum\_{i = 1}^{l}(\frac{q}{\beta^i}\cdot \Phi\_i)) = \text{GLWE}\_{S, \sigma}(\Delta M \cdot \Phi)$$


### Why not Base Decomposition
This proves that the evaluation is nothing but the ciphertext-multiplication. I mentioned one benefit of doing this **gadget-decomposition** earlier by arguing that each resulting `ct-pt multiplication` has less noise growth compared to the original non-decomposed multiplication. But one question might be coming up in your mind: why gadget decomposition? We can also do a base decomposition, which is dividing the plaintext into uniform modulus components, that way we won't have to worry about more noise growth in some components compared to others as each resulting multiplication would have the same resulting noise growth. 

Base decomposition would obviously work with the benefit mentioned above, but at the cost of number of computations. See, when we decompose the plaintext among different gadgets across a base, we end up with fewer terms compared to decomposing them across the same base. Fewer terms imply fewer multiplication operations, and multiplication is not an easy elementary operation. 

### Mathematics behind selection of $\beta$
Obviously, our main goal in selecting $\beta$ should be that the resulting ciphertext doesn't explode with noise. The main risk does not come from the individual multiplications, but rather when we accumulate the increased noise across the components. If each initial GLWE ciphertext in the GLev ciphertext which we considered has a noise polynomial $\Phi\_i$, the final noise polynomial can be written as $$E\_{\text{final}} = \sum\_{i = 1}^l (\Phi\_i \cdot E\_i)$$

Now, keeping the largest absolute coefficient in this final noise polynomial smaller than half of the scaling factor, $\Delta$, should be enough to ensure that the resulting noise is well within the tolerable limit. For this I am just going to use the infinity-norm notation ($ \lVert \cdot \rVert_{\infty} $), which just gives us the largest absolute coefficient. So the goal is to satisfy: $$\lVert E\_\text{final} \rVert\_{\infty} < \frac{\Delta}{2}$$

The plan now is to find an upper-bound, a worst case scenario if you will, for the size of $\lVert E\_\text{final} \rVert\_{\infty}$, then keep that well below the tolerable limit. Let's break down some terms and concepts which we will be using first: 

- **Decomposition Bound**($B\_\text{decomp}$): the coefficients of our plaintext parts $\Phi\_i$ are small, bounded by $\lVert \Phi\_i \rVert\_{\infty} < \frac{\beta}{2}$, this bound is referred to as $B\_\text{decomp}$.

- **Initial Noise Bound**($B\_\text{noise}$): noise polynomials $E_i$ are sampled from a tight distribution, so their coefficients are bounded by some value $B\_\text{noise}$.

- **Polynomial Multiplication Bound**: when we multiply two polynomials $P$ and $Q$, the resulting coefficients are bounded: $\lVert P \cdot Q \rVert_{\infty} \leq n \cdot \lVert P \rVert_{\infty}\cdot \lVert Q \rVert_{\infty}$.

Now, let's build our worst-case noise estimate step-by-step:

1.  **Start with the final noise term:**
    $$\lVert E\_{final}\rVert\_\infty = \left\lVert \sum\_{i=1}^{l} \Phi\_i \cdot E\_i \right\rVert\_\infty$$

2.  **Apply the triangle inequality** (the norm of a sum is at most the sum of the norms):
    $$\le \sum_{i=1}^{l} \lVert\Phi_i \cdot E_i\rVert_\infty$$

3.  **Use the polynomial multiplication bound on each term:**
    $$\le \sum\_{i=1}^{l} n \cdot \lVert\Phi\_i\rVert\_\infty \cdot \lVert E\_i\rVert\_\infty$$

4.  **Finally, substitute our bounds** for the decomposed parts ($B_{decomp}$) and the initial noise ($B_{noise}$):
    $$\le \sum_{i=1}^{l} n \cdot B_{\text{decomp}} \cdot B_{\text{noise}} = l \cdot n \cdot B_{\text{decomp}} \cdot B_{\text{noise}}$$

This gives us our upper bound on the final noise. Now, we just force this bound to be small enough for decryption to work:
$$l \cdot n \cdot B_{\text{decomp}} \cdot B_{\text{noise}} < \frac{\Delta}{2}$$

This is the punchline. By substituting $B_{\text{decomp}} = \beta/2$ and our usual scaling factor $\Delta = \lfloor q/t \rfloor$, we get the master inequality that governs our choice of $\beta$:

$$\boxed{l \cdot n \cdot \frac{\beta}{2} \cdot B_{\text{noise}} < \frac{\lfloor q/t \rfloor}{2}}$$

Or, simplifying it for clarity:

$$\Large l \cdot n \cdot \beta \cdot B_{\text{noise}} < \lfloor q/t \rfloor$$


This final inequality is the key to selecting sound parameters. Think of it like this:

-   **Right side ($\lfloor q/t \rfloor$)**: This is our total **noise budget**. It's the maximum amount of noise the system can tolerate. To get a bigger budget, we need to increase the ratio of $q$ to $t$.
-   **Left side ($l \cdot n \cdot \beta \cdot B_{noise}$)**: This is the **total noise growth** from our homomorphic multiplication.

To ensure our scheme works, the **total noise growth must be less than the noise budget**. This formula perfectly captures the trade-offs we have to make. For instance, if we increase our decomposition base $\beta$, the noise growth per level increases, but the number of levels $l$ decreases. Finding the right balance is what FHE parameter selection is all about.



### Quick Implementation
Ok, now let's see a quick implementation of this concept. I have presented only the relevant C++ code snippets, the full code can be found at this [repo](https://github.com/DA1729/gadget_decomp_blog.git).

Our basic data structures: 
- `poly`: a `vec<int64_t>` representing a polynomial.
- `GLWE_ciphertext`: a struct containing a vector of polynomials `D` and a single polynomial `B`.
- `GLev_ciphertext`: a vector of `GLWE_ciphertext`.


#### Gadget Decomposition Function
This function, as the name suggests, would perform the gadget decomposition of the referenced polynomial `p` and return us a vector (length `l`) of decomposed polynomials. The implementation is using the standard rounding method to find the closest coefficients for each component. 

``` cpp
vector<poly> gadget_decompose(const poly &p, int64_t q, int64_t beta, int l) {
    size_t n = p.size();
    vector<poly> decomp(l, poly(n, 0));
    poly current_rem = p;

    for (int i = 0; i < l; ++i) {
        int64_t g = 1;
        for(int j = 0; j < i + 1; ++j) g *= beta;
        g = q / g;

        if (g == 0) continue;

        for (size_t j = 0; j < n; ++j) {
            int64_t centered_rem = center_rep(current_rem[j], q);
            
            int64_t coeff = round((double)centered_rem / g);

            decomp[i][j] = coeff;
            
            __int128 rem_update = (__int128)coeff * g;
            current_rem[j] = modq(current_rem[j] - (int64_t)rem_update, q);
        }
    }
    return decomp;
}
```

#### External Product
The core of the homomorphic operation (`ct-pt multiplication`). Function takes the decomposed plaintext `decomp_phi` and the `GLev Ciphertext`, then computes the sum of the component-wise products. The result is a single, final `GLWE_ciphertext`.

<div style="font-size: 0.8em;">

```cpp
GLWE_ciphertext external_product(const vector<poly> &decomp_phi, const GLev_ciphertext &c_glev, int64_t q) {
    size_t n = c_glev[0].B.size();
    int k = c_glev[0].D.size();
    int l = c_glev.size();

    GLWE_ciphertext result;
    result.B = poly(n, 0);
    result.D.assign(k, poly(n, 0));

    for (int i = 0; i < l; ++i) {
        // C_i_scaled = decomp_phi[i] * c_glev[i]
        poly b_scaled = poly_scalar_mul(c_glev[i].B, decomp_phi[i][0], q);

        vector<poly> d_scaled(k);
        for(int j=0; j<k; ++j) {
            d_scaled[j] = poly_scalar_mul(c_glev[i].D[j], decomp_phi[i][0], q);
        }

        result.B = poly_add(result.B, b_scaled, q);
        for (int j = 0; j < k; ++j) {
            result.D[j] = poly_add(result.D[j], d_scaled[j], q);
        }
    }
    return result;
}
```

</div>

*Note: Note: For simplicity in this example, `p` is treated as a constant polynomial, so `decomp_phi[i]` only has one non-zero coefficient at index 0.

#### Result
Running the full code (available on my GitHub), we get: 
```shell
------ Parameters ------
n: 1024, q: 2^32, t: 256, k: 2
beta: 1024, l: 3
Original Message M(x) = 5
Plaintext Multiplier Φ(x) = 12
Expected Result (M * Φ) = 60

------ Operations ------
Encrypting M=5 into a GLev ciphertext...
Decomposing Φ=12 into (Φ_1, Φ_2, ...)...
Performing homomorphic external product...
Decryption of the final ciphertext...
------ Noise Analysis ------
Correctness requires noise < q/(2t) = 8388608

Noise from Gadget Method: 1668
Noise from Naive Method: 8232

The gadget-based multiplication resulted in noise ~4x smaller than the naive approach!
```

The results are obviously matching the expectations. 


## Conclusion
This brings us to the end. In this blog, we first quickly revisited LWE, RLWE, then we designed two generalized systems, GLWE and GLev. Then, studying a very specific homomorphic operation(`ct-pt multiplication`), we made us realize the need for the gadget decomposition. One can already appreciate the benefits (obviously we have the tradeoffs, but that's alright) we get using this technique. This concept is very crucial when we build more **complex evaluation systems** and this appears frequently when we deal with even more complex and crucial operations like **bootstrapping**, **key-switch**, **modulus switching**, etc.

peace. da1729

## References 
*I have not used the standard referencing format, but included all the relevant and important references*

- TFHE Deep Dive Series by Ilaria Chillotti [TFHE-deep-dive](https://www.zama.ai/post/tfhe-deep-dive-part-1)
- A Fully Homomorphic Encryption Scheme by Craig Gentry (2009)
- Homomorphic Encryption for Arithmetic of Approximate Numbers by Cheon et al
- TFHE: Fast Fully Homomorphic Encryption over the Torus by Ilaria et al
- The Beginner's Textbook for Fully Homomorphic Encryption by Ronny Ko
