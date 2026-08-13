---
title: "Mechanistic Interpretability, Part 1: The Shape of the Computation"
date: 2026-08-13
tags:
    - Mechanistic Interpretability
    - Transformers
    - Machine Learning
---

> *This is the first post in a series where I try to build mechanistic interpretability from the ground up, in a way that actually made things click for me. This part has almost no "interpretability" in it yet. That is on purpose. Before you can ask how a network does something, you have to be very clear about what it computes and what shapes the computation moves through. So Part 1 is just the skeleton.*

Almost every mech-interp question is some flavor of the same question:

> **How did the network turn this prefix of text into that next-token distribution?**

You cannot answer that until the words *prefix*, *distribution*, and *the network* stop being fuzzy. So let's slow all the way down and pin them.

---

## 1. The thing we are actually modelling

Tokenization turns a piece of text into a sequence of integers:

$$x_0, \dots, x_{T-1}, \qquad x_t \in \\{0, \dots, V-1\\}$$

where

- $T$ is the sequence length,
- $V$ is the vocab size,
- $x_t$ is an integer token id at position $t$.

A language model puts a probability distribution over such sequences, and it does it through the autoregressive factorization:

$$p(x_0, \dots, x_{T-1}) = \prod_{t=0}^{T-1} p(x_t \mid x_{<t})$$

There is nothing deep here yet. This is literally the probability chain rule, plus one modelling choice: instead of writing down each conditional by hand, we learn all of them with a single neural network. During ordinary causal-language-model training, the transformer is solving, at every position at once,

$$(x_0, \dots, x_t) \rightarrow p(x_{t+1} \mid x_0, \dots, x_t)$$

Keep that map in your head. It is the object the whole series circles around: prefix in, distribution over the next token out.

---

## 2. A token is not a word

This sounds elementary, but it trips people up constantly, so I want to say it flatly: the model never sees strings. It sees integers.

The tokenizer is a fixed function that runs *before* the model:

$$\text{text} \rightarrow (x_0, x_1, \dots, x_{T-1})$$

So later, when I say something like "look at the activation at the `Paris` token", what I really mean is "look at the sequence position $t$ whose token id happens to be the vocab entry we call `Paris`." The name is for us. The model only has the index.

For a batch of $B$ prompts, the input is a tensor of shape

$$[B, T]$$

full of integers. From here on, these four letters do a lot of work, so let's fix them once:

- $B$ = batch size,
- $T$ = sequence length,
- $d$ = model width (the dimension we are about to meet),
- $V$ = vocab size.

---

## 3. Integers can't do linear algebra

Say the id for some token is $48787$. That number has no numerical meaning. $48788$ is not "slightly more" than it, and adding two ids gives nonsense. So the very first thing the model does is throw the integer away and replace it with a vector it is allowed to do math on.

It keeps an **embedding matrix**:

$$\mathbf{W}_E \in \mathbb{R}^{V \times d}$$

One row per vocab entry, each row a $d$-dimensional vector. For token $x_t$ we just read off its row:

$$\mathbf{r}_t^{(0)} = \mathbf{W}_E[x_t, :] \in \mathbb{R}^d$$

So the embedding matrix is used as a lookup table: the id picks the row. Across the whole batch this turns the integer tensor into a tensor of vectors,

$$[B, T] \rightarrow [B, T, d]$$

giving us $\mathbf{R}^{(0)} \in \mathbb{R}^{B \times T \times d}$.

**One more way to see the same thing.** If $e_{x_t} \in \mathbb{R}^V$ is the one-hot vector for $x_t$ (a $1$ in the position of that token, $0$ everywhere else), then the lookup is just a matrix product:

$$\mathbf{r}_t^{(0)} = e_{x_t}^\top \mathbf{W}_E$$

Selecting a row and multiplying by a one-hot are the same operation. Worth internalizing early, because a lot of later interpretability tricks are really "which direction in this space did we just pick out."

---

## 4. One vector per position: the residual stream

After the embedding step, each prompt in the batch is now a whole matrix in $\mathbb{R}^{T \times d}$: one $d$-dimensional vector per token position.

That per-position vector is what everyone means by the **residual stream**. You will constantly hear things like:

- "this feature is represented in the residual stream,"
- "the head *writes* to the residual stream,"
- "patch the residual stream at layer 17."

For now, take the most literal reading possible:

> At each token position, the network keeps one $d$-dimensional vector. The transformer's components repeatedly read from that vector and add information back into it.

At layer $l$ the whole batch of residual streams is a tensor

$$\mathbf{R}_l \in \mathbb{R}^{B \times T \times d}$$

and here is the structural fact that makes the residual stream such a clean object to study: the transformer stack deliberately keeps this outer shape fixed the whole way through.

$$[B, T, d] \rightarrow \dots \rightarrow [B, T, d]$$

Attention and MLPs use other dimensions internally, but they always hand back something of the same shape they got. The stream is a fixed-size workspace that gets edited, not resized.

---

## 5. Where does position enter?

There is an immediate problem with everything above. The token embedding tells the network *what* token appeared, but not *where*. A model that only had embeddings would be unable to tell `dog bites man` from `man bites dog`, because it would see the exact same bag of vectors.

So position has to be injected somewhere. There is no single canonical way to do it, and it genuinely varies across architectures. For a GPT-2-style model the choice is a learned **positional embedding**:

$$\mathbf{W}_P \in \mathbb{R}^{T_{\max} \times d}$$

and the initial residual vector at position $t$ is the token embedding plus the positional one:

$$\mathbf{r}_t^{(0)} = \mathbf{W}_E[x_t] + \mathbf{W}_P[t]$$

Notice both live in the same $d$-dimensional space and are simply added. "What token" and "which position" share the residual stream from the very first step. Do not overfit to this specific recipe though (modern models often use rotary or other schemes). The only thing you must take away is:

> A transformer needs *some* mechanism that makes sequence position available to its computation.

---

## 6. The transformer as one big function

Now abstract away every internal detail and just name the whole stack as a function that maps residual streams to residual streams:

$$F_\Theta : \mathbb{R}^{B \times T \times d} \rightarrow \mathbb{R}^{B \times T \times d}, \qquad \mathbf{R}^{(L)} = F_\Theta(\mathbf{R}^{(0)})$$

Its job is to progressively rewrite the vector at each position so that, by the end, that vector carries whatever is useful for predicting the next token. And a warning that took me a while to accept: there is no clean division of labor across layers. It is not "layer 1 does syntax, layer 8 does semantics." That story is too tidy to be true.

What the layers *do* share is a very specific shape. For a pre-norm architecture, one block looks like:

$$\mathbf{U}^{(l)} = \mathbf{R}^{(l)} + \text{Attention}_l\big(\text{Norm}(\mathbf{R}^{(l)})\big)$$

$$\mathbf{R}^{(l+1)} = \mathbf{U}^{(l)} + \text{MLP}_l\big(\text{Norm}(\mathbf{U}^{(l)})\big)$$

Ignore what Attention and MLP actually compute for now. Just stare at the plus signs. Each component does not *replace* the residual stream, it *adds* a vector into it. That additive structure is the whole reason the residual stream is such a good handle: every component is writing its contribution into a common, shared workspace, and in principle you can ask what each one wrote.

---

## 7. Attention moves information, MLPs don't

Before we open either box (that's later parts), there is one structural distinction worth having up front, because it shapes how you reason about everything.

An **MLP** acts on each position on its own. It is a per-position function:

$$\mathbf{r}_t \rightarrow \text{MLP}(\mathbf{r}_t)$$

Position $t$ goes in, a new vector for position $t$ comes out. No other position is involved.

**Attention** is the opposite: it is the mechanism that lets information move *between* positions. At position $t$, attention can effectively say:

> "Go pull some information from positions $0, \dots, t$, transform it, and write the result into position $t$."

So a clean first-order mental model: MLPs think locally at each token, attention is the only thing that ships information across tokens. When you later find a "feature" appearing at some position it had no business knowing about, attention is how it got there.

---

## 8. Causality: no peeking ahead

There is a hard constraint on that information flow. In a decoder-only autoregressive model, position $t$ is not allowed to look at any position ahead of it while predicting the next token.

The reason is not aesthetic, it is that training would otherwise be trivial and useless. If position $t$ could see $x_{t+1}$, then "predict $x_{t+1}$" is not a prediction, it is copying the answer. So attention at position $t$ is masked to only reach positions $0, \dots, t$. That is why the retrieval in the previous section only ever ran over $0, \dots, t$ and never further.

---

## 9. From the final vector to a distribution

After all $L$ blocks, each position $t$ holds a final vector

$$\mathbf{h}_t \in \mathbb{R}^d$$

(there is usually a final normalization first, which we will handle carefully later). To turn this into a guess about the next token, the model applies an **unembedding**, also called the **LM head**:

$$\mathbf{W}_U \in \mathbb{R}^{d \times V}, \qquad \mathbf{b}_U \in \mathbb{R}^V$$

$$\mathbf{z}_t = \mathbf{h}_t \mathbf{W}_U + \mathbf{b}_U \in \mathbb{R}^V$$

The $\mathbf{z}_t$ are the **logits**: one score per vocab entry. As a tensor transformation,

$$[B, T, d] \rightarrow [B, T, V]$$

and a softmax over that last axis finally gives the distribution we started the whole post with:

$$p(x_{t+1} \mid x_{\le t}) = \text{softmax}(\mathbf{z}_t)$$

Note the pleasant symmetry with Section 3. We came *in* through $\mathbf{W}_E$, mapping a one-hot token into a direction in $\mathbb{R}^d$. We go *out* through $\mathbf{W}_U$, reading a vector in $\mathbb{R}^d$ back out into scores over the vocab. Same space, both ends.

---

## The whole skeleton in one line

Strip away everything and this is the outer shape of the entire computation:

$$[B, T] \rightarrow [B, T, d] \rightarrow [B, T, d] \rightarrow [B, T, V]$$

or in words,

$$\text{token IDs} \rightarrow \text{residual stream} \rightarrow \text{transformer} \rightarrow \text{logits}$$

Integers come in, get looked up into vectors, get repeatedly edited in a fixed-size workspace, and get read back out as scores over the vocabulary.

---

## Where we go next

That's the skeleton, and I really do mean it when I say we should let this settle before moving on. Everything ahead is just filling in the boxes we deliberately left closed:

- what actually happens inside one attention block, and why "moving information between positions" turns into queries, keys, and values,
- what the MLP is doing at each position,
- and how the additive, shared structure of the residual stream lets us attribute pieces of the final logits back to individual components.

That attribution step is where interpretation really begins. But it only works because of the shape we set up here: a fixed-width stream that every component writes into by addition. Part 2 opens the attention block.
