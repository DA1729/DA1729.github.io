---
title: "Mechanistic Interpretability, Part 2: Reading the Logits"
date: 2026-08-16
tags:
    - Mechanistic Interpretability
    - Transformers
    - Machine Learning
---

> *Part 2 of building mechanistic interpretability from the ground up. We stopped Part 1 exactly at the logits, and there is a payoff sitting right there that I do not want to walk past, because it is the first place actual interpretability shows up, and it falls straight out of the additive structure we already built.*

Quick recap of where Part 1 left us. The transformer turned a prefix into one final vector per position, $\mathbf{h}_t \in \mathbb{R}^d$, and the unembedding read that vector out into a score per vocab entry:

$$\mathbf{z}_t = \mathbf{h}_t \mathbf{W}_U + \mathbf{b}_U \in \mathbb{R}^V$$

Those scores are the logits. This whole post is about what they mean, how a single interpretability tool falls out of them, and then how training and generation actually use them.

---

## Logits are not probabilities

A logit is just a raw score. To turn the logit vector at a position into a distribution over the next token, we apply a softmax across the vocabulary dimension:

$$p_t(i) = \frac{\exp(z_{t,i})}{\sum_{j=1}^{V} \exp(z_{t,j})}$$

So for a logits tensor of shape `[B, T, V]`, the conceptual operation is one line:

```python
probs = logits.softmax(dim=-1)
```

and each `(batch, position)` pair gets its own distribution over the $V$ possible next tokens.

Here is the thing to actually internalize, because it changes how you read everything downstream:

> A single logit value means almost nothing on its own. What carries meaning is the *difference* between logits.

Saying "the logit for this token is $13.7$" tells you basically nothing, because softmax is invariant to adding a constant to every logit: shift all of them by $+100$ and the probabilities are identical. Only the gaps between logits survive the softmax. So from now on, whenever we ask what the model prefers, we ask about differences.

---

## The unembedding is a bank of linear questions

The cleanest way to see the output layer is not "it produces $V$ scores," but "it asks $V$ separate linear questions of the same vector $\mathbf{h}$." Each vocab token $i$ owns an unembedding vector $\mathbf{w}_i$, and its score is just a dot product:

$$z_i = \mathbf{h}^\top \mathbf{w}_i$$

To make this concrete, imagine for a second that $d = 2$, so we can literally picture the vectors. Suppose the model has finished reading `The capital of France is` and produced

$$\mathbf{h} = (3, 1)$$

and that three of the vocab tokens have unembedding vectors

$$\mathbf{w}_{\text{Paris}} = (2, 1), \qquad \mathbf{w}_{\text{London}} = (0, 2), \qquad \mathbf{w}_{\text{banana}} = (-2, 0)$$

Then the scores are just dot products:

$$z_{\text{Paris}} = \mathbf{h}^\top \mathbf{w}_{\text{Paris}} = 3(2) + 1(1) = 7, \qquad z_{\text{London}} = 3(0) + 1(2) = 2$$

The token `Paris` is asking "how much does $\mathbf{h}$ point along my direction?" and getting a big answer. `London` asks the same question along a different direction and gets a small one. The entire output layer is this, done $V$ times at once:

$$\mathbf{h} \mapsto \big(\mathbf{h}^\top \mathbf{w}_1, \; \mathbf{h}^\top \mathbf{w}_2, \; \dots, \; \mathbf{h}^\top \mathbf{w}_V\big)$$

Nothing semantic or mysterious happens in this last step. The transformer did all the hard work to produce $\mathbf{h}$; the output layer is just a giant linear classifier over vocabulary items, reading that one vector from $V$ different angles.

---

## The logit-difference direction

Now combine the two ideas above: differences are what matter, and each score is a dot product. Say we care about two candidate tokens $a$ and $b$. Because $z_a = \mathbf{h}^\top \mathbf{w}_a$ and $z_b = \mathbf{h}^\top \mathbf{w}_b$, linearity lets us fold the comparison into a single dot product:

$$z_a - z_b = \mathbf{h}^\top \mathbf{w}_a - \mathbf{h}^\top \mathbf{w}_b = \mathbf{h}^\top (\mathbf{w}_a - \mathbf{w}_b)$$

Define the **logit-difference direction**

$$\mathbf{d}_{a,b} = \mathbf{w}_a - \mathbf{w}_b, \qquad z_a - z_b = \mathbf{h}^\top \mathbf{d}_{a,b}$$

That is the whole trick, and it is worth stating plainly what it is and is not. It is exactly the direction in residual-stream space that distinguishes the output score for $a$ from the output score for $b$. Back in our 2D example:

$$\mathbf{d} = \mathbf{w}_{\text{Paris}} - \mathbf{w}_{\text{London}} = (2, 1) - (0, 2) = (2, -1), \qquad \mathbf{h}^\top \mathbf{d} = 3(2) + 1(-1) = 5$$

which is exactly $z_{\text{Paris}} - z_{\text{London}} = 7 - 2 = 5$, as it must be.

Geometrically, $\mathbf{d}$ carves residual-stream space with a hyperplane $\mathbf{h}^\top \mathbf{d} = 0$. If $\mathbf{h}$ points strongly along $\mathbf{d}$ (that is, $\mathbf{h}^\top \mathbf{d} \gg 0$) the model prefers Paris; point it the other way ($\mathbf{h}^\top \mathbf{d} \ll 0$) and it prefers London; land on the plane and the two are tied. This is just the geometry of a linear classifier, with the final residual vector as the thing being classified.

And the softmax makes the difference even nicer to talk about. Take the ratio of the two probabilities and almost everything cancels:

$$\frac{p(\text{Paris})}{p(\text{London})} = \frac{e^{z_P}}{e^{z_L}} = e^{\,z_P - z_L} \quad\Longrightarrow\quad \log \frac{p(\text{Paris})}{p(\text{London})} = z_P - z_L$$

So a logit difference is literally the **log-odds** of one token against the other, independent of every other token in the vocabulary. If $z_P - z_L = 2$, then Paris is $e^2 \approx 7.4$ times as likely as London, and it does not matter what the other fifty thousand logits are doing.

---

## The first real tool: direct logit attribution

Here is where Part 1's obsession with the *additive* residual stream pays off. Recall that every component (an attention head, an MLP) does not overwrite the stream, it adds a vector into it. So the final vector is a sum of contributions:

$$\mathbf{h} = \mathbf{h}_1 + \mathbf{h}_2 + \dots + \mathbf{h}_n$$

Dot products are linear, so the Paris-vs-London preference splits cleanly across those contributions:

$$\mathbf{h}^\top \mathbf{d} = \mathbf{h}_1^\top \mathbf{d} + \mathbf{h}_2^\top \mathbf{d} + \dots + \mathbf{h}_n^\top \mathbf{d}$$

Read that again slowly, because it is the point. If some head writes a vector $\delta\mathbf{h}$ into the stream, its effect on the Paris-vs-London logit difference is just its own projection onto the direction:

$$\Delta(z_P - z_L) = \delta\mathbf{h}^\top \mathbf{d}$$

which means we can hand every component its own scorecard. Project each component's write onto $\mathbf{d}$ and you might get something like:

- head 5.2: $+2.1$
- head 7.4: $+0.1$
- MLP 8: $-0.8$

and now you can say something concrete and testable: head 5.2 is directly pushing the output toward Paris, MLP 8 is mildly pushing the other way, and head 7.4 is barely involved in *this particular* comparison. That is **direct logit attribution**, the first genuinely mechanistic tool in the series, and it exists only because a linear readout interacts beautifully with a sum. There are normalization details (that final `Norm` from Part 1) that we will handle properly later, but this is the whole idea.

---

## A direction is not a concept

I need to fence this off carefully, because it is exactly the kind of place interpretability gets sloppy. When I call $\mathbf{w}_{\text{Paris}} - \mathbf{w}_{\text{London}}$ "the Paris-vs-London direction," I am **not** claiming that this direction is the model's internal concept of Paris. The rigorous statement is much narrower:

> Moving $\mathbf{h}$ along this direction changes the final linear readout in favour of Paris over London.

That is all it says. It is a fact about the output layer, which we know exactly, because we can just read $\mathbf{W}_U$ off the weights. It says nothing about whether the network's internal notion of "Paris" is stored cleanly in one direction, in many, or smeared across components in a way no single vector captures.

An analogy that keeps me honest: imagine you do a long, messy calculation on scratch paper, and at the very end an examiner scores you with the rule $\text{score} = 3x - 2y$, reading off two numbers $x$ and $y$ from your final line. The vector $(3, -2)$ tells you exactly how the examiner *reads* your final state. It tells you nothing about the reasoning that produced it. Same split here:

- $\mathbf{h}$ is the complicated internal state the transformer produced, and understanding it is the hard part.
- $\mathbf{w}_{\text{Paris}}$ is how the output layer reads that state when scoring Paris, and we know it exactly.
- $\mathbf{w}_P - \mathbf{w}_L$ is how the output layer distinguishes the two, and we know that exactly too.

The readout direction is known. The internal representation is the open question. Throughout this series I am going to keep **architecture facts** (things the weights force to be true) and **interpretive hypotheses** (stories about what a direction "means") in separate boxes, and the jump from the first to the second is where most of the real difficulty lives.

---

## Which position predicts which token

One indexing convention has to become completely automatic, because it is the source of endless off-by-one bugs. Suppose the input is $[x_0, x_1, x_2, x_3]$ and the model produces logits $[\mathbf{z}_0, \mathbf{z}_1, \mathbf{z}_2, \mathbf{z}_3]$. The training interpretation is shifted by one: position $t$'s logits are a prediction of the *next* token, $x_{t+1}$. So $\mathbf{z}_0$ scores $x_1$, $\mathbf{z}_1$ scores $x_2$, and so on. The loss is the average negative log-probability of each true next token:

$$\mathcal{L} = -\frac{1}{T-1} \sum_{t=0}^{T-2} \log p_\theta(x_{t+1} \mid x_{\le t})$$

In practice this means the distribution you usually care about at generation time lives at the *last* input position:

```python
next_token_logits = logits[:, -1, :]   # shape [B, V]
```

(As an aside, Hugging Face's GPT-2 does this shift internally when you pass `labels`, which is why `labels = input_ids` is the valid way to get its causal-LM loss. It is not that the model predicts the current token; the shift is just hidden inside.)

---

## Training is parallel, generation is a loop

A subtle but fundamental asymmetry closes out this part.

**Training does not have to be sequential.** Given a full training sequence like `The cat sat on the mat`, all the "correct previous tokens" are already sitting in the input. So a single forward pass can compute every next-token prediction at once:

```text
The             -> cat
The cat         -> sat
The cat sat     -> on
The cat sat on  -> the
...
```

The causal mask (Part 1's "no peeking ahead") is what makes this safe: prediction at position $t$ physically cannot see position $t+1$, so scoring all positions in parallel does not let any of them cheat. This trick has a name, **teacher forcing**: while predicting each next token during training, the prefix is the actual ground-truth history, not the model's own past guesses. The payoff is huge parallelism over sequence positions.

**Generation cannot do this**, because the future tokens do not exist yet. You are forced back into a genuine loop:

```python
tokens = encode(prompt)
for step in range(max_new_tokens):
    logits = model(tokens)
    next_logits = logits[:, -1, :]
    next_token = sample(next_logits)
    tokens = cat(tokens, next_token)
```

Each pass produces the distribution for one new token, you pick one, append it, and run again. Done naively this recomputes the entire prefix every step, which is wasteful, so real implementations keep a **KV cache** to avoid redoing that work (Hugging Face's GPT-2 exposes cached keys and values for exactly this reason).

But notice I just used the words "keys" and "values" as if they mean something, and so far in this series they do not. That gap is exactly where the interesting part of attention lives.
