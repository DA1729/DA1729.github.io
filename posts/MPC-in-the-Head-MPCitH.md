---
title: MPC in the Head (MPCitH)
date: 2025-12-06 18:04:01
tags:
    - Cryptography
    - Secure Computing
    - Zero-Knowledge
    - MPC
---

In continuation to my MPC blogs and my study of [FAEST](https://faest.info/), here is a blog explaining a method of constructing advanced **Zero Knowledge (ZK)** proofs by using simpler tools from **Multi-Party Computation (MPC)**.

## Zero-Knowledge Proof

Let's say that you are competing with a friend over a "Where's Waldo" puzzle and the one who finds Waldo in lesser time wins. Now, if you went first, after finding Waldo, you obviously cannot just prove to your friend that you have solved it by pointing it out in plain sight... it would ruin their turn. And if you don't prove it, your friend could easily argue that you're bluffing.
You must find a way to convince your friend that you know where Waldo is without showing them *where* Waldo is. Again, sounds like a Martin Gardner puzzle... and again I'm pretty sure there must be some similar problem in his collection.

Let's break it down somewhat formally:

- **Goal**: Prove you know a secret (location of Waldo) without revealing the location.
- **Roles**: You are the **Prover** and your friend is the **Verifier**.

Now how to prove it? Imagine you take a giant sheet of cardboard, much larger than the puzzle book, and cut a tiny hole in the center, just big enough to see Waldo.

- **Setup**: You tell your friend to turn around, then you place the cardboard over the puzzle book so that **only** Waldo is visible through the hole.
- **Proof**: You invite your friend to look. They look through the hole and see Waldo.
- **Trick**: Because the cardboard is huge and blocks out all the landmarks, your friend has no context. They just know that you know where Waldo is, but have no idea where on the page he is.

Now, why do cryptographers obsess so much over such proofs? First, let's see what core points are satisfied with a ZKP:

- **Completeness (works for truth-tellers)**: If you actually know where Waldo is, you can always perform this trick successfully.
- **Soundness (fails for liars)**: If you were bluffing, you cannot position the cardboard to show the real Waldo. You can't cheat the verifier.
- **Zero-Knowledge (it leaks nothing)**: After the game, your friend learns nothing new. If they tried to find Waldo themselves afterwards, they would have no advantage compared to before you showed them the proof. The "view" they saw (Waldo through the hole) is something they could have easily imagined themselves without your help.

## Formalization

Ok, now let's formalize things a little. In cryptography, we don't just "prove" things, we prove membership in a language defined by a relation $\mathcal{R}$.

Let $x$ be the public statement (the "instance") and $w$ be the secret witness. We define the relation as a set of pairs:
$$\mathcal{R} = \\{(x, w): \text{statement } x \text{ is true with witness } w \\}$$

In our Waldo example:
- $x$: The specific page of the puzzle book
- $w$: The $(x, y)$ coordinates of Waldo
- $(x, w) \in \mathcal{R}$ if and only if Waldo is actually at those coordinates on that page.

A ZKP is an interactive protocol between two probabilistic polynomial-time (PPT) algorithms: the **Prover** ($\mathcal{P}$) and the **Verifier** ($\mathcal{V}$).

Many of you might already know what probabilistic polynomial-time (PPT) algorithms are. At the time of writing the blog, I did not, so here is a quick explanation for readers like me:

- Its running time is bounded by a polynomial in the size of the input $n$, for all possible random choices it makes. So, if $T(n)$ is the worst-case number of steps, then $T(n) = \text{poly}(n)$.
- **Has access to randomness**, or has an extra stream of input: a stream of random bits. So the algorithm is a function: $A(x, r)$ where $x$ is the actual input and $r$ is the random string the algorithm uses.
- **It is allowed to have a probability of error**. Depending on the problem, the algorithm may give a correct answer with high probability, which can be amplified by repetition.

### Three Properties

#### Completeness
If the statement is true and the prover is honest, the verifier accepts.
$$\forall (x, w) \in \mathcal{R} : \text{Pr}[\langle \mathcal{P}(w), \mathcal{V} \rangle (x) = 1] = 1$$

#### Soundness (Knowledge Error)
If the statement is false (or the Prover doesn't know $w$), a malicious prover $\mathcal{P}^\*$ cannot convince the verifier, except with some negligible probability $\epsilon$ (soundness error).
$$\forall x \notin L, \forall \mathcal{P}^\* : \text{Pr}[\langle \mathcal{P}^\*, \mathcal{V} \rangle (x) = 1 ] \leq \epsilon$$

#### Zero-Knowledge (Simulation Paradigm)
This is the most critical and often the most misunderstood definition. How do we prove mathematically that "no information was leaked"?

We use the **simulation paradigm**. The idea is: if a Verifier could have generated the exact same proof transcript by themselves (without talking to the Prover), then the interaction with the Prover gave them zero new information.

We define the **View** of the verifier during the execution as the tuple of their random coins and the messages they received:
$$\text{View}\_\mathcal{V}[\mathcal{P}(x, w) \leftrightarrow \mathcal{V}(x)]$$

The protocol is Zero-Knowledge if there exists an efficient algorithm called a **Simulator** ($\mathcal{S}$). The simulator takes **only** the public input $x$ (it does not know $w$) and outputs a transcript that is indistinguishable from a real interaction.

$$\\{\mathcal{S}(x)\\} \approx \\{\text{View}\_\mathcal{V}[\mathcal{P}(x, w) \leftrightarrow \mathcal{V}(x)]\\}$$

If this equation holds, the proof reveals nothing about $w$, because anything the Verifier "learned" from the Prover, they could have just computed themselves using $\mathcal{S}$.

## MPC-in-the-Head (Intuition)

Since I have already covered MPC in my previous blogs, I am skipping MPC basics. We know that MPC allows a group of mutually distrusting parties to compute a function $f(x\_1 \dots, x\_n) = y$ without revealing their individual inputs.

Now, how do we turn a multi-player protocol into a single-player proof?

This was pioneered by Ishai, Kushilevitz, Ostrovsky, and Sahai in the famous [IKOS07] paper, and the motivation for coming up with this technique comes from a desire to stop reinventing the wheel.

- **The problem**: Designing custom Zero-Knowledge protocols for complex circuits (like proving you know the preimage of a SHA-256 hash or an AES key) is historically very difficult. You often have to translate your problem into complex number-theoretic assumptions.
- **The observation**: We already have excellent, generic ways to compute *any* circuit securely: **MPC**. MPC protocols can handle boolean circuits, arithmetic circuits, essentially any logic you throw at them.
- **The idea**: What if we don't actually need other people? What if the Prover just **simulates** an entire MPC universe inside their own brain? If the simulation is "consistent", the computation must be correct.

This turns the problem of "Designing a ZKP" into "Designing an MPC Protocol". Since we have very fast MPC protocols for things like AES (using boolean circuits), we automatically get fast ZKPs for AES.



To understand MPCitH, imagine the Prover is not a participant, but a **Puppet Master** controlling a simulation.

- **The Goal**: Prove I know a secret witness $w$ (perhaps a password) that satisfies the circuit $C(w) = 1$.

- **Step 1: The Split Personality (Secret Sharing)**: In my head, I imagine 3 distinct parties: Alice, Bob, and Finn (hahaha... u thought it would be Eve lol). I take my secret $w$ and split it into three shares ($w\_1, w\_2, w\_3$) using standard Secret Sharing (like XOR Sharing). I give one share to each imaginary party. Note that any **two** shares look like random garbage; you need all three to recover $w$.

- **Step 2: The Simulation (Execution)**: Now, I simulate the MPC protocol in my head.
    - Alice sends a message to Bob. I write it down.
    - Bob does a calculation and sends a message to Finn. I write it down.
    - Finn computes the final output.
    - Since I am honest and know the real $w$, the final output of this imaginary MPC is "TRUE".

- **Step 3: The Commitment (Locking the Views)**: I record the entire "View" of each party. A View contains:
    - Their input share
    - Their random coin flips
    - Every message they **received** from the others

I put Alice's view in Box A, Bob's view in Box B, and Finn's view in Box C. I seal them and put them on the table.

- **Step 4: The Challenge (The "Head" Check)**: You (the Verifier) walk in. You point to two random boxes, say **Box A** and **Box B**. I must open them.

- **Step 5: The Verification**: You check the two boxes for **Consistency**:
    - **Incoming/Outgoing Match**: If Alice's log says, "I sent value 5 to Bob," you check Bob's log. Does it say, "I received value 5 from Alice"?
    - **Local Validity**: Did Alice follow the math correctly based on the messages she saw?
    - **Output**: Did the protocol output 1?

#### Why is this Secure?
- **Why is it ZK?** You only opened 2 out of 3 boxes. Because of the properties of Secret Sharing (specifically $t$-privacy), seeing 2 shares reveals **nothing** about the underlying secret $w$. You saw a valid execution, but you didn't see enough to reconstruct the secret.

- **Why is it Sound (Why can't I cheat)?** If I didn't actually know the password $w$, the only way to make the MPC output "TRUE" is to cheat during the execution. I would have to make Alice send a "fake" message to Bob that makes the math work out.
    - If I fake a message, Alice's log will say "I sent X", but Bob's log (derived from honest calculation) would expect "Y".
    - There is a mismatch (inconsistency).
    - If you open the two boxes where the mismatch happened, you catch me.
    - Since I don't know which boxes you will ask for, I run a huge risk of getting caught. (We repeat this process many times to make the risk of cheating essentially 0%).

## Formalizing MPCitH

Now, let's put some mathematical structure on our "puppet master."

We start with an $N$-party MPC Protocol $\Pi$ that computes a function $f(w)$. The Prover $\mathcal{P}$ wants to prove $f(w) = 1$.

### The View
Just like we defined the "View" for the Verifier in ZK, we need to define the View for each imaginary MPC party $P\_i$. The view of party $i$ consists of everything they "know" during the execution:
$$\text{View}\_i = (w\_i, r\_i, m\_{in}^1, m\_{in}^2, \dots)$$

Where:
- $w\_i$: The $i$-th share of the witness (secret input).
- $r\_i$: The internal randomness (coin flips) used by party $i$.
- $m\_{in}^k$: The messages received by party $i$ during the rounds of communication.

**Crucially**: If you have the party's view, you can deterministically replay their entire computation to check if they were honest.

### The Commit Phase
The Prover runs the MPC protocol "in their head" for parties $P\_1 \dots P\_N$. They generate a commitment for each view. Usually, we use a hash function $H$ for efficiency:
$$c\_i = H(\text{View}\_i) \quad \text{for } i \in \\{1, \dots, N\\}$$

The Prover sends all commitments $(c\_1, \dots, c\_N)$ to the Verifier.

### The Challenge Phase
The Verifier picks a random set of parties to "corrupt" (inspect). Let's say we want to open all parties except one (to preserve ZK). The Verifier chooses an index $i^\* \in \\{1, \dots, N\\}$ to keep closed, and asks to open all others.
$$\text{Challenge } I = \\{1, \dots, N\\} \setminus \\{i^\*\\}$$

### The Response Phase
The Prover must reveal the views for all requested parties:
$$\text{Response} = \\{ \text{View}\_j \\}_{j \in I}$$

### The Verification Phase
The Verifier checks two things:
- **Correctness**: They re-run the code for every opened party $P\_j$ using the data in $\text{View}\_j$. They check if the output matches the commitment $c\_j$.
- **Consistency**: They check the messages between opened parties.
    - If Party 1 says "I sent 5 to Party 2", the Verifier checks Party 2's view.
    - Does Party 2's view confirm "I received 5 from Party 1"?

If all consistency checks pass and the output is 1, the Verifier accepts.

This seemingly simple "consistency check" is powerful enough to catch any cheating attempt where the Prover tries to force a "True" output without a valid input.

Now, let's end the blog here, got tired of typing.

peace. da1729
