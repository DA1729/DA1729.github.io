---
title: "A Note on Machine Generated Proofs"
date: "2026-09-09"
description: ""
---

# A Note on Machine Generated Proofs

I am writing a short argument after reading countless tweets and having a discussion with a college batchmate of mine, asserting that now that AI has found a solution to the Navier-Stokes Equation, it's over for mathematicians, and some even go further, asserting that the purpose of human mathematicians is over. My main argument is that even if an AI discovers a proof that no human in practice could find, the proof-producing computation is still, in principle, a finite procedure a human could execute step by step and reproduce exactly.

First, all this talk about mathematics being "over for humans" because AI is producing proofs of difficult problems. What exactly are we comparing here?

By incompleteness, any consistent, effectively axiomatised theory $F$ containing enough arithmetic leaves some statements undecided. A statement unprovable in one theory might be provable in a stronger and more abstract one. We can't really conclude that there is some fixed collection of statements forever inaccessible to every human and every AI yet.

But we do get a useful insight. If neither a statement nor its negation has a proof in $F$, neither a human nor an AI can produce an $F$-proof, no matter the amount of compute granted.

Also, for theories of this kind, there is no algorithm that always terminates and correctly tells us whether an arbitrary statement has a proof. We can enumerate proofs and eventually find one if it exists. When it doesn't, continuing the search is not itself a way of finding that out.

From this, we could infer that, being better at producing proofs and having unrestricted access to mathematical truth are different claims.

Now the second argument.

Suppose a model produces a proof of some difficult PDE result. In principle, I can write down its algorithm, its exact weights, and its inputs, and perform every calculation by hand—with unlimited paper, time, and patience. Grant the idealisation.

Every nuance of the execution: finite-precision arithmetic, randomness, and any relevant tool outputs or external inputs are reproducible. If multiple agents were involved, their interactions belong to the procedure too and are as reproducible. Once everything is specified, it can be executed step by step.

And I get the same output.

Now look at what happened on the paper. I performed a mathematical computation that implemented the model and obtained a proof involving PDEs. The procedure might be absurdly long and completely unfamiliar as a way of doing mathematics, but where exactly did an "alien" access to mathematical truth enter?

Another thing. I perform all those calculations by hand. The physical setup is now me, a pen, and a ridiculous amount of paper. My mind reads the instructions, keeps track of the state, and directs my hand.

Now, since this setup yields the same proof as the AI system, would I conclude that pen and paper have acquired mathematical intelligence? Obviously, that doesn't follow. Nor would I conclude that my faithfully executing the algorithm demonstrates that I understood the proof.

What I have established is an equivalence in one respect: the execution produces the same mathematical output. I have not established an equivalence between everything happening in the two systems.

Now, one might say that AI models generating proofs formally verify their proofs; perhaps that could be the "alien" access to mathematical truth. Well, I could grant my dumb brain, hand, and paper to mindlessly execute the steps, a set of more steps taking care of formal verification. Again, it doesn't make sense to say that I understand the proof now. In fact, I could say that the one who really understands, or is capable of understanding the proof, is the one who gave my dumb self the steps to execute!
