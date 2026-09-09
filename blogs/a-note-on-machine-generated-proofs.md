---
title: "A Note on Machine Generated Proofs"
date: "2026-09-09"
description: ""
---

# A Note on Machine Generated Proofs

I have my exams going on, but I am not able to help myself but think about this really interesting argument I have come up with, therefore, I am writing a note and posting it so that I will come back to it when exams are over...

This connects to [What Can't Be Offloaded](https://da1729.github.io/post.html?post=what-cant-be-offloaded&kind=blog), although I don't want to force the connection yet. For now, there are two arguments I want to keep separate.

First, all this talk about mathematics being "over for humans" because AI is producing proofs of difficult problems. What exactly are we comparing here?

By incompleteness, any consistent, effectively axiomatized theory $F$ containing enough arithmetic leaves some statements undecided. The qualification **in $F$** matters. A statement unprovable in one theory might be provable in a stronger one. We do not get to conclude that there is some fixed collection of statements forever inaccessible to every human and every AI.

But we do get a useful distinction. If neither a statement nor its negation has a proof in $F$, neither a human nor an AI can produce a valid $F$-proof of either. More compute does not put a missing proof into the theory.

Also, for theories of this kind, there is no algorithm that always terminates and correctly tells us whether an arbitrary statement has a proof. We can enumerate proofs and eventually find one if it exists. When it doesn't, continuing the search is not itself a way of finding that out.

This doesn't make comparisons useless. Someone can be much better at finding proofs while facing the same formal limitations. It just means that **being better at producing proofs and having unrestricted access to mathematical truth are different claims.**

Now the second argument, which is the one I keep thinking about instead of studying.

Suppose a model produces a proof of some difficult PDE result. In principle, I can write down its algorithm, its exact weights, its inputs, and perform every calculation by hand.

Obviously I am not doing this over the weekend. Unlimited paper, time, and patience. Grant the idealization.

There are details: I would need to reproduce its finite-precision arithmetic, its random choices, and any relevant tool outputs or external inputs. If multiple agents were involved, their interactions belong to the procedure too. But once the complete terminating run is specified, it can be executed step by step.

And I get the same output.

Now look at what happened on the paper. I performed a mathematical computation, involving the operations implementing the model, and obtained a proof involving PDEs. The procedure might be absurdly long and completely unfamiliar as a way of doing mathematics, but where exactly did an "alien" access to mathematical truth enter?

Another thing. Suppose I actually perform all those calculations by hand. The physical setup is now me, a pen, and a ridiculous amount of paper. My mind reads the instructions, keeps track of the state, and directs my hand.

Now, because this setup produces the same proof as the AI system, would I conclude that the pen and paper have acquired mathematical intelligence? Obviously that doesn't follow. Nor would I conclude that my faithfully executing the algorithm demonstrates that I understood the proof.

What I have established is an equivalence in one respect: the execution produces the same mathematical output. I have not established an equivalence between everything happening in the two systems.

And I am not using this to declare one conscious and the other unconscious, or one superior and the other inferior. The physical implementations differ. Whether their understanding or experience differs, and how, remains a separate question.

There is one distinction I should keep clear, though. **The calculation that generates the proof is not necessarily the argument that proves the theorem.** The model's matrix multiplications explain how I obtained the text. The resulting proof still has to establish the mathematical claim.

Schematically:

$$
\text{specified computation}\longrightarrow \pi,
\qquad
\pi\text{ is a valid proof in }F.
$$

Executing the first does not automatically mean I understand the second.

Nor does this show that I could have discovered the model's procedure or learned its weights myself. I supplied those at the start. Efficiency and access to information still matter enormously.

What it establishes is narrower: the proof-producing computation is human-executable in principle. That fact alone settles neither understanding nor experience, and it does not need Penrose's assumption.

So perhaps "ceiling" is the wrong thing to look for. I want to distinguish formal reach, practical ability to discover proofs, and understanding of the structures involved. Producing the same answer might say much less about these than we casually assume.

This also brings me back to representations: if two systems produce the same proof, what survives when I change the assumptions, change the representation, or ask why the argument fails somewhere else?

That feels like a question I can actually investigate.

After exams. Hopefully.

peace. da1729
