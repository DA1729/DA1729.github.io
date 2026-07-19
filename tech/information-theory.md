---
title: "Information Theory"
date: "2026-07-20"
---

# Information Theory

I am studying Information Theory, from Thomas and Cover's book [1]. I am just writing the things that I have just studied, like the very start, where the authors discuss how info theory is related to all these different fields: Electrical Engineering, Computer Science, Physics and Philosophy.

Ok, so off the top of my head, the things I could remember. Soooo... we have this communication system, well, first talking about the relation with Electrical Engineering. Say that we have a message source, and a message channel. Now, essentially, I think, so I am no expert of Information Theory (yet), it will be nice, if one is interested in the theoretical aspects of information theory, to start looking beyond the conventional intuition behind all these processes like compression and transmission, and rather, view them as "transformations," which can be defined depending on the system we are dealing with, could be defined deterministically, stochastically, or in practical systems, most of the times, empirically through measured characteristics.

Now, earlier people thought that increasing the transmission rate, given a channel, would only increase the errors. But now, this may sound the same, but there's a difference. So Shannon showed that, and, since I have just started, I am writing it very loosely and informally -- given a channel, it has a certain capacity (usually measured in bits per second), and upto that capacity in the transmission, one can achieve near-zero error (asymptotic). It is different because it changes the approach towards engineering communication systems. Say before this theorem, if a communication system faced a lot of errors, the engineer would just blindly reduce the transmission rate, but after the theorem, they would, given that they have computed the channel capacity, first compare the two rates, then, even if the transmission rate is a bit below the capacity, they would rather work upon finding the suitable error-correction codes (not relevant to understand them for now), encoding, decoding techniques, etc.

Now, this is one-dimension to make communication more efficient, another question to ask is, do we really need to send everything? Yep, I am talking about compression and decompression now. Now, compression in a nutshell and intuitively, is finding the shortest description of the data. Now the description cannot be any description, it should be able to describe the data so that it can be reconstructed upto a very high accuracy -- decompression. Shannon also showed that for a stochastic message source, it can be compressed only upto a limit, not more than that without actually losing information. This is where a fundamental Computer Science concept kicks in -- Kolmogorov Complexity.

Kolmogorov Complexity of a string, in a nutshell, is the minimum length of a computer program generating that text. Generally, the length is measured by encoding the program in binary format. Now a question, I asked when I first read this, which language though? So, yeah, the length of a given algorithm does vary language to language, but Kolmogorov was able to show that, let's say that we have two programming systems: $A, B$, then their difference is bounded within a constant independent of the given string being described: $$|K_A(x) - K_B(x)| \leq c$$ where $c$ does not depend on $x$.

Now, Kolmogorov Complexity leads us to one of the fundamental pieces of information theory -- Entropy. Entropy describes a **probability distribution** or information source. It measures the average uncertainty per outcome. For example, a fair coin has entropy of `1 bit per toss`, because we need a whole bit to encode the outcome of the coin toss. Now, let's say that you flip a coin $n$ times, and you get a string $$x = HTTTHTTHHHTHTTHHTHHTHTHHH\cdots$$. Now the complexity of the string is $n$, because, statistically speaking, there is no better way to describe the string rather than just printing the entire thing. Now, we know that its entropy $H$ equals $1$, so if we denote the complexity by $K(x)$, we have the following relation: $$K(x) \approx nH$$ for a typical $n-$length sequence.

Another thing to note is that the Kolmogorov complexity $K(x)$ is the ultimate data compression. Since it literally represents the shortest possible description of $x$. But it's generally hard to actually find an algorithm/description with length equal to the complexity. It's an ultimate theoretical limit, but not a practical compression algorithm. And, according to my interpretation, this is where we actually see them living in genuinely different domains. It sounds boring and obvious, but it's really trippy to think about, literally experiencing the boundaries between two fields, MUCH more interesting than, well not always, there are some amazing borders here in Europe, but yeah -- more interesting than international borders.

Gonna end it right here, and also, why am I studying info theory all of a sudden -- well, I have some serious ideas and I need to become an information theorist for it, so yeah, I am serious with it. tot ziens :)

da1729

**References**

1. *Elements of Information Theory* by Thomas M. Cover and Joy A. Thomas
