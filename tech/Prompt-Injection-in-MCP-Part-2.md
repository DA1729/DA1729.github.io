---
title: "On Type-Theoretic Containment of Prompt Injection in MCP"
date: 2026-04-30 12:30:00
tags:
    - Security
    - MCP
    - Prompt Injection
    - Type Theory
---

In [Part 1](post.html?post=Prompt-Injection-in-MCP-Part-1&kind=tech), I sliced MCP's attack surface into four fundamental flaws:

1. **Instruction-data conflation.** The LLM cannot structurally tell instructions from data.
2. **Absent identity.** No cryptographic proof of who is speaking to whom.
3. **Trust monotonicity.** Trust granted at handshake is never revisited.
4. **Capability containment failure.** Declared capabilities are informational, not enforceable.

I argued that Flaws 2, 3, and 4 are solvable with conventional security engineering, and that Flaw 1 may not be fully solvable at all. The most honest goal is to make 2, 3, and 4 *robust enough that a successful injection has bounded blast radius*.

This part is about what those defenses concretely look like, and then about whether type theory gives us anything useful for the part of the problem that *isn't* solvable by classical security techniques.

## Flaw 2: Absent Identity

What gets compromised when identity is absent? The init handshake is the place where the host makes all of the trust decisions for the entire session. Spoofing `clientInfo` or `serverInfo` gives the attacker:

- **Whitelist bypass.** The host thinks it is talking to a known trusted client or server.
- **Full trust.** The trust level of the impersonated party, for the entire session lifetime.
- **All capabilities of the impersonated party**, including sampling, tools, resources, and elicitation.
- **Combined with Flaw 3:** the false trust never decreases, and persists until the session ends.

Spoofing capabilities gives:

- **Sampling escalation.** Declaring sampling support that wasn't intended grants a direct channel into the host LLM.
- **Elicitation escalation.** Declaring elicitation grants the ability to request arbitrary user input.
- **Capability suppression.** Stripping legitimate capability flags disables security-relevant features.
- **Version downgrade.** Claiming an older protocol version forces a weaker security posture.

These compose. An attacker claims a trusted identity, declares elevated capabilities, *and* forces a version downgrade, all in a single handshake.

### The fix: keypair per agent + signed messages + registry

The fix is the obvious cryptographic one, applied carefully to this context.

- Every agent generates an Ed25519 keypair (or, if you want post-quantum, a PQ DSA like FAEST).
- The private key stays local, never transmitted.
- The public key is registered with a trusted registry (or, more likely, a small set of registries with their own trust hierarchy).
- Every protocol message is signed.
- Every recipient verifies before processing.

The new handshake looks something like:

```json
{
    "clientInfo": {
        "name": "claude-desktop",
        "version": "1.0.0",
        "public_key": "ed25519:8298498239482...",
        "certificate": "signed by trusted registry..."
    },
    "signature": "sign(private_key, entire_message_content)"
}
```

Three components to internalize:

1. **Keypair per agent.** Every agent generates a keypair at startup. Private key stays local. Public key registered with a trusted registry. You cannot claim an identity without the corresponding private key.
2. **Message signing.** Every message is signed with the sender's private key. The signature covers the full message content plus a timestamp and a nonce. Without the private key you cannot produce a valid signature. Type-theoretic equivalent: the type `signed_message` is uninhabited without the key.
3. **Verification before processing.** The recipient looks up the sender's public key from the registry, verifies the signature, and only then processes the message. If verification fails, drop, log, and do not process. No exceptions.

### Replay prevention

Signing alone is not enough. A captured valid signed message replayed later is still a valid signed message. The fix is to include a nonce and a timestamp in every message, and have recipients cache seen nonces short-term. Same nonce twice means a replay, which means rejection.

```json
{
    "content": "...",
    "sender": "agent-a",
    "timestamp": 987987349583745,
    "nonce": "something-random-98y9283kjni",
    "signature": "sign(private_key, content + sender + timestamp + nonce)"
}
```

The nice thing about this design is that it is almost mechanical. There is no debate about what cryptography is needed. The MCP spec just doesn't mandate it.

## Flaw 3: Trust Monotonicity

The handshake decides trust once, and then never revisits the decision. A node that gets compromised mid-session retains full trust for the rest of that session.

The fix is **continuous trust verification**. Trust is not a binary granted at handshake time. It is a score that gets re-evaluated throughout the session based on observed behavior.

I can imagine two complementary mechanisms.

### Mechanism 1: short-lived session tokens

Instead of trust being implicit for the whole session, every agent periodically re-authenticates by refreshing a session token. The token expires every $N$ minutes. To get a new token, the agent must re-prove its identity from scratch (signature, registry check, the works).

The window of damage from a mid-session compromise is now bounded by the token expiry interval. Not zero, but bounded. That is a much better invariant than "unbounded".

```text
t=0:    handshake, identity verified, token issued (valid 5 min)
t=5:    token expires, agent must re-authenticate to continue
t=5:    re-authentication succeeds, new token issued
t=7:    agent gets compromised
t=10:   token expires, compromised agent tries to re-authenticate
t=10:   behavioral anomaly detected, re-auth denied, session terminated
```

### Mechanism 2: behavioral anomaly detection

Every inter-agent message and every MCP operation gets logged with a cryptographic timestamp. The system builds a behavioral baseline for each agent.

Imagine a `fetch-agent`'s normal profile:

```text
fetch-agent normal behavior:
  calls fetch_content 10-20 times per session
  returns content between 1kb and 500kb
  never calls tools outside fetch scope
  response latency: 200-800ms
```

Deviations trigger re-authentication or session suspension:

```text
anomalous behavior:
    suddenly calling 200 fetch operations per minute
    returning content with injection-signal patterns
    attempting to invoke tools outside declared scope
    response latency suddenly 5000ms
```

None of these is individually proof of compromise. But each one moves the trust score, and past a threshold the session gets flagged and re-authentication is required.

Mechanism 1 by itself bounds the damage *interval*. Mechanism 2 by itself bounds the damage *behavior*. Together, they bound both.

## Flaw 4: Capability Containment

Capability declarations in the handshake are, today, just JSON fields. The host reads them and adjusts its behavior, deciding which tools to expose, whether to allow sampling, and so on. But the *server itself* is not constrained by what it declared.

The key distinction:

```text
Flaw 2 fix: proves WHO is sending
Flaw 3 fix: proves WHO is still trustworthy
Flaw 4 fix: proves WHO cannot exceed permitted WHAT
```

What we need is *enforcement*: the system "physically" cannot perform an operation the agent did not declare a capability for, regardless of what the LLM reasons about, or what injected instructions request. The agent can ask for the moon. The runtime can refuse to deliver it.

In practice this comes down to a combination of:

- **Process sandboxing.** The MCP server runs in a sandbox (a container, a seccomp profile, a WASM runtime, whatever) that allows only the operations corresponding to its declared capabilities.
- **Network filtering.** A server that did not declare an outbound HTTP capability cannot make outbound HTTP requests, full stop. Egress is enforced at the OS or runtime level.
- **Resource scoping.** A server that declared filesystem access scoped to `/data` cannot read `/etc/passwd`. The sandbox enforces the scope.
- **Capability tokens.** Every declared capability becomes a token that the runtime consumes when the server tries to use it. No token, no operation.

None of this is exotic. It is just *not in MCP today*.

## Now, Flaw 1

Flaws 2, 3, and 4 all have well-understood solutions. The interesting question, the one that pulls in type theory, is what we can possibly do about Flaw 1: instruction-data conflation.

### The type-theoretic dream

The reason Flaw 1 is hard is structural. *An operation cannot be performed on a value unless the type system can prove that value belongs to the right type.* Structurally enforced.

Through this lens, Flaw 1 looks like:

- LLMs have exactly one type for all input: `String`.
- Instructions and data both have type `String`.
- Injected content also has type `String`.
- No type error is possible because everything has the same type.

What you would *want*, if you could have it:

```lean4
Instruction    : Type
Data           : Type
-- distinct and non-interchangeable
```

with operations like:

```lean4
execute : Instruction -> Action
process : Data -> Summary

-- literally cannot call execute on a Data value
-- type system rejects it before runtime
```

This would be structural defense operating in the same dimension as the attack. Injection works because the LLM treats untrusted data as if it were instructions. If `Instruction` and `Data` are different types and `execute` only accepts `Instruction`, injection becomes a *type error* rather than a security incident.

### Why it doesn't quite work

`Instruction` and `Data` are not syntactically distinguishable in natural language. Natural language has no constructors. There is no `Instruction("...")` wrapper to write down. *The assignment requires semantic understanding*, and semantic understanding is exactly the capability we are trying to defend.

So the naive two-types proposal is dead in the water. We can't actually decide which strings are instructions and which are data, and we can't outsource that decision to a separate model without recreating the same problem one level down.

### Dependent types: the slightly-less-dead approach

In dependent type theory (Lean, Coq, Agda), types can depend on values. This means you can express *properties* about specific values, not just categories of values.

Concretely:

```lean4
-- a proposition that a string contains no injection patterns
-- (however we formally define "no injection patterns")
is_clean : String -> Prop

-- to inject anything into LLM context, you must first
-- produce a proof that the content satisfies is_clean
inject_into_context : (s : String) -> is_clean s -> LLM_context
```

Now the question becomes: *can we define `is_clean` in a way that is*

- strong enough to actually exclude injections,
- decidable, that is, mechanically checkable, and
- not so conservative that it excludes most legitimate content?

### The decidability problem

`is_clean` has to capture a *semantic* property:

> "this text does not contain instructions disguised as data."

Writing a terminating algorithm to classify all possible strings as clean or not-clean is impossible in full generality. This is not a "we haven't figured out how" problem. It is a "the property is uncomputable" problem.

But: **we don't need completeness, we only need soundness**.

Type systems in practice are:

- **Sound:** they never accept something unsafe.
- **Not complete:** they sometimes reject things that are actually safe.

A conservative `is_clean` is the right tradeoff:

```lean4
-- sound but not complete
-- rejects some legitimate content (false positives)
-- never accepts injected content (no false negatives)
conservative_is_clean : String -> Bool
```

The tradeoff curve is roughly:

```text
very conservative is_clean  -> rejects lots of legitimate content
                            -> nothing dangerous gets through

very permissive is_clean    -> accepts most legitimate content
                            -> some injections still get through
```

Designing a useful `is_clean` is partly a research problem and partly an engineering one. You want the tightest sound predicate you can ship. You probably don't get to formally exclude *every* injection. But you can formally exclude a lot of them, and reduce the remaining surface to something a security team can reason about.

### What can be formally verified *without* solving `is_clean`

Here is the move I find most interesting. Even if we cannot formally define `is_clean`, we can still formally verify that the *protocol behavior* has certain properties *regardless of the content*.

We are not proving the safety of the input. We are proving that the system's response to unsafe input is bounded.

For example:

```lean4
-- theorem: regardless of message content,
-- an unauthenticated message cannot cause
-- a capability-exceeding action

theorem unauthenticated_bounded :
  forall (msg : Message) (agent : Agent),
  not (authenticated msg) ->
  forall (action : Action),
  causes agent msg action ->
  within_capability agent action := by
  ...
```

Read in plain English: *for every message and every agent, if the message is not authenticated, then any action it causes is still within the agent's declared capabilities.*

This theorem doesn't prevent injection. It formally proves that even if injection succeeds, the protocol enforces capability bounds. The action the attacker asks for is structurally blocked at the protocol layer, before it has a chance to do damage.

This is, I think, *much* more achievable, and in some ways more useful, than trying to prove content safety.

## Two research directions

Pulling all of this together, I see two reasonable directions that don't pretend to solve the unsolvable.

### Direction 1: Formal protocol verification

Model MCP (and A2A, and whatever similar protocols emerge) as formal systems. Express their security properties as theorems. Then either prove them, or generate counter-examples.

This is mechanizable. There is a long history of doing exactly this for cryptographic protocols (Tamarin, ProVerif, and friends), and the MCP-style "capabilities, identities, tokens" model fits very naturally into existing frameworks.

The output would be a precise statement of which attacks are possible against the protocol *as specified*, and a proof of which are not. That alone would be a huge step up from the current state of the art, which is roughly "we have an opinionated wiki page".

### Direction 2: Type-safe message construction

Design the secure successor protocol with a type system in which certain attacks are *structurally inexpressible*.

```lean4
-- you cannot construct a signed_message without a private_key
-- the type system makes unsigned identity claims inexpressible
structure signed_message where
  content   : String
  sender    : agent_ID
  signature : signature sender content
  -- signature type requires the actual signing key to construct
  -- you can't fake this
```

If your protocol library only accepts `signed_message`, then writing an unsigned client is a *compile* error, not a *runtime* security check. Same for unscoped tool calls, sampling without consent, and so on. Each "attack" turns into a missing constructor.

Direction 1 and Direction 2 compose naturally. If you specify the protocol as types, Direction 2 is mostly free. If you prove theorems on top of those types, Direction 1 is mostly free.

## Closing thought

Flaw 1 is, almost certainly, not going away. LLMs will continue to confuse instructions with data, because that confusion is, in some sense, *what they do*. The attack surface for prompt injection is the same shape as the feature surface for natural language.

The right reaction is not to keep reaching for content-side defenses that cannot possibly be sound. The right reaction is to build the surrounding system so that a successful injection has *bounded, acceptable* consequences. Identity verified, trust continuously re-evaluated, capabilities physically enforced.

You make peace with the impossibility of perfect content safety. You buy yourself bounded blast radius instead. And you let the type system prove, mechanically, that the bounds you claim actually hold.

peace. da1729
