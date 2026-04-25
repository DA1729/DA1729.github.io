---
title: Mapping the Max-Cut Problem to Ising Spin Glasses
date: 2026-03-09 15:25:15
tags:
    - Graph Theory
    - Physics
---

In the previous post, I explored the physics of Ising models and how Simulated Annealing can find global minima in complex energy landscapes. Now, I want to step away from pure physics and enter the realm of combinatorial optimization. I am going to look at a classic NP-hard problem: the **Max-Cut Problem**. The goal here is to show how I can mathematically "trick" a physics engine into solving a telecommunications network problem through a process known as QUBO (Quadratic Unconstrained Binary Optimization) mapping.

## 1. The Problem: Max-Cut in Networks

Imagine I am managing a telecommunications network. I have a set of nodes (routers or cities) connected by edges (cables). Each edge has a weight $W\_{ij}$, representing the cost or traffic capacity of that connection.

The objective is simple: I must divide every node into exactly two distinct teams, let's call them Team A and Team B.
*   If two connected nodes are on the same team, the edge between them is **kept**.
*   If they are on opposite teams, the edge is **cut**.

The goal is to maximize the total weight of all cut edges. I want to slice the network so that the heaviest connections are severed. Because there are $2^n$ possible ways to divide $n$ nodes, this problem is mathematically intractable for large $n$. There is no known polynomial-time algorithm to solve it perfectly.

## 2. The Integration: QUBO Mapping

To solve this using an Ising model, I need to translate "Teams" and "Cables" into "Spins" and "Couplings."

### Step 1: Mapping the Variables
In Max-Cut, a node belongs to Team A or Team B. In the Ising model, a spin $s\_i$ is $+1$ or $-1$.
*   If node $i$ is on Team A: $s\_i = +1$
*   If node $i$ is on Team B: $s\_i = -1$

### Step 2: The Mathematical Switch
I need a formula that outputs $1$ if an edge is cut ($s\_i \neq s\_j$) and $0$ if it is kept ($s\_i = s\_j$). Consider the expression:
$$\frac{1 - s\_i s\_j}{2}$$
If $s\_i = s\_j$, their product is $1$, and the expression becomes $\frac{1-1}{2} = 0$. If $s\_i \neq s\_j$, their product is $-1$, and we get $\frac{1-(-1)}{2} = 1$. The switch works flawlessly.

### Step 3: Formulating the Total Cut
The total weight of the cut $C$ is the sum of weights multiplied by this switch:
$$C = \sum_{i < j} W_{ij} \left( \frac{1 - s\_i s\_j}{2} \right)$$

By expanding this, I get:
$$C = \frac{1}{2} \sum_{i < j} W_{ij} - \frac{1}{2} \sum_{i < j} W_{ij} s\_i s\_j$$

The first term is a constant (half the total weight of all edges). To maximize $C$, I must maximize the second term: $-\frac{1}{2} \sum W_{ij} s\_i s\_j$.

### Step 4: Algebra to Physics
Compare this to the Ising Hamiltonian $H$ I want to minimize:
$$H = -\sum_{i < j} J_{ij} s\_i s\_j$$

The mapping is now obvious. If I set the coupling strengths $J\_{ij}$ to be the negative of the weights $W\_{ij}$:
$$J_{ij} = -W_{ij}$$
Then minimizing the physical energy $H$ is mathematically identical to maximizing the graph cut $C$. By setting $J\_{ij} < 0$, I am using **Antiferromagnetic Coupling**: the spins "hate" each other and want to point in opposite directions, which is exactly what happens when I cut an edge.

## 3. The Algorithmic Showdown

To prove this works, I pitted three agents against the same random weighted network:
1.  **Blind Guesser:** Randomly assigns teams (the baseline).
2.  **Greedy CS Heuristic:** Flips a node only if it immediately increases the cut (the standard approach).
3.  **Physics Engine (SA):** My QUBO-mapped Ising solver using Simulated Annealing.

<p align="center">
  <img src="/images/max_cut_algo_comp.png" width="80%">
</p>

As the plot shows, the Greedy heuristic (red) climbs quickly but often hits a local maximum and flatlines. The Physics Engine (blue) initially dips as it explores the landscape with thermal noise, but eventually climbs past the greedy plateau to find a superior solution.

## 4. Statistical Superiority

I never trust a single run. To definitively prove the advantage of the physics-based approach, I ran a Monte Carlo simulation across 50 unique network topologies.

<p align="center">
  <img src="/images/max_cut_monte_carlo.png" width="90%">
</p>

The results are conclusive: the Physics Engine (SA) won 80% of the trials. The bar chart on the right shows the "Advantage" of SA over Greedy. While Greedy occasionally gets lucky with a favorable starting position, Simulated Annealing's ability to take "bad" moves to escape local traps makes it the mathematically superior architecture for complex optimization.

By mapping computer science to thermodynamics, I have created a solver that doesn't just look for a quick answer, but uses physical randomness to navigate toward the global optimum. This justifies the effort of designing dedicated hardware to solve these equations in silicon.

---
