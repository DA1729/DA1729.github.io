---
title: "Recovering a Ruined Roll, Part Three"
date: "2026-08-08"
---

# Recovering a Ruined Roll, Part Three

Now we're just going crazy nerd...

There is one defect on this roll that survived both previous rounds, and it survived them for a structural reason rather than because I did not try hard enough. Some of the light leaks land as a **hard, frame-spanning step**: a vertical band down one side of the frame where the film was exposed and everything abruptly changes level and colour.

Every tool I had reached for was a filter. Filters estimate a smooth field and subtract it. A Gaussian estimate smears straight across a discontinuity, so it is wrong on both sides of it. The edge-aware guided filter [1] did better, because it follows luminance boundaries, but it still treats the seam as a thing to be *estimated around* rather than a thing to be *deleted*.

The fix is to stop working on intensities and start working on derivatives.

## Editing gradients instead of pixels

The observation behind gradient-domain processing [2] is that the human visual system cares about local differences far more than absolute values, so the useful place to edit an image is its gradient field. You modify $\nabla I$ however you like, and then you ask: what image has that gradient?

Generally, no image does. A hand-edited gradient field $g$ is not conservative -- there is no $u$ with $\nabla u = g$ exactly. So you settle for the closest one in a least-squares sense,

$$\min_{u} \iint \|\nabla u - g\|^2$$

whose Euler-Lagrange equation is the Poisson equation

$$\nabla^2 u = \nabla \cdot g$$

This is the same machinery behind seamless cloning [2] and behind panorama stitchers hiding exposure seams between frames [3], which is essentially my problem wearing different clothes. They have two images that disagree along a join; I have one image that disagrees with itself along a join.

So: find the seam, zero the gradient across it, reintegrate. The step cannot survive, because after reintegration the only thing determining the offset between the two sides *is* the gradient you deleted.

## Solving it, quickly

I do not want an iterative solver here. With Neumann boundary conditions -- natural for images, they just mean "no flux across the frame edge" -- the discrete Poisson equation diagonalises under the DCT [4]. Take the DCT of the divergence, divide by the eigenvalues of the discrete Laplacian,

$$\hat{u}_{ij} = \frac{\hat{d}_{ij}}{2\cos\frac{\pi i}{M} + 2\cos\frac{\pi j}{N} - 4}$$

and inverse transform. The $(0,0)$ term is a division by zero, which is just the statement that the solution is only defined up to an additive constant, so you set it to zero and restore the original mean afterwards.

That is $O(MN\log MN)$ and about twenty lines with `scipy.fft.dctn`. No iteration, no convergence tuning, exact up to floating point.

## Finding the seam

The detection is embarrassingly simple and this is the part I found genuinely satisfying.

Average the absolute horizontal derivative of $L^*$ down each column. A real image edge is a local event -- it occupies part of a column. A leak seam runs the entire height of the frame, so every row contributes to the same column, and it stands up out of the profile like nothing else in the picture. Score it with a robust z using the median and MAD rather than mean and standard deviation, because the thing you are looking for would otherwise inflate the very statistics you are measuring it against.

![seam profile, and the grain scaling curve](https://raw.githubusercontent.com/DA1729/m35_photo_edit/main/docs/seam_and_grain.png)

Look at the left panel. The two leak seams score **z = 196 and z = 51**. Ordinary image structure -- the edges of a building, the outline of a spire, everything that is actually a photograph -- sits in the grey band below 20. On another frame the seam scores 118.

That is a two-orders-of-magnitude separation, which is about as clean as a detection problem ever gets in practice.

And it matters, because my first attempt used a threshold of 12 and appeared to help on seven of eight frames. It was quietly flattening real edges, and the aggregate metric I was watching went down anyway. At a threshold of 40 the detector fires on exactly two frames out of eight, which is the correct answer, and those are the two frames with visible bands.

The result, on the frame with the worst band:

![the seam, before and after](https://raw.githubusercontent.com/DA1729/m35_photo_edit/main/docs/seam_removal.jpg)

The column-208 seam goes from **z = 196 to z = -2.1** -- that column is now, if anything, marginally quieter than typical background, which is what you would expect after deleting its gradient outright. The dark band on the left does not get brightened or masked or painted over -- it merges into the sky, because after reintegration there is nothing left to say it was ever a separate region. The sky also comes out considerably more neutral, since the chroma flattening downstream is finally working on a smooth field instead of fighting a discontinuity.

## The idea that was wrong

Before any of that, I had what felt like a better idea, and I want to record it because the failure is more interesting than the success.

A real object boundary produces a gradient in **both** luminance and chroma. A colour cast boundary should produce a chroma gradient with **no** luminance counterpart. So: attenuate chroma gradients that luminance does not corroborate, reintegrate, and the leak's colour step dissolves while every real edge survives untouched. One rule, no thresholds, no seam detection.

It does not work at all. Measured zonal chroma spread came out worse than the guided filter I already had -- $\sigma_{b^*}$ of 14.5 against 9.6.

The reason is obvious about ten seconds after you see the numbers. A light leak is *additive light*. It does not merely tint the region, it brightens it. So the seam has a luminance step too, my "corroboration" test corroborates it enthusiastically, and the gradient I most wanted to delete is the one the rule protects hardest. The premise was wrong about the physics, not the maths.

## The idea that was right but inapplicable

The other thing I wanted was defect filling -- dust, scratches, the small dark specks. The classical literature here is good: exemplar-based inpainting propagates both texture and structure into a hole by copying patches in a confidence-ordered sequence [5], and for small defects the fast-marching method [6] is available directly in OpenCV.

So I built a detector: median filter, look at the residual, flag compact regions that deviate by more than a few robust sigma.

It found **11,236 defects in one frame**.

That is not dust. The right panel of the figure above is the diagnostic. I swept the detection threshold and plotted the count on a log axis, and it falls in a straight line from about 14,000 down to 3. A straight line on a log axis is exponential decay -- the tail of one continuous distribution. If there were a real population of dust specks sitting on top of the grain, they would survive as the threshold rose and the curve would flatten into a plateau at however many specks there actually are. There is no plateau anywhere.

So at this resolution, dust and film grain are the same measurement. Anything I remove is grain. The feature exists in the repo and is off by default, which is the only defensible setting.

## The thing I refused to do

Deconvolution was the obvious remaining move. These frames are soft, Richardson-Lucy [7,8] is the classical iterative choice, it is well studied for photographic restoration, and it would visibly sharpen them.

I did not do it, and not because it would not work.

The M35 has a fixed plastic meniscus lens. Its softness is not degradation that happened to the photograph, it is the optical signature of the camera that took it. Deconvolving it away does not recover anything -- it removes the reason for using that camera. There is a real line between restoring a photograph and modernising it, and undoing the lens is on the wrong side of it. The fog is damage. The leak is damage. The scanner noise is damage. The lens is the instrument.

Same reasoning for the grain, which is why the grain result above did not disappoint me much.

## Where this leaves the roll

The seam removal is the last genuine recovery I expect to get from these files. What remains is the fundamental limit from the first post -- red and green hold 37 to 102 code values, and no PDE fixes a quantisation floor.

What I like about this round is that all three outcomes came from the same habit. The seam detector works because I measured the separation between seams and real edges instead of guessing a threshold. The chroma-gradient idea died because I measured it against the method it was supposed to replace instead of just looking at the picture and deciding it seemed better. The dust detector got switched off because I plotted the count against the threshold instead of accepting a number that flattered me. Two of those three are negative results, and they took about as long as the one that worked.

Code, as before: [github.com/DA1729/m35_photo_edit](https://github.com/DA1729/m35_photo_edit). The Poisson solver and the seam detector are in `m35/lab/recon.py`, and the restoration outputs from round one are still sitting there untouched.

peace. da1729

## References

[1] He, K., Sun, J., & Tang, X. (2013). *Guided Image Filtering.* IEEE TPAMI 35(6).

[2] Pérez, P., Gangnet, M., & Blake, A. (2003). *Poisson Image Editing.* ACM SIGGRAPH.

[3] Agarwala, A. (2007). *Efficient Gradient-Domain Compositing Using Quadtrees.* ACM SIGGRAPH.

[4] Simchony, T., Chellappa, R., & Shao, M. (1990). *Direct Analytical Methods for Solving Poisson Equations in Computer Vision Problems.* IEEE TPAMI 12(5).

[5] Criminisi, A., Pérez, P., & Toyama, K. (2004). *Region Filling and Object Removal by Exemplar-Based Image Inpainting.* IEEE TIP 13(9).

[6] Telea, A. (2004). *An Image Inpainting Technique Based on the Fast Marching Method.* Journal of Graphics Tools 9(1).

[7] Richardson, W. H. (1972). *Bayesian-Based Iterative Method of Image Restoration.* JOSA 62(1).

[8] Lucy, L. B. (1974). *An Iterative Technique for the Rectification of Observed Distributions.* The Astronomical Journal 79(6).

Also relevant, and used earlier in this series: Burt & Adelson (1983), *The Laplacian Pyramid as a Compact Image Code*, IEEE Trans. Communications 31(4); He, Sun & Tang (2011), *Single Image Haze Removal Using Dark Channel Prior*, IEEE TPAMI 33(12); Dabov et al. (2007), *Image Denoising by Sparse 3-D Transform-Domain Collaborative Filtering*, IEEE TIP 16(8).
