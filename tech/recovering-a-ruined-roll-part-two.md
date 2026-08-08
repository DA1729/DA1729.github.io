---
title: "Recovering a Ruined Roll, Part Two"
date: "2026-08-08"
---

# Recovering a Ruined Roll, Part Two

Ok, we recovered a few shots in the previous post, but let's keep playing because why not, and results are not disappointing...

The last post ended on an observation I did not do anything with. The black and white conversions came out better than the colour ones. Not "better" as in I preferred them aesthetically -- better as in visibly less noisy, with more actual detail in them. That is a strange thing to be true. The B&W is *derived* from the colour. It should not contain more information than its own source.

Except it does, and the reason is in how it was built.

## Why the monochrome was cleaner

The B&W conversion never used a standard luma mix. A standard mix is

$$Y = 0.21R + 0.72G + 0.07B$$

which on this roll is close to the worst possible choice, because red and green are the two channels that got destroyed. They hold 37 to 102 code values; blue holds around 200. So the standard mix builds 93% of the image out of the noisy channels and throws away the good one.

Instead the weights came from measured tonal range,

$$w_c = \frac{r_c}{\sum_k r_k}, \qquad r_c = \text{range of channel } c$$

blended with the perceptual weights, which lands around $(0.15, 0.29, 0.56)$. Blue does most of the work, because blue is where the picture actually survived.

So the monochrome image is not a reduction of the colour image. It is a *different, better-conditioned estimate of the same scene luminance*.

Which suggests the obvious move: keep it.

## Transplanting the luminance

LAB is convenient here precisely because it splits the thing into a lightness channel and two chroma channels that you can source independently. So:

- take $L^*$ from the information-weighted monochrome,
- take $a^*, b^*$ from the colour pipeline,
- recombine.

Nothing here is invented. Both halves are measurements of the same frame, just computed along different paths, and I am picking the better-conditioned one for each component.

![restored neutral, and the fused version](https://raw.githubusercontent.com/DA1729/m35_photo_edit/main/docs/neutral_vs_fused.jpg)

Left is where the last post finished. Right is the fusion. The window frames, the road sign, the lettering on the sign at the top -- those were mush before.

Then the first surprise: the result came out almost monochrome. Mean LAB chroma of about 3.7, which is essentially grey. The colour pipeline had spent so much effort suppressing amplified colour noise that there was barely any chroma left to transplant. Fine when it is sitting under a matching noisy luminance. Very obvious when it is sitting under a clean one.

So push it back up. Multiply $a^*$ and $b^*$ by a gain, with a soft ceiling so nothing goes electric:

$$C' = C_{\max}\tanh\!\left(\frac{C}{C_{\max}}\right), \qquad C = \sqrt{a^{*2} + b^{*2}}$$

A gain of about 3.5 brings mean chroma to roughly 9.6 and the frames start looking like colour photographs again.

## The second surprise, which is more interesting

A uniform gain makes everything worse in exactly one place.

The residual cast that survived the light-leak correction is *also* chroma. Multiplying all chroma by 3.5 multiplies the leftover green and yellow blotches by 3.5 too. The frames with the worst leaks got noticeably uglier, which is a slightly annoying way to learn that your enhancement is not selective.

The fix reuses something already built. The restore pipeline has a per-pixel confidence map -- structure over noise, separated by scale -- that decides where colour is a real measurement rather than amplified grain. So make the gain a function of it:

$$g(x) = g_{\text{floor}} + (g - g_{\text{floor}})\,w(x)$$

where $w(x)$ is the normalised confidence. Where there is structure, full gain. Where the frame is dead fog, the gain stays near 1 and the residual cast is left where it is instead of being tripled.

![uniform gain, and confidence-weighted gain](https://raw.githubusercontent.com/DA1729/m35_photo_edit/main/docs/gain_weighting.jpg)

Left is a flat 3.5x everywhere. Right is weighted. Look at the lower-left corner and the band down the right side -- same photograph, but the parts of it that are not really data have stopped shouting.

I like this one because the rule is not "enhance colour". It is "enhance colour in proportion to how much you believe it".

## Some local contrast while we are here

The luminance channel is now clean enough to take some abuse, so it gets multi-scale local contrast rather than another round of CLAHE. Build a Gaussian pyramid, take the difference between each level and the upsampled level above it,

$$D_i = G_i - \text{up}(G_{i+1})$$

scale each $D_i$ with a clamp, and reconstruct. Boosting detail band by band gives you local contrast at every scale at once, and because the clamp is on the band rather than on the final pixel, it does not carve halos around hard edges the way a single large-radius unsharp mask does.

## Measuring whether any of this actually helped

Here is the part I wanted to check honestly, because "it looks better" is not a measurement.

![noise against contrast, restored versus fused](https://raw.githubusercontent.com/DA1729/m35_photo_edit/main/docs/fusion_tradeoff.png)

Grey is the restored output, blue is the fused version, and each arrow is one frame. Down means less grain, right means more local contrast. Normally these trade off against each other -- that is the whole difficulty of the problem -- so the interesting thing is that all eight arrows point down *and* right. Mean noise $\sigma$ goes 1.98 to 1.77 while mean $L^*$ contrast goes 21.5 to 24.2. Eight out of eight improve on both axes at once.

That is what you would hope for from replacing an estimator with a better-conditioned one, rather than from turning knobs.

The red triangles are the versions I actually ship, and they sit well above everything else at around $\sigma \approx 3.3$. That is not the pipeline failing. That is grain I deliberately add back at the end, along with halation and a slight vignette, because a fully denoised scan of 35mm film looks like a photograph of a photograph. It is a choice, and it is worth being clear that it is a choice rather than quietly reporting the pre-grain number and letting you assume the shipped files are that clean.

## And then the part that is just for fun

With a clean luminance channel sitting there, the toning options open up. Split-toning maps shadows and highlights to different points in $a^*b^*$, interpolated by lightness:

$$\begin{pmatrix} a^* \\ b^* \end{pmatrix}(x) = w_{\text{lo}}(x) \begin{pmatrix} a_s \\ b_s \end{pmatrix} + w_{\text{hi}}(x)\begin{pmatrix} a_h \\ b_h \end{pmatrix}$$

Cool shadows and warm highlights gives you selenium. Push both toward negative $b^*$ and you get cyanotype. There is also a bleach bypass (desaturate, then screen a high-contrast luminance back over it) and a cross-process curve set.

![fused, bleach bypass, selenium, cyanotype](https://raw.githubusercontent.com/DA1729/m35_photo_edit/main/docs/grades_strip.jpg)

And honestly, these are the best images on the roll:

![cyanotype and selenium](https://raw.githubusercontent.com/DA1729/m35_photo_edit/main/docs/mono_grades.jpg)

Which is a slightly funny conclusion to arrive at after two posts of work on colour correction. The reason is not mysterious though. Every remaining problem on this roll lives in chroma -- the residual leak tint, the blotching, the zones that neutralise inconsistently. A monochrome split-tone throws that entire failure mode away and keeps the one channel that came out of the camera in reasonable shape. The tonality these frames have, flat and misty and low contrast, happens to be exactly what an alt-process print wants anyway.

## Where it stands

The colour is genuinely better than it was. On the frames without severe leaks it is a real photograph now rather than a recovered one. On the leaked frames it is improved and still compromised, and no amount of further processing is going to change that, because the measurement is not there to recover.

The thing I keep noticing across both of these posts is how much of it came from measuring first. The luma transplant is not a clever filter, it is a consequence of having written down that blue had 200 levels and red had 44. The confidence weighting is not a new technique, it is a map that already existed being used for a second purpose. Neither idea would have occurred to me from looking at the pictures.

Everything is in the same repo, `experiment.py` alongside `restore_film.py`: [github.com/DA1729/m35_photo_edit](https://github.com/DA1729/m35_photo_edit). The restoration outputs are untouched -- the experiments write to their own folder, because I wanted to be able to go back and see what the honest version looked like.

peace. da1729
