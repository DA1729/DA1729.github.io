---
title: "Recovering a Ruined Roll"
date: "2026-08-08"
---

# Recovering a Ruined Roll

So, I have gotten this Kodak M35 analog camera on me, and I have been roaming around cities -- Leuven, Brugge, Ghent, Lier -- dressed like a homeless person, always a little drunk, headphones on and this camera. And CLEARLY I do not know how to work with it.

![A frame straight off the roll](https://raw.githubusercontent.com/DA1729/m35_photo_edit/main/scans/frame_04.jpeg)

There is a church tower in there, against a sky. You would not know it.

These are the few recoverable shots out of a 36-exposure roll.

The whole roll came back like this. Eight frames, every one of them a flat yellow-green wash. My first instinct was that the lab had ruined it, or that the film was expired, or that light had gotten in somewhere it should not have. Probably some combination. But before deciding it was unrecoverable I wanted to actually look at the numbers, because "this looks bad" and "this contains no information" are very different claims.

They turned out to be very different claims here.

## Looking at the histogram first

The useful thing about a digital scan is that you can stop guessing. Split it into channels and look at where the values actually sit.

![Channel histograms, before and after](https://raw.githubusercontent.com/DA1729/m35_photo_edit/main/docs/histograms.png)

The left panel is the frame as scanned. Red and green are jammed into a narrow spike between roughly 150 and 200. Blue is spread across almost the entire range.

Concretely, per channel, taking the 0.6th and 99.6th percentiles as the endpoints:

| frame | R range | G range | B range |
|---|---|---|---|
| frame_05 | 44 | 49 | 200 |
| frame_07 | 50 | 58 | 223 |
| frame_08 | 37 | 39 | 197 |

So red is carrying 37 to 50 distinct code values out of 255. That is

$$\log_2(44) \approx 5.5 \text{ bits}$$

against blue's

$$\log_2(200) \approx 7.6 \text{ bits}$$

This is the entire problem in one line. It is not that the picture is missing. It is that two of the three channels have been compressed into a sliver at the top of their range by an additive yellow veil, while the third channel is more or less fine.

Which also tells you the shape of the fix: whatever the correction is, it has to multiply red and green by something like 3 to 4x. And a 4x gain on a channel with 45 usable levels means your effective quantisation step goes from 1 to about 5.7. You will be manufacturing banding and noise. That constraint governs basically every decision that follows.

## Auto-levels is not enough, and this surprised me

The obvious first move is per-channel auto-levels: find each channel's black point and white point, stretch each to fill [0, 255]. Textbook. I did it in linear light rather than gamma-encoded values, since fog is additive light landing on the scan, so subtracting it is a linear-space operation.

It helped. It did not fix it. The image was still noticeably yellow, and when I measured the residual in LAB I got a b\* bias of **+32 to +43** through the midtones. That is not subtle. b\* is the blue-yellow axis; +40 is a strongly yellow image.

This confused me for a while, because if you have matched the black points and matched the white points, what is left?

The midtones are left. Matching two points on a curve does not match the curve.

A colour negative has three dye layers, and on a fogged, badly exposed roll they do not have the same effective contrast. So each channel's transfer function has a different shape between the endpoints. You can pin both ends and still have the middles pull apart. No amount of gain-and-offset -- which is an affine map, two degrees of freedom -- can fix a difference in curvature.

So add a third point. For each channel, after normalising the endpoints, find the median $m_c$ and apply a gamma $\gamma_c$ that maps it onto a shared target $t$:

$$m_c^{\gamma_c} = t \quad \Longrightarrow \quad \gamma_c = \frac{\ln t}{\ln m_c}$$

with $t$ taken as the geometric mean of the three medians, so the overall brightness stays put and only the *relative* bias between channels moves. Then blend toward identity by a strength $s$:

$$\gamma_c \leftarrow 1 + (\gamma_c - 1)\,s$$

so a frame that is genuinely warm does not get scrubbed grey.

Measured on frame_05: medians after endpoint matching were (0.492, 0.356, 0.145), giving gammas of about (1.65, 1.17, 0.67). The residual b\* went from **+32 to +5**.

The right panel of the histogram above is what that looks like. Three channels lying on top of each other instead of three channels living in different neighbourhoods.

## The part that actually made it look like a photograph

Cast fixed, the frames were still ugly, because now I was looking at a 4x amplification of red and green quantisation noise. Grainy, blotchy, and with colour speckle everywhere.

The fix came from re-reading the table above. Blue has 200 usable levels. Red has 44. Almost everything that looks like "detail" in the red channel at that point is quantisation steps and JPEG artefacts. The actual structure of the scene -- edges, texture, the stonework on a spire -- is sitting in blue.

So: use blue as a guide, and fit red and green to it locally with a guided filter. For a guide $I$ and input $p$, over a window $\omega_k$, the guided filter solves for the linear coefficients

$$a_k = \frac{\operatorname{cov}_{\omega_k}(I, p)}{\operatorname{var}_{\omega_k}(I) + \epsilon}, \qquad b_k = \bar{p}_k - a_k \bar{I}_k$$

and outputs $q_i = \bar{a}_i I_i + \bar{b}_i$. It is a local linear regression of the noisy channel onto the clean one. Nothing is copied between channels -- the output of each channel is still built from its own local mean -- it just stops that channel from inventing edges the guide does not agree with.

This is the same reasoning as chroma subsampling in every video codec ever written: keep detail where it was measured, let the weak channels carry colour only.

Measured luma noise went from $\sigma \approx 4.5$ to $\sigma \approx 2.0$ (in 8-bit units), which is the single biggest visual improvement in the whole pipeline, and it cost nothing in sharpness.

![frame_05, as scanned and restored](https://raw.githubusercontent.com/DA1729/m35_photo_edit/main/docs/frame_05_pair.jpg)

There is a building there. There are windows. There is a taxi sign.

## Declining to invent colour

One more idea I liked, mostly because it is a statement about epistemics rather than about filtering.

In the regions where the light leak was worst, there is genuinely no signal left. Whatever colour you compute there is amplified noise. Most pipelines will happily hand you a confident hue for those pixels anyway.

So I built a per-pixel confidence map. Separate structure from noise by scale: what survives a small blur is grain, and the band between a small and a large blur is real image structure. Then

$$\text{conf} = \frac{\sigma_{\text{structure}}}{\sigma_{\text{structure}} + \lambda\,\sigma_{\text{noise}}}$$

Where confidence is low, chroma fades toward neutral. Where it is high, colour is untouched.

The nice property is that it distinguishes the two cases that naively look identical. A dead flat fogged patch has no structure but plenty of noise, so it scores near zero. A genuinely smooth surface -- a clear sky, a painted wall -- is smooth at *both* scales, so its noise term is small too and it keeps full colour.

This does not remove anything real. It refuses to claim colour it cannot support. Given that the whole roll is marginal, that felt like the honest default.

## Light leaks have hard edges

Look back at the frame at the top of the post: there is a sharp-edged yellow band down the right side of it. I first tried removing the low-frequency chroma drift with a big Gaussian blur, which is the standard move for a vignette.

It did nothing useful, and the reason is obvious in hindsight: a Gaussian smears straight across a hard boundary, so the estimate of the leak is wrong on both sides of it. The estimator has to be edge-aware. Using the same guided filter with luminance as the guide, the zonal colour spread (b\* standard deviation across the frame) dropped from 25.5 to 17.0 on that frame, and from 13.8 to 5.8 on frame_07.

Luminance is never touched by this step and the shift is hard-capped, so it cannot erase image structure -- it can only move colour.

## Three outputs, because one is a lie

The program writes three versions of every frame.

`_neutral` is the strongest attempt at true colour. `_film` is a full correction followed by a deliberate warm grade -- and that ordering matters. My first version got "warm" by just under-correcting the cast, which looks fine on a mildly tinted frame and stays broken on a frame like these. The cast here is a defect, not a mood. Fix it, then add warmth on purpose.

![neutral and film](https://raw.githubusercontent.com/DA1729/m35_photo_edit/main/docs/frame_07_pair.jpg)

`_bw` is the one that actually works.

And the black and white conversion is where the earlier measurement pays off again. A standard luma mix is $0.21R + 0.72G + 0.07B$, which on this roll would build the image almost entirely out of the two channels carrying 5.5 bits, and throw away the one carrying 7.6. So instead the channels are weighted by how much tonal range each one actually has, which comes out around $0.15R + 0.29G + 0.56B$.

![frame_04, as scanned and as black and white](https://raw.githubusercontent.com/DA1729/m35_photo_edit/main/docs/frame_04_bw_pair.jpg)

That is the frame from the top of the post. Tracery, pinnacles, the lot.

## What it cannot do

All eight frames come out graded `recovery: poor` by the program's own diagnostics, and I left that in rather than tuning the grading until it flattered me. Red and green hold 37 to 102 code values. You cannot multiply your way out of that. The residual banding is real, some of the leaked regions stay slightly tinted, and no amount of processing changes the fact that the information was gone before I opened the file.

I could have made the colour versions look cleaner by leaning harder on denoising and saturation. It would have been smoother and less honest.

The thing I keep coming back to is that almost every good decision here came from measuring something before changing it. "Looks yellow" gets you to auto-levels, which does not work. "b\* is +40 after endpoint matching" tells you the problem is curvature, not offset, and that is a different fix. "Red has 45 levels, blue has 200" tells you which channel to trust. The picture only started coming back once I stopped looking at it and started looking at its statistics.

Code is here: [github.com/DA1729/m35_photo_edit](https://github.com/DA1729/m35_photo_edit)

Next roll I will try to keep the camera closed until I mean to take a picture.

peace. da1729
