---
layout: post
title: Technical AI Alignment
date: 2022-01-20
permalink: /technical_ai_alignment/
---

*This post was written based on my favourite parts of the [MLAB](https://www.redwoodresearch.org/community-and-team-growth) bootcamp.*

*To make it (hopefully) interesting for those with and without machine learning familiarity, background details are enclosed in **Background** collapsibles, and technical details are enlcosed in **Technicality** collapsibles.*

<img src="../assets/mlab.png">

If you believe AI (through machine learning, or otherwise) might transform society, you might also be concerned whether such transformation is very good or [very bad](https://en.wikipedia.org/wiki/Existential_risk_from_artificial_general_intelligence). One approach to working on this problem is *technical AI alignment*, which in this post I define to be working with existing ML systems. 

This post is accompanied with a colab notebook [here](https://colab.research.google.com/drive/10DkmAwc7FXokD1_scwvvWEav0F9egtK5?usp=sharing).

## Interpretability 

<details>
<summary><b>Background</b></summary>
<p>
One problem with existing ML systems is that they are often used as a <i>black-box</i>, performing a task of use to us, while we don't understand exactly how it does this. A particularly scary <a href="https://www.pulmonologyadvisor.com/home/topics/practice-management/the-potential-pitfalls-of-machine-learning-algorithms-in-medicine/">example</a> of this involved software in medicine recommending patients with asthma were *less* likely to develop complications from pneumonia than the baseline of patients with pneumonia.
</p>
</details>


*Interpretability* work aims to understand what and how ML systems are learning from data.

# Computer Vision 

<details>
<summary><b>Background</b></summary>
Computer vision affects us daily (if we use facial recognition software to unlock our phones) and is likely (e.g self-driving cars) to be one of the most economically important application of ML in the near future. A look under the hood suggests that computer vision systems 'see' the world from how we do.
</details>

*Just read [this distill article](https://distill.pub/2017/feature-visualization/), it's fantastic*

We can see this through feature visualization with optimization. We can isolate neurons[^fn1] in [InceptionV1](https://microscope.openai.com/models/inceptionv1?models.technique=deep_dream), a network trained to [classify images](https://en.wikipedia.org/wiki/ImageNet#History_of_the_ImageNet_challenge) and then optimise input images to maximise their sensitivity to such input images. The results are very unlike natural images, and offer an insight into the psychadelic world of the inside of computer vision models[^fn2]:

Optimized Image            |  Similar Dataset Examples
:-------------------------:|:-------------------------:
![](../assets/MLAB/CatBonnet.png)    |  ![](../assets/MLAB/CatBonnetDataset2.png)

However, it's worth noting that this approach is *hard* to get working. Running optimization in the naivest way possible, initialising a random image and then optimizing for output in one of ImageNet's 100 classes results in ... something ... but certainly nothing like any natural image (from here on, all experiments can be repeated in the colab notebook):

Randomly Initialised Image            |  Naive Optimization, 1000 steps
:-------------------------:|:-------------------------:
![](../assets/MLAB/NaiveRandom.png)    |  ![](../assets/MLAB/NaiveOptim.png)

A really nice partial solution is to optimize over the Fourier basis of the image rather than the natural pixel basis:

<details>
<summary><b>Technicality</b></summary>

<p>
In computer vision, we generally optimize over the <i>C x H x W</i> vector space of images, with one dimension per pixel per channel. However, this is a fairly unnatural basis over which to optimize, since it considers adjacent pixels completely independently, which in part causes the noisy, neon images seen above. If we instead consider the Fourier basis associated with the pixel basis, we have a basis (i.e we can reproduce any image) which, each individually are continuous images rather than isolated pixels:

<img src="https://images.slideplayer.com/24/7284448/slides/slide_31.jpg">
</p>

<p>
This <i>still</i> leads to very noise images when initialised, however, since enough of a proportion of the Fourier basis still has a high frequency. We can mitigate this by rescaling a basis vector

<img src="https://latex.artofproblemsolving.com/texer/m/mpnvggsn.png?time=1642816751484">

by dividing by a factor

<img src="https://latex.artofproblemsolving.com/texer/m/mpnvggsn.png?time=1642816824292">,

the intuition being that this will cause the norms of the gradients of these 2D function to all be 1 at the origin.

The implementation of such a Fourier inversion are non-trivial: the discrete Fourier transform fundamentally operates on complex vector spaces, and our images only make sense as real vector spaces. There are implementations (https://pytorch.org/docs/stable/generated/torch.fft.irfft2.html that work around this, but it is still a good exercise TODO
</p>
</details>



# Footnotes

[^fn1]: as addressed in the next footnote, neurons in InceptionV1 are neurons by name only; these can just be thought of as intermediate high-dimensional vectors in the mapping from an image to the one-dimensional category of image that that image belongs to in ImageNet.

[^fn2]: it's worth emphasis that despite the fact that almost all state-of-the-art approaches to problems on which machine learning succeed use neural networks, the extent to which these are biologically inspired (and by extension, 'human-like') is [pretty weak](https://shlegeris.com/2019/08/20/cnn.html). Additionally, progress on such problems seems to be driven by [compute](http://www.incompleteideas.net/IncIdeas/BitterLesson.html), rather than the either encoding (human) intuitions or studying how humans learn things.