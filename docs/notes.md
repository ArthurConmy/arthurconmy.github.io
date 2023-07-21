---
layout: page
title: Notes
permalink: /notes/
---

This page will hopefully store my notes on various maths courses, and will hopefully grow over time.

### Maths

## Part II

# Mathematics of Machine Learning <a href="../assets/MML/MML.pdf" target="_blank">[link]</a>, complete.

This is a new course, and I am not aware of any complete sets of notes available other than these at present. Highlights are the much nicer references to previous equations than in existing notes, and also the elementary proof of the full form of Hoeffding's Lemma.

## Part IB

# Geometry <a href="../assets/Geometry/Geom.pdf" target="_blank">[link]</a>, near complete.

I found this course hard, and found a number of sections of the course covered at a rapid speed. My goal in these notes is to distill the important intuitions, but also fill in details in the sections I was less clear about.

All sections of the course are covered in some detail, except the final section on hyperbolic geometry.

# Analysis <a href="../assets/Anal.pdf" target="_blank">[link]</a>, very rough.

<!-- Currently only covers a useful trick to verify the general derivative form of Cauch -->

Currently covers some more background on the differentiation section of Analysis and Topology, as well as a technical detail that allows the general Cauchy's Integral Formula for derivatives to be deduced.

### Machine Learning

# AI Safety Workshop (CUAI) <a href="https://colab.research.google.com/drive/1Yfk1a4EkCEEddzW-iNfIaH_8-yPK3Ddo" target="_blank">[link]</a>

A workshop (with exercises!) on the 'off switch' problem in AI. Explores 'safe interruptibility': one way of thinking about this problem in the context on current Reinforcement Learning systems.

## Cambridge Societies

I spent some time at Cambridge in different societies:

* Founded, was president of, and put [talks](https://web.archive.org/web/20211022220732/https://uccps.soc.srcf.net/talksarchive/) together with the UCCPS. 
* Into [Effective Altruism](https://web.archive.org/web/20210925210707/https://www.eacambridge.org/about).
* Organised [workshops](http://web.archive.org/web/20220121045119/https://cuai.org.uk/workshop-gpt-3-and-codex/) with the [CUAI](http://web.archive.org/web/20220121182105/https://cuai.org.uk/committee/).
* A lot of [running](https://web.archive.org/web/20210925205921/https://cuhh.soc.srcf.net/about/committee/juniormembers/ez-run-organisers/), too.

# Drafts

## Backpropagation and einsum

This post assumes familiarity with the `einsum` function. A great introduction can be found <a href="https://rockt.github.io/2018/04/30/einsum">here</a> which describes more than what we'll need in this blog post.

### The setup

I was recently preparing for machine learning interviews, and was told to have familiarity with Python 3.7 and numpy. It's not surprising to need to code in Python for ML interviews, but using numpy? It seemed a good idea to write some forwards and backwards passes for common functions in neural networks. <b>It turns out that when you use einsum this is a lot less of a headache than you might have expected!</b>

Let's suppose we have some linear module in a neural network with a weight `W` computing some example (intentionally complicated) operation 

```
Y_{bij} = \sum_{k, l} X_{bik} W_{jkl}
```

This may seem scary, but of course we'll just compute this as 

```
Y = torch.einsum(
    "bik,jkl->bij",
    X,
    W,
)
```

Now, how about writing the backwards method for this linear layer?

Previously, I would have gotten out pencil and paper to work out this mess. Now, we'll use a result (proven later in this post) to make this process much faster. Recall that the backwards pass takes some upstream gradient `dL_dY` as input and needs to calculate `dL_dW` to do backprop on this weight matrix and `dL_dX` the upstream gradient for further backprop (here, `L` is the loss we're optimizing).

Indeed, we can immediately calculate

```
dL_dW = torch.einsum(
    "bik,bij->jkl",
    X,
    dL_dY,
)
```

and 

```
dL_dX = torch.einsum(
    "jkl,bij->bik",
    W,
    dL_dY,
)
```

by simply permuting the three terms in the einsum string, and inserting the similarly shaped tensors (note that we only compute gradients from `dL_dY`, rather than `Y`).

Why is this true? Let's start with the `dL_dW` expression. The key idea is to use the chain rule, so that 

```
dL_dW = \sum dL_dY * dY_dW
```

where the * represents the product, and we sum this over all possible `Y` indices. I think the easiest way to see the result is then to imagine fixing one particular `Y` index and calculating the gradient of that `Y` value with respect to the weight tensor `W`. Indeed, it's now clear that we can compute `dL_dY` by looking at the relevant indices in the `W` tensor, and since we're summing over everything, we get the einsum expression at the end of the day. Similarly for `dL_dX`, here we can use `dL_dX = \sum dL_dY * dY_dX` and this explains the `W` that appears in the einsum expression for the calculation here.

<!-- In future, I'd like to expand this when I know more analysis. -->
<!-- # Principles of Statistics <a href="../assets/PoS/pos.pdf" target="_blank">[link]</a>. created 22nd October 2021. -->
<!-- I have just started writing up notes for the Principles of Statistics course. -->