---
layout: post
title: Automatic circuit discovery
date:   2022-12-18
permalink: "/automatic_circuit_discovery/"
---

<b> Warning: </b> This blog post describes scattered early research progress on automating circuit discovery. With several collaborators, we expanded this work greatly and published our work: <a href="https://arxiv.org/abs/2304.14997">https://arxiv.org/abs/2304.14997</a> . The code is under development (as of May 31st 2023) at <a href="https://github.com/ArthurConmy/Automatic-Circuit-Discovery">https://github.com/ArthurConmy/Automatic-Circuit-Discovery</a>

<h1>Automatic patching for discovering circuits</h1>

<i>Work done at Redwood Research, with Haoxing Du. The ideas are due to discussions with Alexandre Variengien and Jacob Steinhardt. Please send feedback to arthurconmy@gmail.com</i>

<i>Check out Chris Mathwin's interpretability hackathon project that uses this tool <a href="https://itch.io/jam/mechint/rate/1889871">here</a></i>

<p>I recently finished working on the <a href="https://arxiv.org/abs/2211.00593">IOI paper</a>, which was the most exciting project I have ever been part of. Our work finds a circuit that performs a task in a language model. This blog shares how this approach can be generalized, and some code <a href="https://colab.research.google.com/github/ArthurConmy/Easy-Transformer/blob/main/AutomaticCircuitDiscovery.ipynb">here</a> (see footonote 1) for anyone interested in doing this. This post assumes some familiarity with <a href="https://transformer-circuits.pub/2021/framework/index.html">language model interpretability</a>.</p>

<img src="https://i.imgur.com/3ONKQBB.png">


Code output on IOI! <b style="color:green;">K</b>, <b style="color:red;">Q</b> and <b style="color:blue;">V</b> composition are labelled. The threshold is a 7% change in logit difference. The automatic circuit discovery discovers Name Movers (all heads that write to "resid out"), S Inhibition heads (8.6) and the induction mechanism (4.11 -> 5.5).

<h2>Tasks</h2>
Why are we studying tasks that language can perform? The most impressive models are trained on diverse datasets, with 50,000 or more distinct tokens. This means that understanding in general what the function of components of these models do is intractable, if not impossible. Instead, <b>consider a "task" defined by a dataset of prompts (see the next section) and single token completions to these prompts.</b> We can i) verify that a language model predicts the correct token completion to all the prompts and then ii) begin work interpreting which language model components are responsible for this.

By studying many such tasks, we hope to eventually be able to understand the most impressive models. This could be by directly understanding all constituent components of simpler systems, or by gaining the intuitions that are required to do productive work on the more complex models.
<h2>Circuits</h2>
In <a href="https://arxiv.org/abs/2211.00593">the IOI paper</a> we define circuits "top down" rather than <a href="https://distill.pub/2020/circuits/zoom-in/">bottom up</a> (see section 2.1 of the paper). This lends itself easily to automation. All we need, in addition to the dataset-with-completions introduced above are 

<ol>
    <li>labels for particular important tokens that occur in all inputs</li>
    <li>a "baseline" dataset of prompts that have the same labels, but different behavior</li>
    <li>a metric for the model behavior</li>
</ol>

For example, suppose dataset of sentences with completions contains sentences like "Last month it was February so this month it is" that have completions like " March". Then 1) the important labels could be the token positions where "Last month", "February", "this month" and the end token " is" are present. 2) the baseline dataset[^fn2] could be sentences like "This time it is here and last time it was", that would presumably produce similar activations to the main dataset, but don't introduce any context about months, or that the next word should be about a date in future. 3) a metric could be the difference in the logits the model places on " March" compared to " February", as this will roughly measure how well the model knows how to complete the sentence correctly.

<h2>Implementation</h2>

<i>Warning: we are developing automatic circuit discovery with Redwood's in-house tools, which are hopefully soon going to be open-sourced. The prototype implemented here is not mantained and I would advise waiting for Redwood's software release.</i>

See the notebook <a href="https://colab.research.google.com/github/ArthurConmy/Easy-Transformer/blob/main/AutomaticCircuitDiscovery.ipynb">here</a> for an exploration of the path patching applied to the IOI case.

The method is as follows: we iteratively build the circuit by starting with a single node that's the END position. We then look at all the direct connections from previous attention heads and MLPs. For each connection, we replace it with its value on the new dataset, and see if this results in a significant change in the logit difference. If this is above some threshold, we include the edge in the graph.

When we look at nodes other than the end node, we consider inputs from previous positions too (if the node is an attention head) and we also only propagate changes through the edges that we've found.
<h2>Limitations</h2>
There are at least two limitations to this work:

1. There are cases where the way we add nodes to the graph is problematic
2. We don't have a way of converting the subgraphs into something that is automatically human-understandable

1.: consider a model where all attention heads `h1` to `h5` add 1 to the (one-dimensional) output, like so:

<img src="https://i.imgur.com/Uv5vG9n.png">

Then suppose we set a threshold of 2. Here, none of the edges in the above diagram have an effect size at least 2, and so ACDC would find NO edges relevant to the task of producing a large output. In general, these failures occur when behavior is distributed, not sparse.

2.: this automated approach assumes that the "units" of interpretability are the attention heads and MLPs. However, both induction heads and the head classes in the IOI paper are strong examples of cases where individual heads are not the correct unit to study model behavior with, as several heads are identical (and should be grouped together). Not aggregating components is a problem when we want to produce explanation that have clean human causal graphs, for example for verification by <a href="https://www.alignmentforum.org/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing">causal scrubbing</a>.

<h2> Footnotes </h2>

[^fn1]: the code has struggled in Google Colab for me, if you have access to a better machine you may see performance improvements. If you are interested in collaborating on improving the state of this code, please reach out.

[^fn2]: there is a lot of freedom in choosing the baseline dataset. The reason we need a different dataset at all is that this is essential for verifying the <a href="https://en.wikipedia.org/wiki/The_Book_of_Why#Chapter_1:_The_Ladder_of_Causation">causal</a> role of model components.