---
layout: post
title: Motivating mesa-optimization
date:   2021-09-19
permalink: "/mesa_opt/"
---

This post is written as part of the [AGI Safety Fundamentals Course](https://www.eacambridge.org/agi-safety-fundamentals), organised by [Effective Altruism Cambridge](https://www.eacambridge.org).

One of the first technical concepts the AGI Safety Fundamentals course covers is *mesa-optimization*. At first, I struggled to find motivation for the concept: not only could I not see why it was likely to occur in an optimization process (I call this the 'likelihood uncertainty'), but also I could not see why it would be more problematic than an optimization procedure not itself producing an optimizer (I call this the 'importance uncertainty').

In this post, I do not explain mesa-optimization. Instead, I write a background that I wish I had had before being introduced to the concept. The intended reader has *attempted* to understand mesa-optimization, though had a uncertainty or confusion similar to at likelihood and importance uncertainty I describe above.

In the alignment newsletter that covers the original mesa-optimization paper[^fn1] a good definition of optimization, along with a lack of existing example of mesa-optimization are two big sources of confusion regarding optimization. These will therefore be the two topics covered.

# Optimization

To understand mesa-optimization, optimization must first be understood. A working definition is that from LessWrong[^fn3]: 'any kind of process that systematically comes up with solutions that are better than the solution used before'.

There are two immediate takeaways from this defintion. The first is of *solutions*, i.e optimization is a tool applied to solving problems. Notably, solutions can be found to problems without optimization ever having to be used (for example, caps to water bottles solve a problem without optimizing anything[^fn4]). The second is that an optimizer needs a notion of *better*, that is to say it must be able to rank solutions. 

**Example**: Bayesian updating is an optimization process, since it finds solutions (posterior distributions) which are better (return lower loss than previous solutions)[^fn5].

Why might optimization be problematic in AGI safety?

Some thought experiments in AGI (and philosophy more generally) produce confusing results due to the reliance on optimization procedures necessarilly requiring a ranking of solutions. I think the repugnant conclusion[^fn2] and other population ethics problems rely on a ranking between different worlds, which can lead to the conclusions of what are the best worlds being very counterintuitive. If we train an AGI to maximise human happiness, will it do this by simpling maximising our experience of pleasure, so in an intoxicated state our momentary subjective experience will be of great pleasure? If so, is this concerning[^fn6]?

Additionally, the population ethics example only scratches the surface of the problem of value learning; it seems very difficult (citation needed!) to quantify human values into an objective such that an optimization procedure could rank different events as reflecting human values better or worse, or worlds as being more closely aligned with human values or not.

This gives a reasonable answer to the importance uncertainty. With regard to the likelihood uncertainty, one motivation is provided in the Risks From Learned Optimization[^fn7] work: data compression.

This line of reasoning goes as follows: if a system is optimizing for a complex, long term goal, then simple heuristics that continually provide an approximation of the long term goal will be efficient representations would be desirable solutions to the optimization process. Of course, we are reasoning about very general systems and hence this line of reasoning is difficult to provide evidence for, let alone to prove theoretical results about. This sits in my mind as a reasonable argument due to the bitter lesson[^fn9] and the power of existing systems that have relatively simple architectures to do complex things. But this may not convince all readers, and for this reason an example of mesa-optimization would be incredibly useful.

# Examples

In this section, I brainstorm some ideas about how a real-world example of mesa-optimization could be found.

Optimization procedures that produce planning algorithms (that are themselves optimization procedures) and biological evolution producing humans, who themselves optimize for various goals, are offered as two examples in Risks From Learned Optimization, though as the Alignment Newsletter notes, an example that doesn't arise by design (optimizing planning algorithms can't *not* produce an optimizer), but still within a deep learning architecture would be highly desirable.

It seems most likely that an example of mesa-optimization could be found in a reinforcement learning setting. This is because reinforcement learning deals with sequential actions and therefore optimization (which must produce *different* solutions) could be useful, unlike for example in a binary classifier.

The data compression motivation suggests that it is plausible that in a game scenario, which reinforcement learning has widely been applied to, optimization could lead to an agent being produced that optimizes for a proxy goal that provides less sparse or noisy signal than the goal of the optimization process. I am curious as to whether an experiment could be designed to constrain architectures of agents such that mesa-optimization could indeed arise.

# Footnotes

[^fn1]: [Alignment Newsletter #58.](https://www.lesswrong.com/posts/XWPJfgBymBbL3jdFd/an-58-mesa-optimization-what-it-is-and-why-we-should-care)
[^fn3]: [LessWrong.](https://www.lesswrong.com/tag/optimization)
[^fn5]: [CFAR has an online exercise on applying Bayesian reasoning to (human!) decision making.](https://programs.clearerthinking.org/question_of_evidence.html#.YUejt3WYVNg)
[^fn2]: [Wikipedia.](https://en.wikipedia.org/wiki/Mere_addition_paradox)
[^fn6]: [Linked is the 'experience machine' thought experiment. With regard to AGI, I heard about this through 'perverse instantiation', in Bostrom's 'Superintelligence'.](https://en.wikipedia.org/wiki/Experience_machine)
[^fn4]: [Daniel Filan.](https://danielfilan.com/2018/08/31/bottle_caps_arent_optimisers.html)
[^fn7]: [Risks from Learned Optimization.](https://www.alignmentforum.org/s/r9tYkB2a8Fp4DN8yB/p/q2rCMHNXazALgQpGH)
[^fn9]: [Incomplete Ideas.](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)