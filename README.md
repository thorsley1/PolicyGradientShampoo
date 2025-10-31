## What's going on with this repo?

I'm trying to implement a Shampoo style optimiser for simple policy gradients.
Currently using Craftax as the environment, though can be easily switched out for any gymnax environment.
I've tried to keep things as simple as I can and have everything be JAX-idiomatic.
That said, this is really a personal research repo so I e.g. haven't really been writing the best most maintainable code, just trying to prioritise on speed. 



## Some rough research motivations / thoughts
### Apologies for the ramblingness. Feel free to ignore

The goal of this research repo is to try to move ideas which have been successful in distribution matching - i.e. things trained with cross-entropy loss like supervised / unsupervised learning and notably large scale pretraining - and test them in the case of policy gradients.

In particular, I haven't really seen much work on the second order optimisers such as Shampoo, SOAP, etc which have been so popular in large scale pretraining applied to any policy gradients (the only similar work being ACKTR). There are probably a whole load of reasons for why there hasn't been much work on this, however I thought it seemed like a fun thing to try out.

One nice property of experimenting with Shampoo on policy gradients is that they are on-policy (or at least one can importance sample) and there are experiments in several papers (Morwani et al 2024 comes to mind) suggesting that true Fisher approximations are much better than empirical Fisher from an off-policy distribution. 

One issue it does have is that the reward weighting of the policy gradient (e.g. GAE advantage in PPO / PPG or just return in REINFORCE) being different for different items in the batch and the contraction over the batch axis in the parameter gradient backward pass calculation means that one cannot just get the preconditioner / update gradient from the other for ~ free as one can in cross entropy losses where they're the same. There are ways to get both the accumulated log prob and policy gradients for only <= 1/3 additional compute through reuse of the activation gradient calculations and custom vjps w/ tracking the reward weighting seperately, though for simplicity I use the double backward pass because I just wanted to test things quickly instead of going straight for performance optimisations.

I think there are some general reasons to expect to see gains here - particularly with stability which is a major issue in policy gradient training, though this may require large batch sizes to see. Raising the critical batch size is another property observed in pretraining which may carry over, though it seems plausible this is less of a big deal than in pretraining because there the frontier is pushing up against the CBS due to a mix of 1. batch size scaling laws say CBS ~ independent of models size and ~ sqrt of dataset size and thus training time increases ~ sqrt of dataset size and 2. the gradient noise of distribution matching feels in principle like it should be _way_ lower than policy gradients (esp when you're just taking the hit of simple reward weightings) and so people are actually hitting the CBS which is an issue because training time is limited by the returns of algorithmic improvements and so pretraining people (probably - I don't actually know any) care about any old way to increase the CBS (data clustering being an obvious one). 1 may still hold here, I haven't seen any work on it, but I think 2 is going to mean things should be a bit better for policy gradients and optimisers will matter less than e.g. not being stupid about the reward weighting, though probably will matter at some point. [Note to self, get back on track]

Anyway, yes, the main thing I expect to be better is stability and then also the use of off-policy gradients. There's some evidence from the PPO-EWA paper that the important thing about PPOs clipping is just that it approximates a natural policy gradient. I haven't seen replication of this for REINFORCE / critic-less policy gradients, it's plausible the main mechanism for improvement there is that a natural policy gradient is also just better to allow the critic to adapt over time but it does at least seem worth looking into the more straightforward 'natural gradient descent in just good for policy learning' story. 

I probably don't expect it to help increase num_policy_epochs to above 1. I think the results I've seen from PPG, PPG Reloaded, all the REINFORCE style policy gradients seem to suggest that multiple policy epochs is ~ worthless or actively harmful in most cases (would like to see more work here).

In this project, I'm probably mainly going to be using VPG as a baseline. That's going to mean I'm not expecting to get SOTA results or anything because the returns to TD learning at small scale in simple game like environments are huge, but at larger scale TD learning isn't really used probably because of it's extremely low CBS, returns requiring loads of gradient steps, use of a replay buffer which probably just fills up HBM too much, other weird optimisation dynamics, bias which hurts more at larger scale, weakness over long time horizons with sparse rewards, etc etc. so that's the motivation for that baseline. It's also just like easier to do because I can just remove one line.


## What's the current status of this project?

I'm taking a break for a bit because I've realised I should really just actually be going for reasoning on larger models if I want to show this is actually useful for things people care about. Also a lot of the motivation is like 'ah yes this will work at moderate scale' and then just experimenting at tiny scales of game environments is sort of dumb. Because that's a bigger shift and motivation is important when you're doing things alone, I'm going to break for a month maybe to learn Pallas and then come back in early November and do this properly.