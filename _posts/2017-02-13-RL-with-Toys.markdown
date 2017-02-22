---
layout: post
title:  'Toys pave the Way'
date:   2017-02-13 10:00:05 +0100
categories: jekyll update
---

Testing GP regression on Toy data
--------------------------------

In order to move forward, some testing with easily observable toy data was necessary. This is to ensure that the fundamentals of the code were working as intended. As such, to begin with, a toy function that is linear in nature was used. The toy functions were designed such that their state evolutions can be easily visualised. As such, they each have one output dimension, and 2 input dimensions, a action and the present state. These functions were used as a simulation of some real process.

A notebook that contains a summary of the tests that were carried out can be found [here](https://github.com/janithPet/FYP/blob/master/Code/Semester1/toy_data/toy_data_complete.ipynb). Most of the code uses a class in [gympy.py](https://github.com/janithPet/FYP/blob/master/Code/Semester1/toy_data/gympy.py) named *rl_components* that creates the necessary components required to carry out the iterative learning process. One can define their own functions for the system model, policy, reward function, as well as a function that determines when the termination condition has been reached. The first three are of the form:

$$s_{t+1} = f(s_t,a_t)$$

$$a_t = g(\theta_i,s_t)$$

$$r_t = h(s_t, s_T)$$

Here $s_T$ is the termination or goal state. Note that for the ensuing results, the goal state was defined as being 100 units.

The termination function can be defined as necessary using the measurements made in the class.

Linear system; linear policy
-----------------------------
The first test was carried out for a linear toy function defined as:

$$s_{t+1} = 10a_t + s_t$$

This is a simple linear function that can reach the proposed goal state relatively quickly. Note that the actions were limited to $\pm \pi$, in order to reflect limitations in the high dimensional case of the problem.

Initially, the world model was trained with 10 randomly generated data points. Please look at the comments at the bottom regarding a possible error caused by this. The accuracy of the GP at being able to predict progressive states are shown below.

![Figure1]({{site_url}}/pictures/toy_data_1/allplot_3.svg)

As can be seen from this, the GP is able to learn the model fairly soon. This is of course attributed to the simplicity of the system. Moving from this, the RL loop was carried out 3 times, with a 100 data points being added to the training data set of the world model. The acquisition function used for this, and other test was the Expected Improvement algorithm. Furthermore, each iteration was limited to 150 optimisation attempts. For this, the policy and the reward function were defined as:

$$a_t = \theta_1a_t + \theta_0$$

$$r_t = |s_T - s_t|$$

The termination function was defined such that if the agent was able to stay within $\pm10\%$ of the goal state for 30 time steps, the agent was assumed to have achieved its goal. Alternatively, if this was not reached, a total of 300 time steps was allowed.

Following this, the policy that was found to be the best given the experiments carried out produced the following state evolution.

![Figure2]({{site_url}}/pictures/toy_data_1/final_solution_4.svg)

As can be seen, this has achieved the goal as defined by the termination function. What is fascinating is that agent was able to find a policy that allowed its state to quickly rise to the goal, then maintain its position afterwards; that is, it can have a high action when it is faraway from its goal, and then quickly reduce it when appropriate. This first order behaviour, having been achieved using a linear policy, is remarkable.

Linear System; non-linear policy
----------------------------------

Now, a different policy was used; it was defined as follows:

$$ a_t = \sum^{3}_{n=1} \theta_0e^{\frac{-(s_t - \theta_1)^2}{\theta_2^2}}$$

This is a 3 node RBF network.

As before, the policy produced by the RL loop of 3 iterations produced the following state evolution.

![Figure3]({{site_url}}/pictures/toy_data_1/final_solution_7.svg)

As can be seen, this plot isn't as 'optimal' as the previous. This is possibly due to the fact that optimiser has 9 variables to account for. However, it was noted that the optimiser completed faster for each iteration, but appears to need more iterations to achieve a more suitable policy.

Non-Linear system, non-linear policy
------------------------------------
Moving on, a simple non-linearity was added to the system model in the form of a sinusoid. The model was now defined as:

$$s_{t+1} = 20\sin(a_t) + s_t$$

The coefficient of the action was increased to 20 to allow for the state to reach the goal in a reasonable number of time steps that is similar to the linear model; the system was slowed down by the sinusoid. For this, the RBF network was used; a linear policy was found to be unable to deal with the non-linearities.

As before, a set of randomly generated training data was used; this time 100 data points were used because it was found that the non-linearity made it difficult for the model to learn. Of course, the GP does not 'see' this non-linearity, and only observes its effects in the data; in this case, it appears that the data is more sporadic than previously. Furthermore, the kernel used for initialising the GP was different for this test; reasons for this can be found in the [notebook](https://github.com/janithPet/FYP/blob/master/Code/Semester1/toy_data/toy_data_complete.ipynb), and will be discussed in a subsequent post.

![Figure4]({{site_url}}/pictures/toy_data_1/allplot_36.svg)

The RL loop was carried out for 5 iterations. The resulting policy produced the following state evolution.

![Figure5]({{site_url}}/pictures/toy_data_1/final_solution_37.svg)

As before, it hasn't reached its goal of being with $10\%$ of the goal state; but it is showing the required behaviour of quickly approaching the goal state, and attempting to maintain its state at that position. As such, it is possible that with more iterations, the agent would be able to find the appropriate policy.

Observations and Conclusions
-------------------------------
A key observation that was made was that, particularly when the non-linear policy was used, the world model must initially trained with data that were within the range of states that the system is expected to travel through to reach its goal. In this, when the training data were not within $0 \< s_t \< 100$, the optimisation algorithm has trouble finding the policies that would allow the agent to reach its goal; sometimes, it would never find a policy that would allow the agent to learn about the necessary space. This is to be expected as the GP isn't knowledgable about what to do in the needed state space.

Possible solutions to this could be to be more careful about choosing the initial training data; perhaps there needs to be some rhyme and rhythm to it. Although this could be viable, it would lead to induction bias, where our beliefs of the best solution are transferred to the agent. This for one, would be difficult in higher dimensional problems, and could also lead to less than optimal solutions. An alternative could be to define the reward function such that the optimiser is forced to widen it's search sooner. The reward function used in this example was rudimentary, and did not contain any tuneable parameters. As such, this could be the next point of investigation.

It should be further be noted that a smarter way of identifying the subset of observed data points that are to be used as training points for the world model needs to be devised. Presently, if the agent has found a regime of  parameter space that it thinks the optimal parameters reside in; the training points for the GP will be more or less similar. Thus, simply adding those new data points to the training data set could lead to computational redundancy. Perhaps a metric that uses a moving average of the average uncertainty through cascaded predictions within particular regions of the state space to identify if more data points are needed in that region can be used (does that sentence make sense). That is, if the model is quite certain about making predictions for a particular set of input parameters through a required number of time steps, then maybe it isn't necessary to add new training points that are found within that set of input parameters.

Finally, as can be seen by the examples above that used the RBF-policy, the higher number of dimensions in the policy made it difficult for the optimiser to find the optimal policy. This is to be expected because of the higher computational costs. Furthermore, from a previous [blog post](https://gympy.github.io/jekyll/update/2016/11/24/First-RL.html)) it was found that a similar issue was encountered when the system itself was higher dimensional. Having being introduced to the wonderful world of latent variable models, that use techniques such as principal component analysis, factor analysis and the like to reduce the dimensionality of the system, I am quite intrigued by the possibility of the value of using such techniques to simplify the RL algorithm used in this project. By reducing the dimensionality of the system, the dimensionality of the policy could also be reduced, thus possibly leading to significant reductions in computational costs. Of course, the effects of such models on the accuracy and precision of the agent will need to be investigated and accounted for. 
