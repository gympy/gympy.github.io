---
layout: page
title: About
permalink: /about/
---
Dear Reader,

Thanks for finding this blog. Here is a brief introduction about the project that it talks about.

The main aim of my project is to develop an algorithm that a robotic entity can carry out to learn the dynamic model of its parts, be it for actions such as reaching, grasping or walking. The motivation for this idea stems from the difficulty of carrying out usual control strategies over complex dynamic models. As Prof Russ Tedrake says in his elegant [introduction](https://www.youtube.com/watch?v=2inIBRmDXWk), our controllers tend to overrule the natural dynamics of the system, say through high gain motors in the joints of a walking robot, and put in the response we need. As he further explains, nature makes use of these natural dynamics to gain great efficiencies in the physical processes it carries out.

This disadvantage often arises due our inability to either fully represent the system in our dynamic model, or because our control strategies are lacking in their ability to efficiently and effectively manuipulate complex models. My goal, therefore, for this project is to use statistical methods employed over a latent space of variables we do not know about to allow robots to learn more about their own dynamics than we can tell them. In turn, I hope that such a black box method can then lead to greater efficiencies in robotic movement.

The work that I will be carrying out will follow from and build on the influential paper Marc Deisenroth et al published on the use of Gaussian Processes for such purposes. This paper can be found [here](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6654139), and follw the link [here](https://www.youtube.com/watch?v=f7y60SEZfXc) to watch Marc Deisenroth give a talk about it.

In order to test my method, I will be employing it to the arm of a iCub humanoid robot. Before that however, I will need to test my algorithm out in other, less complex systems. As such, I will be using the OpenAI gym environment to try the algorithm out to control systems such as the inverted pendulum, the double pendulum and the cart-pole. This blog will detail the exercises I carried out to learn the necessary programming modules to complete this first task. I will also, from time to time, submit any interesting thoughts I have about my work.

I hope you enjoy.

Janith
