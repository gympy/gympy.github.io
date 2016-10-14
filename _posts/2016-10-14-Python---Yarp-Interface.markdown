---
layout: post
title:  "Python-Yarp Interface"
date:   2016-10-14 10:17:05 +0100
categories: jekyll update
---
In order to get started with the project, it was necessary for me to get used to using YARP in conjunction with python. YARP, to me, sounds like a communication platform that allows for different modules of code, written in different languages to communicate with each other; it provides the tools to allow data to be transferred between these modules robustly.
The exercises that I carried out in order to master this are listed belows, with a few words about their relevance.

* The first exercise I completed was to create a simply python script that can take in messages from a */write* port, and pass these on to a */read* port. This exercise outlays the basic functionality of yarp, and how it can be utilised through python.

  The jupyter notebook that contains my code for this exercise can be found [here]({{site.url}}/downloads/basicPython.ipynb).

* Following this, similar functionality was expressed in a different way using a [RF module](http://www.yarp.it/classyarp_1_1os_1_1RFModule.html). This interesting module contains 2 loops that run simulataneously (or to be more technical, in parallel). One of them loops infinitely at intervals that are defined in the class, while the other looks for a command from a [RPC port](http://www.yarp.it/rpc_ports.html), and runs a *respond()* method when it receives one. *respond()* is then able to send a reply to the RPC port (it waits for one), and also carry out other functions that are only executed when a command is received.

  If that didn't make much sense, check out the code [here]({{site.url}}/downloads/pythonRFMod.ipynb). In this, I have implemented what was done in the previous exercise, but also made it such that messages are sent to */read* only when the command **print** is recieved from the *RPC Port*.

* Later on in my project, I will need to make use of information received from a camera in the form of a *yarp* image. As such, my final exercise for this week was to replace the messages replied in the form of text in the previous exercise with images from my webcam.

  As with the previous exercise, the program is to display the inputs it gets from the yarp port which sends out the data from the webcam (named */grabber*) on one output port (named */internalView* in my code). A RPC port will still send out commands; when it sends out the command **capture**, a still image of the most recent image sent by */grabber* will be shown on a separate port (named */pictureView* in my code).   

  In my first attempt, I cheated a little bit, and completed the task without using the processes that would have allowed me to learn what I will need from this exercise. The main point of this exercise was to look at how a yarp image could be imported as a numpy array. The numpy array could then be manipulated and used how I like; this, in other words, allows the images received from a camera to be used as an input source for the code I wrote for this exercise, and for the code I will write in the future. Furthermore, this introduced the code structures that allow for this kind of data transfer; hopefully this practice will be useful in the future when handling different kinds of data.

  In the code [here]({{site.url}}/downloads/imageRFMod.ipynb), to prove that I have indeed transferred the yarp image to a numpy array, I have added a block of black pixels on the top left corner to the output video and image.
