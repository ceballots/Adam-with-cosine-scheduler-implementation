# Adam-with-cosine-scheduler-implementation
This report is motivated by different discussions on https://arxiv.org/pdf/1711.05101.pdf. In here, RMSProp and Adam are compared to Stochastics Gradient Descent (SGD) in terms of convergence and performance. Then,  one way to improve learning rules is by using schedulers. This leads us to use the cosine annealing scheduler with and without warm restarts. We compare them with constant learning rates for both SGD and Adam. Finally, the differences raised by https://arxiv.org/pdf/1711.05101.pdf between L2 and weight decay in adaptive learning rules are explored by using constant and schedule learning rates.  
All the code is implemenet from scratch in Python. All the experiments were carried out using the EMNIST (Extended MNIST) Balanced data set.  
