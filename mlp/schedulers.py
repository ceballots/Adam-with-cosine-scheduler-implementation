# -*- coding: utf-8 -*-
"""Training schedulers.

This module contains classes implementing schedulers which control the
evolution of learning rule hyperparameters (such as learning rate) over a
training run.
"""

import numpy as np
import math


class ConstantLearningRateScheduler(object):
    """Example of scheduler interface which sets a constant learning rate."""

    def __init__(self, learning_rate):
        """Construct a new constant learning rate scheduler object.

        Args:
            learning_rate: Learning rate to use in learning rule.
        """
        self.learning_rate = learning_rate

    def update_learning_rule(self, learning_rule, epoch_number):
        """Update the hyperparameters of the learning rule.

        Run at the beginning of each epoch.

        Args:
            learning_rule: Learning rule object being used in training run,
                any scheduled hyperparameters to be altered should be
                attributes of this object.
            epoch_number: Integer index of training epoch about to be run.
        """
        learning_rule.learning_rate = self.learning_rate

class CosineAnnealingWithWarmRestarts(object):
    """Cosine annealing scheduler, implemented as in https://arxiv.org/pdf/1608.03983.pdf"""

    def __init__(self, min_learning_rate, max_learning_rate, total_iters_per_period, max_learning_rate_discount_factor,
                 min_learning_rate_discount_factor = 1.,period_iteration_expansion_factor = 1):
        """
        Instantiates a new cosine annealing with warm restarts learning rate scheduler
        :param min_learning_rate: The minimum learning rate the scheduler can assign
        :param max_learning_rate: The maximum learning rate the scheduler can assign
        :param total_epochs_per_period: The number of epochs in a period
        :param max_learning_rate_discount_factor: The rate of discount for the maximum learning rate after each restart i.e. how many times smaller the max learning rate will be after a restart compared to the previous one
        :param period_iteration_expansion_factor: The rate of expansion of the period epochs. e.g. if it's set to 1 then all periods have the same number of epochs, if it's larger than 1 then each subsequent period will have more epochs and vice versa.
        """
        self.min_learning_rate = min_learning_rate
        self.max_learning_rate = max_learning_rate
        self.total_epochs_per_period = total_iters_per_period

        self.max_learning_rate_discount_factor = max_learning_rate_discount_factor
        self.period_iteration_expansion_factor = period_iteration_expansion_factor
        
        self.max_learning_rate_constant = max_learning_rate
        self.initial_period = total_iters_per_period
        
        self.till_epoch = total_iters_per_period
        self.last_epoch = 0
        
        self.frac = 0


    def update_learning_rule(self, learning_rule, epoch_number):
        """Update the hyperparameters of the learning rule.

        Run at the beginning of each epoch.

        Args:
            learning_rule: Learning rule object being used in training run,
                any scheduled hyperparameters to be altered should be
                attributes of this object.
            epoch_number: Integer index of training epoch about to be run.
        """
       #pivot = self.max_learning_rate
       #if epoch_number == 0:
       #    self.learning_rate = self.max_learning_rate    
       #elif epoch_number%self.total_epochs_per_period != 0:
       #    cycle = (1. + math.cos( (epoch_number%self.total_epochs_per_period) * math.pi / self.total_epochs_per_period)) / 2
       #    self.learning_rate = (cycle * (self.max_learning_rate - self.min_learning_rate)) + self.min_learning_rate                
       #    if epoch_number == 201:
       #        print(self.max_learning_rate)
       #else:
       #    self.max_learning_rate = self.max_learning_rate*0.9 
       #    pivot = pivot*0.9
       #    self.learning_rate = pivot        
        if epoch_number == 0:
            
            self.learning_rate = self.max_learning_rate 
            
        elif (epoch_number%
              self.till_epoch) != 0:
            
            self.frac += (1/self.total_epochs_per_period)*self.initial_period
            cycle = 0.5*(1. + 
                     math.cos(self.frac*
                              math.pi / self.initial_period))
            
            
            self.learning_rate = ((cycle*
                                  (self.max_learning_rate - self.min_learning_rate))+
                                  self.min_learning_rate) 

        else:   
            
            self.max_learning_rate = ((self.max_learning_rate_discount_factor**
                                      (epoch_number/self.total_epochs_per_period))*
                                      self.max_learning_rate_constant)
            
            
            self.learning_rate = self.max_learning_rate   
            self.total_epochs_per_period *= self.period_iteration_expansion_factor
            self.last_epoch = epoch_number
            self.till_epoch = self.total_epochs_per_period + self.last_epoch
            self.frac = 0
        

        return self.learning_rate

class CosineAnnealingNoWarmRestarts(object):
    """Cosine annealing scheduler, implemented as in https://arxiv.org/pdf/1608.03983.pdf"""

    def __init__(self, min_learning_rate, max_learning_rate, total_epochs):
        """
        Instantiates a new cosine annealing with warm restarts learning rate scheduler
        :param min_learning_rate: The minimum learning rate the scheduler can assign
        :param max_learning_rate: The maximum learning rate the scheduler can assign
        :param total_epochs: The total number of epochs
        """
        self.min_learning_rate = min_learning_rate
        self.max_learning_rate = max_learning_rate
        self.total_epochs = total_epochs


    def update_learning_rule(self, learning_rule, epoch_number):
        """Update the hyperparameters of the learning rule.

        Run at the beginning of each epoch.

        Args:
            learning_rule: Learning rule object being used in training run,
                any scheduled hyperparameters to be altered should be
                attributes of this object.
            epoch_number: Integer index of training epoch about to be run.
        """
    
        if epoch_number == 0:
            
            self.learning_rate = self.max_learning_rate 
            
        else:
            
            cycle = 0.5*(1. + 
                     math.cos(epoch_number*
                              math.pi / self.total_epochs))
            
            
            self.learning_rate = ((cycle*
                                  (self.max_learning_rate - self.min_learning_rate))+
                                  self.min_learning_rate) 
            
        return self.learning_rate

