#!/usr/bin/env python3
# -*- coding: utf-8 -*-



# Author: MenggeXue
# Last update: 2020.01.07
# First create: 2020.01.07
# Description:
# utils functions and classes

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count