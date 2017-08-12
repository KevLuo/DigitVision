#!/usr/bin/env python

import sigmoid as sig

def sigmoidGradient(z):
    g = sig.sigmoid(z) * (1 - sig.sigmoid(z))
    return g
