#!/usr/bin/env python

import sigmoid as sig

def sigmoidGradient(z):
    g = sig.sigmoid(z) * (1.0 - sig.sigmoid(z))
    return g
