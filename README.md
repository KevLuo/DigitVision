# DigitVision

This project applies computer vision principles to automatically recognize handwritten digits.  DigitVision has several defining characteristics:

1.  A neural network which I built from scratch
2.  PCA from scikit-learn (Principle Component Analysis)
3.  Diagnostic visualizations, including learning curves and validation curves

I originally started this to better understand neural networks, by writing algorithms like backpropogation myself and building computations like the cost function with my own code, so I could figure out how all of this actually worked.  But ever since my initial architecture surprisingly scored at 93% on Kaggle (I was just hoping that it wouldn't blow up), I've been slightly obsessing over how I could take my homemade model to the next level.  

I'm still toying with all of this, but I'm happy to say that my accuracy has risen to 96.6%, through the addition of methods like scikit-learn's PCA and the construction of illuminating diagnostics like learning curves.
