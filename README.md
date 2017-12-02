# DigitVision

This project applies computer vision principles to automatically recognize handwritten digits.  DigitVision has several defining characteristics:

1.  A fully-connected neural network which I built from scratch
2.  PCA from scikit-learn (Principle Component Analysis)
3.  Diagnostic visualizations, including learning curves and validation curves

The main goal of this expedition was not to gain the highest accuracy possible, but to understand the fundamentals of neural nets. By writing algorithms like backpropogation myself and building computations like the cost function with my own code, I hoped to figure out how all of this actually worked.  But ever since my initial architecture surprisingly scored at 93% on Kaggle (I was just hoping that it wouldn't blow up), I've began exploring model tuning to see how far I can elevate a homemade model!

I'm still toying with all of this, but I'm happy to say that my accuracy has risen to 96.6%, through the addition of methods like scikit-learn's PCA and the construction of illuminating diagnostics like learning curves.

Code Organization:  The main script which controls the flow of the program is vision_main.py.  From there, it calls the necessary methods that are segmented out into different files for structure and readibility.
