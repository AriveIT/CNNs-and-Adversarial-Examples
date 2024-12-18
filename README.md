# Abstract

Convolutional neural networks (CNNs) have proven themselves to be capable of classifying images with high accuracy. However,
they are vulnerable to adversarial examples, which are inputs crafted
specifically to cause the model to misclassify them. This project explores
training CNNs with various techniques to become resistant to these adversarial examples. The examples are generated with Fast Gradient Sign
Method (FGSM), and the techniques used to create more robust models
are Adversarial Training, and Defensive Distillation. The Defensive Distillation method is analyzed, and a variation is used. An initial baseline
CNN evaluated on a dataset generated with FGSM scores an accuracy of
20.4%. The final model, which is trained with Adversarial Training and
Defensive Distillation variation, adds some noise to inputs as a preprocessing step and scores an accuracy of 59.8%.

Notebooks were often used as scratch. Paper may include more readable results and insights than notebooks. 
This project was completed for SENG 474 at UVic

# Files
ImageClassifier.ipynb: Baseline CNN, and some initial visualizations of CIFAR-10  
FGSM.ipynb: Generating adversarial examples with Fast Gradient Sign Method, and applying it to the baseline model  
AdversarialTraining.ipynb: Using 2 variations of adversarial training to create new, more robust models. Visualizations with confusion matrices  
DefensiveDistillation.ipynb: Implemented defensive distillation, and a variation as described in the included paper.  
Visualizations.ipynb: Generated visualizations for use in the included paper  
AdversarialExamples.py: Aggregation of methods from above notebooks to simplify subsequent notebooks.

