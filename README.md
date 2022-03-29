# Intelligent Strong Motion Prediction

This repository is the official PyTorch  implementation of [Neural Network-Based Strong Motion Prediction for On-Site Earthquake Early Warning](https://www.mdpi.com/1424-8220/22/3/704/pdf)

ISMP (Intelligent Strong Motion Prediction) uses a convolutional neural network to effectively extract relevant features from the initial P-waves to predict whether the peak ground acceleration of subsequent waves surpasses 80 Gal.

# Dataset
* Can be download directly from "dataset" folder of this repository.
* Please place the dataset under the "data" folder which in the same directory of "run_train.py" and "run_test.py"

# Training
```
python run_train.py 
```

# Testing
```
python run_test.py 
```
