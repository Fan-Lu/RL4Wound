# DARPPA RL

## Question:
### 1. Why SVM? 
To predict the wound day?
Then why not the 5 state into RL model? 
At each state, we can use SVM to predict the wound day,
and give the reward.

### 2. Problem: No summation in the reward, since the $Q$ function already contains such summation.

### 3. In Figure 2.A, is there any threshold that determines whether wound healing process ends?

### 4. SVM model is wrong. We cannot train SVM or other regression model based on non-treatment data and then predict wound stage using treatment data.


## Code Log

## 07/21/2023
### Done:
1. Change number of actions from 2 to 10. Now action can take values from [0, 0.1, ..., 0.9, 1.0].
2. Add configuration files. All hyperparameters go to ./cfgs/configs.py
3. Pay attention to the check_opt in the config.py. This should be set to False when training and testing RL models.

## 07/25/2023
### Done
1. Add PPO, A2C, DQN

## 08/10/2023
1. PPO is good for continuous control: the simple model.
2. change ratio clip from 0.1 to 0.2, waiting for results
3. TODO: it seems reducing k_epochs can improve performances?

## 08/14/2023
1. Fixed random seed for generating trajectories with k = [0.5, 0., 0.1]

## 09/05/2023
1. WoundIonEnv contains two types of action: position of ion pump, and amount of actuation

## 09/07/2023
1. Need to ask what is the maximum resistence. Is 3.3 voltage is enough? Saturation!!!


## Notes on HealNet
1. Gaussian Filtering

    The Gaussian Smoothing Operator performs a weighted average of surrounding pixels based on the Gaussian 
    distribution. It is used to remove Gaussian noise and is a realistic model of defocused lens. 
    Sigma defines the amount of blurring. The radius slider is used to control how large the template is. 
    Large values for sigma will only give large blurring for larger template sizes. Noise can be added using 
    the sliders.
    https://www.southampton.ac.uk/~msn/book/new_demo/gaussian/
2. Potential Improvements:
   1. How to normalize images? Currently, in the code, pixel values in each channel are normalized
   by subtracted from the average values and added by some deviation avg_dv. This avg_dv is a 
   hyper-parameter need to be fine tuned.
   2. The whole image is then cropped to have only the center of the wound by selecting pixels 
   in the range [[1000, 4000], [1500, 5500]]. In the future, we may consider some autodectection 
   method.
   3. We blur the image through a Gaussian filter.
   4. Patched are created randomly, i.e., for a image with size wxd, patch positions are created
   by randomly select values fron [0, w+crop_size].
   (This is done in the testing code. Note sure what Hector did during training phase.)
   5. During testing, patches of wound image are not normalized!
   6. max 0.45 kg / wound / day (max is 15 kg) high dose in 0.025 for experiment 16 has better re-epi result.




### TODOs:
1. Maybe actor-critic? since we would like to learn an optimal actuator function, which is supposed to be smooth.
2. Read Ksenia's matlab code. Considering in the future change ODE list to matrices.
3. Add RNN/LSTM
4. Use A3C.
5. Attention is all your need. (self attention)
6. Check the origional paper of TD3: do we need to add noise to the Q_next?
