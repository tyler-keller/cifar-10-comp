# COMPETITION

#########################################
# DO NOT MODIFY THIS SECTION

# load the competition data
# The data is in the numpy array format:
#   competition_images: (100,32,32,3) contains 100 images
#   competition_labels: (100,1) contains class lables (0 to 9)
import numpy as np
competition_data = np.load('competition_data.npz') 
competition_images = competition_data['competition_images']
competition_labels = competition_data['competition_labels']


#########################################
# YOUR CODE/MODEL GOES HERE:

# load your model and/or trained weights
import tensorflow as tf
from tensorflow import keras
from keras import models

my_model = models.load_model('simple_cnn.keras')

# evaluate your model on the competition data
# make any adjustment to the data format as needed to run your model
# you must return accuracy of your model on the competition data 
competition_loss, competion_acc = my_model.evaluate(competition_images,  competition_labels)

# MUST PRINT OUT THE ACCURACY OF YOUR MODEL ON THE COMPETITION DATA
print('Accuracy:', competion_acc) 

