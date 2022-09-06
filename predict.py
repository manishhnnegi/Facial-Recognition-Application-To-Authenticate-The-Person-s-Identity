import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt
# Import tensorflow dependencies - Functional API
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf
# Import metric calculations
from tensorflow.keras.metrics import Precision, Recall







class dogcat:
    def __init__(self,filename):
        self.filename =filename


    def predictiondogcat(self):
        # load model
        #model = load_model('model.h5')
        # Reload model 
       
        # summarize model
        #model.summary()
        def preprocess(file_path):
    
            # Read in image from file path
            byte_img = tf.io.read_file(file_path)
            # Load in the image 
            img = tf.io.decode_jpeg(byte_img)
            
            # Preprocessing steps - resizing the image to be 100x100x3
            img = tf.image.resize(img, (100,100))
            # Scale image to be between 0 and 1 
            img = img / 255.0
            
            # Return image
            return img

        class L1Dist(Layer):
    
            # Init method - inheritance
            def __init__(self, **kwargs):
                super().__init__()
            
            # Magic happens here - similarity calculation
            def call(self, input_embedding, validation_embedding):
                return tf.math.abs(input_embedding - validation_embedding)

        l1 =L1Dist()
        
        model = tf.keras.models.load_model('siamesemodellatest.h5',custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy})

        def verify(model, detection_threshold, verification_threshold):
            # Build results array
            results = []
            for image in os.listdir(os.path.join('application_data', 'verification_images')):
                input_img = preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
                validation_img = preprocess(os.path.join('application_data', 'verification_images', image))
                
                # Make Predictions 
                result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
                results.append(result)
            
            # Detection Threshold: Metric above which a prediciton is considered positive 
            detection = np.sum(np.array(results) > detection_threshold)
            
            # Verification Threshold: Proportion of positive predictions / total positive samples 
            verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images'))) 
            verified = verification > verification_threshold
            
            return results, verified

        results, verified = verify(model, 0.1, 0.1)
        print(verified)


            #imagename = self.filename
            #test_image = image.load_img(imagename, target_size = (64, 64))
            #test_image = image.img_to_array(test_image)
            #test_image = np.expand_dims(test_image, axis = 0)
            #result = model.predict(test_image)

       
        if verified:
            prediction = 'Image verified as MANISH!!!!'
            return [{ "image" : prediction}]
        else:
            prediction = 'Not varified'
            return [{ "image" : prediction}]
     