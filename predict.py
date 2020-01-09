from model import img_width,imd_height,batch_size,train_generator,ImageDataGenerator
import numpy as np 
import pandas as pd
from keras.model import load_model

test_data_dir  = 'test'
test_datagen = ImageDataGenerator(rescale = 1./255)

#make the test data generator
test_generator = test_datagen.flow_from_directory(
                 directory = test_data_dir,
                 target_size=(img_width,imd_height),
                 batch_size = batch_size,
                 color_mode = "grayscale",
                 shuffle = False,
                 class_mode = "None")
test_samples = test_generator.n

#make the predictions using the model
model = load_model("hand_gestures.h5")
test_generator.reset()
pred=model.predict_generator(test_generator,
                             steps=test_samples//batch_size,
                             verbose=1)
predicted_class_indices=np.argmax(pred,axis=1)

#labeling the predicted output
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

#storing the result in csv file
filenames=test_generator.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
results.to_csv("results.csv",index=False)