from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K 
import matplotlib.pyplot as plt

# specify the image height and width
img_width, imd_height = 50,50

train_data_dir = 'train'
validation_data_dir = 'val'
epochs = 10
batch_size = 16

#specify the input shape
if K.image_data_format == 'channels_first':
    input_shape = (1,img_width,imd_height)
else:
    input_shape = (img_width,imd_height,1)

#init a sequential model
model = Sequential()

#Layer1
model.add(Conv2D(32,(2,2)), input_shape = input_shape, activation = 'relu')
model.add(MaxPooling2D(pool_size = (2,2)))

#Layer2
model.add(Conv2D(32,(2,2)), input_shape = input_shape, activation = 'relu')
model.add(MaxPooling2D(pool_size = (2,2)))

#layer3
model.add(Conv2D(64,(2,2)), input_shape = input_shape, activation = 'relu')
model.add(MaxPooling2D(pool_size = (2,2)))

#two fully connected layers
model.add(Flatten())
model.add(Dense(64,activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(4,activation = 'sigmoid'))


model.compile(losss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])


#generate data using the flow_from_directory
train_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size = (img_width,imd_height),
                                                    batch_size = batch_size,
                                                    color_mode = "grayscale",
                                                    shuffle = True,
                                                    class_mode = "categorical")
valid_generator = train_datagen.flow_from_directory(validation_data_dir,
                                                    target_size = (img_width,imd_height),
                                                    batch_size = batch_size,
                                                    color_mode = "grayscale",
                                                    shuffle = True,
                                                    class_mode = "categorical")

#fit the weights
train_samples = train_generator.n
validation_samples = valid_generator.n
history = model.fit_generator(train_generator,
                              steps_per_epoch = train_samples // batch_size, 
                              epochs = epochs, validation_data = valid_generator, 
                              validation_steps = validation_samples // batch_size)
print(history.history.keys())

# Plots for the Loss function
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#save the weights for future
model.save_weights('hand_gestures.h5')

