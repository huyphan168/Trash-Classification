#%%
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from imutils import paths
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.layers import Input
from keras.layers import Activation
from keras.models import Model
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.applications import mobilenet
from keras import callbacks
from keras.constraints import max_norm
import matplotlib.pyplot as plt
import random
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical



image_path = list(paths.list_images('A:/K/GC/'))
random.shuffle(image_path) 

labels = [p.split(os.path.sep)[-2] for p in image_path]
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = to_categorical(labels)

list_image = []
for (j, imagePath) in enumerate(image_path):
    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image)
    
    image = np.expand_dims(image, 0)
    image = imagenet_utils.preprocess_input(image)
    
    list_image.append(image)
    
list_image = np.vstack(list_image)

img_width, img_height = 224, 224
shape = (img_width, img_height, 3)
Base_model=mobilenet.MobileNet(weights='imagenet', include_top=False, input_shape=shape)

x = Base_model.output
x = Flatten(name='flatten')(x)
x = Dense(512,activation='relu', kernel_constraint=max_norm(3))(x)
x = Dropout(0.5)(x)
x = Dense(128,activation='relu', kernel_constraint=max_norm(3))(x)
x = Dropout(0.3)(x)
x = Dense(64,activation='relu', kernel_constraint=max_norm(3))(x)
x = Dropout(0.2)(x)
x = Dense(4, activation='softmax')(x)   

model = Model(inputs=Base_model.input, outputs=x)

print(model.summary())
X_train, X_test, y_train, y_test = train_test_split(list_image, labels, test_size=0.2, random_state=42)

aug_train = ImageDataGenerator(rescale=1./255, rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, 
                         zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
aug_test= ImageDataGenerator(rescale=1./255)
#cell 1
#%%
for layer in Base_model.layers:
    layer.trainable = False


opt = RMSprop(0.001)
model.compile(opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])
numOfEpoch = 50
history = model.fit_generator(aug_train.flow(X_train, y_train, batch_size=32), 
                        steps_per_epoch=len(X_train)//32,
                        validation_data=(aug_test.flow(X_test, y_test, batch_size=32)),
                        validation_steps=len(X_test)//32,
                        epochs=numOfEpoch)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
#cell 2
#%%
for layer in Base_model.layers[11:]:
    layer.trainable = True

numOfEpoch = 60
opt = SGD(0.001)
model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])
history = model.fit_generator(aug_train.flow(X_train, y_train, batch_size=32), 
                        steps_per_epoch=len(X_train)//32,
                        validation_data=(aug_test.flow(X_test, y_test, batch_size=32)),
                        validation_steps=len(X_test)//32,
                        epochs=numOfEpoch)
#Lưu model trong filepath
model.save("filepath")
#Vẽ đồ thị acc và loss function
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# %%
