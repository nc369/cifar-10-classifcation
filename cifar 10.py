from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import Sequential, models
from keras.layers import Dense, InputLayer, Conv2D as C2D, MaxPool2D as M2D,\
    Flatten, Dropout, BatchNormalization as Normalise
from keras.callbacks import ModelCheckpoint

inp_shp = (32,32,3)

Layers = [
    
    C2D (filters=32, kernel_size=3, padding='same', strides=1, activation='relu', input_shape = inp_shp),
    Dropout (0.1),
    Normalise(),
    C2D (filters=32, kernel_size=3, padding='same', strides=1, activation='relu'),
    #Dropout (0.1),
    #C2D (filters=32, kernel_size=2,padding='same', strides=1, activation='relu'),
     
    Normalise(),
    M2D(2,2),
    Normalise(),
    C2D (filters=32, kernel_size=3, padding='same', strides=1, activation='relu'),
    Dropout (0.1),
    Normalise(),
    C2D (filters=32, kernel_size=3, padding='same', strides=1, activation='relu'),
    
    Normalise(),
    M2D (2,2),
    Normalise(),
    C2D (filters=64, kernel_size=3, padding='same', strides=1, activation='relu'),
    Dropout (0.2),
    Normalise(),
    C2D (filters=64, kernel_size=3, padding='same', strides=1, activation='relu'),
    #Dropout (0.2),
    #C2D (filters=64, kernel_size=2,padding='same', strides=1, activation='relu'),

    Normalise(),
    M2D (2,2),
    Normalise(),
    C2D (filters=64, kernel_size=3, padding='same', strides=1, activation='relu'),
    Dropout (0.25),
    Normalise(),
    C2D (filters=64, kernel_size=3, padding='same', strides=1, activation='relu'),
    #Dropout (0.25),
    #C2D (filters=64, kernel_size=2,padding='same', strides=1, activation='relu'),
    
    Normalise(),
    M2D (2,2),
    Normalise(),
    C2D (filters=128, kernel_size=3, padding='same', strides=1, activation='relu'),
    Dropout (0.25),
    Normalise(),
    C2D (filters=128, kernel_size=3, padding='same', strides=1, activation='relu'),


    Flatten(),
    Normalise(),
    Dense(128, 'relu'),
    
    Dropout(.2),
    Normalise(),
    Dense(64, 'relu'),
    
    Dropout(.15),
    Normalise(),
    Dense(16, 'relu'),
    
    Normalise(),
    Dense(10, 'softmax', name='Predictor')

]


model = Sequential(Layers,"Cifar10")
model.compile( 'adam', 'sparse_categorical_crossentropy', metrics='accuracy'  )
model.summary()



datagen = ImageDataGenerator(
    rotation_range = 45,
    width_shift_range = 1.0,
    height_shift_range = 1.0,
    brightness_range = [.7,1.3],
    zoom_range = .3,
    horizontal_flip = True,
    vertical_flip = False,
    shear_range = .3,
    rescale=1/255,
    validation_split = .2
)

train_it = datagen.flow_from_directory( 
    directory = "D:/Downloads/IML/cnn project/data/train",
    target_size=inp_shp[:2],
    color_mode='rgb',
    class_mode='sparse',
    batch_size=40,
    subset='training',
    shuffle =True
    )

val_it = datagen.flow_from_directory( 
    directory = "D:/Downloads/IML/cnn project/data/train",
    target_size=inp_shp[:2],
    color_mode='rgb',
    class_mode='sparse',
    batch_size=40 ,
    subset='validation', 
    shuffle = True
    )

test_it = ImageDataGenerator(rescale = 1/255).flow_from_directory(
    directory = "D:/Downloads/IML/cnn project/data/test",
    target_size=inp_shp[:2],
    color_mode='rgb',
    class_mode='sparse',
    )


#path = where to store the model. define this before running the model

callback = ModelCheckpoint(
    filepath = path,
    monitor = 'val_accuracy',
    verbose = 1,
    save_best_only = True
    )

#model.load_weights(path)

model.fit(
    x = train_it,
    epochs = 50,
    verbose = 2,
    callbacks = [callback],
    validation_data = val_it,
    steps_per_epoch=750,
    validation_steps = 250
    )
          )


model.evaluate(test_it) # computes and return the accuracy and loss


