import os
import cv2
import numpy as numpy
import matplotlib.pyplot as pyplot
import tensorflow as tensorflow


#Load dataset
mnist = tensorflow.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Reduce scale to [0,1]
x_train = tensorflow.keras.utils.normalize(x_train, axis=1)
x_test = tensorflow.keras.utils.normalize(x_test, axis=1)

'''
#THIS TRAINS THE MODEL
model = tensorflow.keras.models.Sequential() #Unsure

#Transforms 28x28 grid, to a one dimensional vector of  784 neuron, FIRST LAYER
model.add(tensorflow.keras.layers.Flatten(input_shape=(28,28)))

#Basic neural network layer, can change activation type by tensorflow.nn.**
model.add(tensorflow.keras.layers.Dense(128, activation='relu'))

model.add(tensorflow.keras.layers.Dense(32, activation='relu'))

#10 neurons for answer result, softmax defines that the answer neurons total to 1
model.add(tensorflow.keras.layers.Dense(10, activation='softmax'))

#Compile model??
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#Epochs says how many times the model sees the training data
model.fit(x_train, y_train, epochs=100)

model.save('handwritten.model')
'''

''' 
#THIS SHOWS LOSS VS ACCURACY
model = tensorflow.keras.models.load_model('handwritten.model')

print(x_test.shape)
print(y_test.shape)


loss, accuracy = model.evaluate(x_test, y_test)

print(loss)
print(accuracy)
'''


#THIS ACTUALLY TESTS IT VS NEW IMAGES
model = tensorflow.keras.models.load_model('handwritten.model')

image_number = 1
while os.path.isfile(f"Original_SampleData/{image_number}_1.png"):
    try:
        img = cv2.imread(f"Original_SampleData/{image_number}_1.png")[:,:,0]
        img = numpy.invert(numpy.array([img]))
        prediction = model.predict(img)
        print(f"This digit is probably a {numpy.argmax(prediction)}")
        pyplot.imshow(img[0], cmap=pyplot.cm.binary)
        pyplot.show()
    except:
        print("Error!")
    finally:
        image_number += 1

