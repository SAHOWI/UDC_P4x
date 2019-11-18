
#!/usr/bin/python3

import csv
import cv2
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D
from keras.layers import Lambda, Cropping2D
from keras.layers.pooling import MaxPooling2D

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split





### ensure that the workspace will not stop
from workspace_utils import active_session
 
with active_session():
    # do long-running work here
    



    debug=0
    OS='unix'
    # OS='win'


    ### local laptop implementation
    # data_dir='C:\\temp\\data\\data\\'
    ### UDACITY WorkSpace implementation 
    data_dir='/home/workspace/CarND-Behavioral-Cloning-P3/data/'





    def print_sep():
        print("----------------------------------------")
    # end of def: print_set





    ### read the data first
    lines = []
    print_sep()
    print("Begin reading data")
    print_sep()

    with open(data_dir + 'driving_log.csv') as csvfile:
        ### to skip he header line of the CSV file
        reader = csv.reader(csvfile)
        next(reader)
        for line in reader:
            lines.append(line)

    print("Logdata read") 
    print("Lines read =", len(lines))

    


    #simply splitting the dataset to train and validation set usking sklearn. 
    # 0.15 is 15% of the dataset is validation set
    train_samples, validation_samples = train_test_split(lines,test_size=0.15)


    def generator(samples, batch_size=32):
        num_samples = len(samples)
        while 1: 
            shuffle(samples) #shuffling the total images
            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset+batch_size]


                images = []
                measurements = []

                for batch_sample in batch_samples:
                    correction = 0.2

                    for i in (0,2):
                        ### first fix path in log_file
                        ### and remove leading and trailing white spaces (they are existing in the provided data.zip!!!)
                        source_path = batch_sample[i].strip()
                        if (debug == 1):
                            print("file #", i, "= ", source_path)
                        if (OS == 'win'):
                            tokens = source_path.split('\\')
                        else:
                            tokens = source_path.split('/')
                            
                        filename = tokens[-1]
                        if (OS == 'win'):
                            local_path = data_dir + filename
                        else:
                            local_path = data_dir + 'IMG/' + filename
                        if (debug == 1):
                            print("Local Path = ", local_path)
                        
                        image = cv2.cvtColor(cv2.imread(local_path), cv2.COLOR_BGR2RGB)
                        if (debug == 1):
                            print(image)
                        images.append(image)
                        measurement=float(line[3])
                        if (i == 0):
                            measurements.append(measurement)
                        elif (i == 1):
                            steering_left = measurement + correction
                            measurements.append(steering_left)
                        elif ( i == 2):
                            steering_right = measurement - correction
                            measurements.append(steering_right)



                        
                        
                        #### Data Augmentation
                        
                        augmented_images = []
                        augmented_measurements = []
                        for image, measurement in zip(images, measurements):
                            augmented_images.append(image)
                            augmented_measurements.append(measurement)
                            flipped_image=cv2.flip(image,1)
                            flipped_measurement = measurement * -1.0
                            augmented_images.append(flipped_image)
                            augmented_measurements.append(flipped_measurement)

                        ### end of data augmentation

                        print("Data configured")
                        print_sep()
                        print("Images                 =", len(images))
                        print("Measurements           =", len(measurements))
                        print("Augmented Images       =", len(augmented_images))
                        print("Augmented Measurements =", len(augmented_measurements))
                        print_sep()


                X_train = np.array(augmented_images)
                y_train = np.array(augmented_measurements)
                yield sklearn.utils.shuffle(X_train, y_train)



    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)

    ### define our Model
    print_sep()
    print("Building the Model")
    print_sep()

    model = Sequential()
    # Preprocess incoming data, centered around zero with small standard deviation 
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    #model.add(Cropping2D(cropping=((75,25),(0,0))))
    #model.add(Conv2D(6,5,5, activation="relu"))
    #model.add(MaxPooling2D())
    #model.add(Conv2D(16,5,5, activation="relu"))
    #model.add(MaxPooling2D())
    #model.add(Flatten())
    #model.add(Dense(120))
    #model.add(Dense(84))
    #model.add(Dense(1))

    # trim image to only see section with road
    model.add(Cropping2D(cropping=((70,25),(0,0))))           

    #layer 1- Convolution, no of filters- 24, filter size= 5x5, stride= 2x2
    model.add(Conv2D(24,(5,5),strides=(2,2), activation='relu'))
    

    #layer 2- Convolution, no of filters- 36, filter size= 5x5, stride= 2x2
    model.add(Conv2D(36,(5,5),strides=(2,2), activation='relu'))
    
    #layer 3- Convolution, no of filters- 48, filter size= 5x5, stride= 2x2
    model.add(Conv2D(48,(5,5),strides=(2,2), activation='relu'))
    
    #layer 4- Convolution, no of filters- 64, filter size= 3x3, stride= 1x1
    model.add(Conv2D(64,(3,3), activation='relu'))
    

    #layer 5- Convolution, no of filters- 64, filter size= 3x3, stride= 1x1
    model.add(Conv2D(64,(3,3), activation='relu'))
    

    #flatten image from 2D to side by side
    model.add(Flatten())

    #layer 6- fully connected layer 1
    model.add(Dense(100))
    
    #Adding a dropout layer to avoid overfitting. Here we are have given the dropout rate as 25% after first fully connected layer
    model.add(Dropout(0.25))

    #layer 7- fully connected layer 1
    model.add(Dense(50))
    
    #layer 8- fully connected layer 1
    model.add(Dense(10))
    
    #layer 9- fully connected layer 1
    model.add(Dense(1)) #here the final layer will contain one value as this is a regression problem and not classification




    print_sep()
    print("Done building the Model")
    print_sep()

    ### compile the model
    print_sep()
    print("Compiling the Model")
    print_sep()
    model.compile(optimizer='adam', loss='mse')

    print_sep()
    print("Done compiling the Model")
    print_sep()

    print_sep()
    print("Training the Model")
    print_sep()

    #### this must be changed to model.fit_generator(...)
    #### which requires impementation of a GENERATOR :-)
    # model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)
    model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator,   nb_val_samples=len(validation_samples), nb_epoch=5, verbose=1)
    print_sep()
    print("Done training the Model")
    print_sep()


    model.save('model.h5')


