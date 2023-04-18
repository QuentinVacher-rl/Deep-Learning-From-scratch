"""Program file
"""
import pandas as pd
import numpy as np
import random


from model import Model

import layers

import time

def main():
    print("hello world")

    random.seed(0)
    np.random.seed(0)

    model = Model()
    model.add(layers.Conv2D(32, 3, "ReLu", input_shape=[28,28]))
    model.add(layers.MaxPooling())
    model.add(layers.Dropout(coef=0.1))
    model.add(layers.Conv2D(32, 3, "ReLu"))
    model.add(layers.MaxPooling())
    model.add(layers.Dropout(coef=0.1))
    model.add(layers.Conv2D(32, 3, "ReLu"))
    model.add(layers.Flatten())
    model.add(layers.Dropout(coef=0.2))
    model.add(layers.Dense([300], "ReLu"))
    model.add(layers.Dropout(coef=0.2))
    model.add(layers.Dense([100], "ReLu"))
    model.add(layers.Dropout(coef=0.2))
    model.add(layers.Dense([10], "Sigmoid")) 

    
    size = 42000

    df = pd.read_csv('digit-recognizer\\train.csv')
    label = np.array(df["label"])
    values = np.array(df.drop("label", axis=1)).reshape(-1, 1, 28, 28) / 255
    #values = np.array(df.drop("label", axis=1)) / 255

    rapport_train = 0.8
    size_train = int(size * rapport_train)

    label_train, label_test = np.split(label, [size_train])
    values_train, values_test = np.split(values, [size_train])

    model.compile(optimizer="adam")

    t1 = time.time()

    model.fit(values_train, label_train, nb_epoch=150, size_batch=1, validation_data=(values_test, label_test))
    print(time.time() - t1)

    accuracy, loss = model.test(values_test, label_test)    
    print("Accuracy is {}%".format((accuracy)))
    
        

    

if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    main()
