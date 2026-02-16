from models.model import Model
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.optimizers import RMSprop, Adam

class BasicModel(Model):
    def _define_model(self, input_shape, categories_count):
        # Your code goes here
        # you have to initialize self.model to a keras model

        # Initialize model using the Sequential API
        model = Sequential()
        model.add(Rescaling(1./255, input_shape=input_shape))

        # First convolutional layer - finds basic patterns/edges
        model.add(layers.Conv2D(4, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2))) # MaxPooling downsamples by taking max values

        # Second convolutional layer - finds complex patterns
        model.add(layers.Conv2D(8, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        # Third convolutional layer - finds combinations of the complex patterns
        model.add(layers.Conv2D(8, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        # Flatten the maps - dimension converter / reorganizer
        model.add(layers.Flatten())
        
        # Connection layer with dropout
        model.add(layers.Dense(32, activation='relu'))
        
        # Softmax layer - converts raw scores into class probabilities between 0 and 1
        model.add(layers.Dense(categories_count, activation='softmax'))

        self.model = model

    def _compile_model(self):
        # Your code goes here
        # you have to compile the keras model, similar to the example in the writeup
        self.model.compile(
            optimizer=RMSprop(learning_rate=0.001), # Keras documentation (Adam, RMSprop, or SGD with learning rate 0.01 and momentum 0.0)
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )