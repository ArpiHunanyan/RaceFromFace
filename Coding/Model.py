# Author: ArpiHunanyan
# Created: 16 May, 2022
# Email: arpi_hunanyan@edu.aua.am


from calendar import EPOCH
from tkinter import N
from tensorflow.keras import applications
from tensorflow.keras import Input, Model 
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import SpecificityAtSensitivity
from tensorflow.keras.models import load_model
import inspect




class Classifier:



    def __init__(self, input_shape = (224, 224, 3), classes = 7, modelName = None, createModel = True, path = None):
        """Inatializ the CNN model with pre-trained Keras application"""
        
        self.input_shape = input_shape
        self.classes = classes
        self.modelName = modelName

        if (createModel):
            # Instantiate a base model and load pre-trained weights into it.
            self.base_model = eval("applications." + self.modelName)(include_top = False,
                                                                weights = 'imagenet', # pre-training on ImageNet
                                                                input_shape = self.input_shape                                     
                                                                )

            # Freeze all layers in the base model 
            self.base_model.trainable = False

            # Create a new model on top of the output of one (or several) layers from the base model.
            input_tensor = Input(shape = self.input_shape, name = "input_face")
            base_output_tensor = self.base_model(input_tensor, training = False ) #"pre_traind_model"
            pooled_output_tensor = GlobalAveragePooling2D(name = "pooled_output")(base_output_tensor)
            output_tensor = Dense(self.classes,  activation = "softmax", name = "output_class")(pooled_output_tensor)
            self.model = Model(input_tensor, output_tensor, name = self.modelName )
    


        else :
            # Load model in the path
            self.model = load_model(path)
            self.base_model = self.model.layers[1]
            self.base_model.trainable = False


        




    def compile(self, optimizer = Adam(), metrics = ['accuracy',  'Recall', 'Precision',  SpecificityAtSensitivity(sensitivity = 0.5)] ):

        """ Compiles the model with the specific argumantes."""

        self.model.compile(optimizer = optimizer,
                          loss = 'categorical_crossentropy',
                          metrics = metrics
                          )


    def fit(self, x = None, y = None, batch_size = 16, epochs = 1, validation_data = None, callbacks = None):

        """ Train the model with spesifice arguments. Returns the genaratied metrics and loss for each epoch."""

        training = self.model.fit( x = x, 
                                   y = y, 
                                   batch_size = batch_size, 
                                   epochs = epochs, 
                                   validation_data = validation_data ,
                                   callbacks = callbacks
                                )

        return training

    def baseModelParamsCount(self):

        """Returns count of parametrs in pre-traind model."""

        return self.base_model.count_params()

    def baseModelLayersCount(self):

        """Returns count of layers in pre-traind model."""

        return len(self.base_model.layers)


    def setTrainable (self, trainable = False):

        """Sets pre-traind models' layers in training mode if trainable is True. Otherwise infrance mode."""

        self.base_model.trainable = trainable
    

    def unfreazeLastLayers(self, num = 10): 

        """Sets pre-traind models' last layers in training mode with corsponding number.""" 

       # Lower layers refer to general features (problem independent)
       # higher layers refer to specific features (problem dependent)
        if (num == 0):
            return
            
        for layer in self.base_model.layers[len(self.base_model.layers) - num : ]:
            layer.trainable = True



    def summary(self):

        """Returns the summary of the structure of the model.""" 

        return self.model.summary(expand_nested = True,
                                 show_trainable = True)


    def save(self, path = None):

        """ Save the model. """ 

        self.model.save(path)
        print("Model is succesfuly saved in ", path, ".")

    def evaluate(self, data):

        """ Evaluate the data. """

        return self.model.evaluate(data, batch_size = 16, verbose = 1, return_dict = True)
     


def kerasModelNames():

    """ Returns the Names of  all Keras aplication names."""

    return  [m[0] for m in inspect.getmembers(applications, inspect.isfunction)]

