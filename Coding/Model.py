# Author: ArpiHunanyan
# Created: 29 April,2022, 14:58 PM
# Email: arpi_hunanyan@edu.aua.am




from tensorflow.keras import applications
from tensorflow.keras import Input, Model 
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import SpecificityAtSensitivity
import inspect





class Classifier:



    def __init__(self, input_shape = (224, 224, 3), classes = 7, modelName = "DenseNet121"):

        '''Constructs a new DenceNetClassifier.
           Besides the top layer, all other layers are not trainable. 

      
            # Arguments 
               input_shape: optional shape tuple, defold (224, 224, 3)
               classes: optional number of classes to classify images into, defold 7

        '''

   

        self.input_shape = input_shape
        self.classes = classes
        self.modelName = modelName
    
        # Instantiate a base model and load pre-trained weights into it.
        self.base_model = eval("applications." + modelName)(include_top = False,
                                      weights = 'imagenet', # pre-training on ImageNet
                                      input_shape = self.input_shape)

        # Freeze all layers in the base model 
        self.base_model.trainable = False

        # Create a new model on top of the output of one (or several) layers from the base model.
        input_tensor = Input(shape = self.input_shape, name = "InputFaceImg")
        base_output_tensor = self.base_model(input_tensor, training = True)
        pooled_output_tensor = GlobalAveragePooling2D(name = "PooledOutput")(base_output_tensor)
        output_tensor = Dense(self.classes,  activation = "softmax", name = "OutputClass")(pooled_output_tensor)
        self.model = Model(input_tensor, output_tensor, name = modelName)



    def compile(self, optimizer = Adam(), metrics = ['accuracy',  'Recall', SpecificityAtSensitivity(sensitivity = 0.5)] ):
        self.model.compile(optimizer = optimizer,
                          loss = 'categorical_crossentropy',
                          metrics = metrics
                          )


    def fit(self, x = None, y = None, batch_size = 16, epochs = 1, validation_data = None, callbacks = None):

        '''Trains the model for a fixed number of epochs (iterations on a dataset).
           # Arguments
            x: Input data
            y: Target data
            batch_size: Integer or None.
            epochs: Integer
            validation_data: Data on which to evaluate the loss and any model metrics at the end of each epoch.
        
          # Returns
            A History object. Its History.history attribute is a record of training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values (if applicable).
        '''
        training = self.model.fit( x = x, 
                                   y = y, 
                                   batch_size = batch_size, 
                                   epochs = epochs, 
                                   validation_data = validation_data ,
                                   callbacks = callbacks
                                )
        return training
    def baseModelParamsCount(self):
        return self.base_model.count_params()

    def setTrainable (self, trainable = False):
        self.base_model.trainable = trainable
    


    def unfreazeOddLayers(self): 
        index = 0   
        for layer in self.base_model.layers:
            if index % 2 == 0:
                layer.trainable = True
            index = index + 1 

    def unfreazeLastLayers(self, num = 10): 
        
        '''
        Lower layers refer to general features (problem independent)
        higher layers refer to specific features (problem dependent)

        '''
        for layer in self.base_model.layers[len(self.base_model.layers) - num : ]:
            layer.trainable = True



    def summary(self):
        return self.model.summary(expand_nested = True,
                                 show_trainable = True)


    def save(self, path = "Model"):
        self.model.save(path)
        print("Model is succesfuly saved in ", path, ".")



def kerasModelNames():
        return  [m[0] for m in inspect.getmembers(applications, inspect.isfunction)]




    


