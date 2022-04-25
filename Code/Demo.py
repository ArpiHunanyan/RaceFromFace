
# from gc import callbacks
from tensorflow import keras
from  DataPreparation import getValidation, getTrain
from DenseNetImplementation import DenceNetClassifier
import json

# import resource
# class MemoryCallback(callbacks):
#     def on_epoch_end(self, epoch, log={}):
#         print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


print("Data Preperation")
train_data, train_labels = getTrain(30)
validation_data = getValidation(30)

model = DenceNetClassifier()
model.basicModel()
model.compile()
model.summary()
print()

# ## Train the top layer (for classifaction) 
print("Training the top layer (for classifaction) ")
trainingFirst = model.fit(x = train_data, y = train_labels,  epochs = 2, validation_data = validation_data, batch_size = 16, verbose=0, callbacks=[MemoryCallback()])

  
with open('trainingFirst_1', 'w') as convert_file:
     convert_file.write(json.dumps(trainingFirst.history))


print()
## Fine - Tuning
print("Fine - Tuning ")
model.setTrainable(True)
model.compile(optimizer = keras.optimizers.Adam(1e-7)) # 1e-5 => 1e-7

#model
trainingSecond = model.fit(x = train_data, y = train_labels,  epochs =  1, validation_data = validation_data)

with open('trainingSecond_1', 'w') as convert_file:
     convert_file.write(json.dumps(trainingSecond.history))

