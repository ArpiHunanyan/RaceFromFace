
# from gc import callbacks
from tensorflow import keras
from  DataPreparation import getValidation, getTrain
from DenseNetImplementation import DenceNetClassifier
import json

print("Data Preperation")
tarin_data = getTrain( M = 30, tensor = True)
validation_data  = getValidation( tensor = True)
# train_data, train_labels = getTrain( tensor = True)
# validation_data = getValidation( tensor = True)

model = DenceNetClassifier()
model.basicModel()
model.compile()
# model.summary()
print()

# ## Train the top layer (for classifaction) 
print("Training the top layer (for classifaction) ")
trainingFirst = model.fit(tarin_data,   epochs = 2, validation_data = validation_data, batch_size = 16)

  
with open('trainingFirst_1', 'w') as convert_file:
     convert_file.write(json.dumps(trainingFirst.history))


print()
## Fine - Tuning
print("Fine - Tuning ")
model.setTrainable(True)
model.compile(optimizer = keras.optimizers.Adam(1e-7)) # 1e-5 => 1e-7

#model
trainingSecond = model.fit(tarin_data , epochs =  1, validation_data = validation_data, batch_size = 16)

with open('trainingSecond_1', 'w') as convert_file:
     convert_file.write(json.dumps(trainingSecond.history))

