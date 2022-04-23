from tensorflow import keras
from  DataPreparation import getValidation, getTrain
from DenseNetImplementation import DenceNetClassifier
import json


print("Data Preperation")
train_data, train_labels = getTrain(12000)
validation_data = getValidation(5000)

model = DenceNetClassifier()
model.compile()
print()

# ## Train the top layer (for classifaction) 
print("Training the top layer (for classifaction) ")
trainingFirst = model.fit(x = train_data, y = train_labels,  epochs = 50, validation_data = validation_data, batch_size = 16)

  
with open('trainingFirst_1', 'w') as convert_file:
     convert_file.write(json.dumps(trainingFirst.history))


print()
## Fine - Tuning
print("Fine - Tuning ")
model.setTrainable(True)
model.compile(optimizer = keras.optimizers.Adam(1e-7)) # 1e-5 => 1e-7

#model
trainingSecond = model.fit(x = train_data, y = train_labels,  epochs =  5, validation_data = validation_data)

with open('trainingSecond_1', 'w') as convert_file:
     convert_file.write(json.dumps(trainingSecond.history))

