import sys
import tensorflow as tf
# import tensorflow_datasets as tfds
import pandas as pd
import matplotlib.pyplot as plt
import inspect
from tqdm import tqdm
from  DataPreparation import getValidation, getTrain
from tensorflow.keras.applications import DenseNet201


# Set batch size for training and validation
batch_size = 16


# List all available models
model_dictionary = {m[0]:m[1] for m in inspect.getmembers(tf.keras.applications, inspect.isfunction)}


# Download the training and validation data
# (train, validation), metadata = tfds.load('cats_vs_dogs', split=['train[:70%]', 'train[70%:]'], with_info=True, as_supervised=True)

train_data =  getTrain(tensor = True)
validation_data =  getValidation(tensor = True)
# Number of training examples and labels
num_train = 86744 #len(list(train))
num_validation = 10954 #len(list(validation))
num_classes = 7
num_iterations = int(num_train/batch_size)

# Print important info
print(f'Num train images: {num_train} \
        \nNum validation images: {num_validation} \
        \nNum classes: {num_classes} \
        \nNum iterations per epoch: {num_iterations}')


def normalize_img(image, label, img_size):
    # Resize image to the desired img_size and normalize it
    # One hot encode the label
    image = tf.image.resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.
    return image, label
    
    
# Run preprocessing
# train_processed_224 = normalize_img(train, y_train, (224,224))
# validation_processed_224 = normalize_img(validation, y_validation, (224,224))
# train_processed_331 = normalize_img(train, y_train, (331,331))
# validation_processed_331 = normalize_img(validation, y_validation, (331,331))

train_processed_224 = train_data # train, y_train
validation_processed_224 = validation_data # validation, y_validation


#sys.exit(0)
# Loop over each model available in Keras
model_benchmarks = {'model_name': [], 'num_model_params': [], 'validation_accuracy': []}


count = 0
for model_name, model in tqdm(model_dictionary.items()):
    print("------------------------------------------------")
    print("Count: ", count)
    print("model_name", model_name)
    count = count + 1
    print("------------------------------------------------")
    # Special handling for "NASNetLarge" since it requires input images with size (331,331)
    if 'NASNetLarge' in model_name:
        continue 
        input_shape=(331,331,3)
        train_processed = train_processed_331
        validation_processed = validation_processed_331
    else:
        input_shape=(224,224,3)
        train_processed = train_processed_224
        validation_processed = validation_processed_224
        
    # load the pre-trained model with global average pooling as the last layer and freeze the model weights
    pre_trained_model = model(include_top=False, pooling='avg', input_shape=input_shape)
    pre_trained_model.trainable = False

    # custom modifications on top of pre-trained model and fit
    input_tensor = tf.keras.Input(input_shape, name = "InputFaceImg")
    base_output_tensor = pre_trained_model(input_tensor, training = False)

    output_tensor = tf.keras.layers.Dense(num_classes,  activation = "softmax", name = "OutputLayer")(base_output_tensor)
    clf_model = tf.keras.Model(input_tensor, output_tensor, name = "DenseNet121")


#     clf_model = tf.keras.models.Sequential(input_tensor)
#     clf_model.add(pre_trained_model)
#     clf_model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    clf_model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
    history = clf_model.fit(train_processed, epochs = 3, validation_data = validation_processed, steps_per_epoch=num_iterations)
    
    # Calculate all relevant metrics
    model_benchmarks['model_name'].append(model_name)
    model_benchmarks['num_model_params'].append(pre_trained_model.count_params())
    model_benchmarks['validation_accuracy'].append(history.history['val_accuracy'][-1])


# Convert Results to DataFrame for easy viewing
benchmark_df = pd.DataFrame(model_benchmarks)

# sort in ascending order of num_model_params column
benchmark_df.sort_values('num_model_params', inplace=True)

# write results to csv file
benchmark_df.to_csv('benchmark_df.csv', index=False)
benchmark_df


# # Loop over each row and plot the num_model_params vs validation_accuracy
# markers=[".",",","o","v","^","<",">","1","2","3","4","8","s","p","P","*","h","H","+","x","X","D","d","|","_",4,5,6,7,8,9,10,11]
# plt.figure(figsize=(7,5))

# for row in benchmark_df.itertuples():
#     plt.scatter(row.num_model_params, row.validation_accuracy, label=row.model_name, marker=markers[row.Index], s=150, linewidths=2)
    
# plt.xscale('log')
# plt.xlabel('Number of Parameters in Model')
# plt.ylabel('Validation Accuracy after 3 Epochs')
# plt.title('Accuracy vs Model Size')

# # Move legend out of the plot
# plt.legend(bbox_to_anchor=(1, 1), loc='upper left');