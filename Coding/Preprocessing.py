################ MUTED CONTEND #####################






# import pandas as pd
# from tqdm import tqdm
# from Model import kerasModelNames, Classifier

## adding colums for mask images
# df = pd.read_csv("Data/fairface_label_val.csv", header = 0)
# df["file_masked"] = [ i.replace('.jpg', '_surgical.jpg').replace('val', "val_masked")  for i in df.file ]
# df.to_csv("Data/fairface_label_val.csv", index = False)



# df = pd.read_csv("Data/fairface_label_train.csv", header = 0)
# df["file_masked"] = [ i.replace('.jpg', '_surgical.jpg').replace('train', "train_masked")  for i in df.file ]
# df.to_csv("Data/fairface_label_train.csv", index = False)











# # # model layers addition
# path = 'Results/selection/benchmark_df.csv'

# # model_names = kerasModelNames()
# df = pd.read_csv(path)
# # df["num_model_layers"] = 0

# # model_names = kerasModelNames()
# # for model_name in tqdm(model_names):
# #         #  "NASNetLarge" requires input images with size (331,331)
# #         if 'NASNetLarge' in model_name:
# #             continue 

# #         clf_model = Classifier( modelName = model_name)
# #         print(clf_model.baseModelLayersCount())
# #         df["num_model_layers"][df["model_name"] ==  model_name] = clf_model.baseModelLayersCount()
# # df.columns = df.columns.str.replace('num_model_params', 'num_model_parameters')
# # df.columns = df.columns.str.replace("validation_accuracy", 'accuracy')
# # df.columns = df.columns.str.replace('val_recall', 'recall')
# # df.columns = df.columns.str.replace('val_specificity_at_sensitivity', 'specificity at sensitivity')
# df.columns = df.columns.str.replace( 'specificity at sensitivity', 'specificity')
# print(df.head())
# df.to_csv('Results/benchmark_df_2.csv', index = False)















