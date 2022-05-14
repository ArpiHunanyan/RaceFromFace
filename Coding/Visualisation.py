from inspect import trace
from numpy import size
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Plot:
    
    def __init__(self, path = 'Results/selection/benchmark_df_2.csv'):
        self.coloring = """ np.where(self.benchmark_df["model_name"] == "EfficientNetV2B2", '#FFE873', np.where(self.benchmark_df["model_name"] == "DenseNet121", '#4B8BBE', "#BFC2C5")) """
        # self.coloring = """ np.where(self.benchmark_df["model_name"] == "EfficientNetV2B2", '#FFE873', np.where(self.benchmark_df["model_name"] == "DenseNet121", '#4B8BBE', np.where( np.where( self.benchmark_df["model_name"] == "EfficientNetV2L", "#A22160", "#BFC2C5"))) """
        self.benchmark_df = pd.read_csv(path)
        # self.benchmark_df.sort_values(["validation_accuracy" ], inplace = True, ascending = True)
        # self.benchmark_df.to_csv("Results/benchmark_df_1.csv", index = False)

    def metricBar (self, x = "accuracy"):
        col = "self.benchmark_df.validation_" + x
        name = x.capitalize()
        self.benchmark_df.sort_values(["validation_" + x ], inplace = True, ascending = True)
        self.benchmark_df.reset_index(drop=True, inplace = True)
        self.barH(col, name)

    def parameterBar (self):
        col = "self.benchmark_df.num_model_params" 

        self.benchmark_df.sort_values(['num_model_params'], inplace = True, ascending = False)
        self.benchmark_df.reset_index(drop=True, inplace = True)

        self.barH(col = col, name = "Parameters", textVisible = False, title = " of Freezing Pre-Trained Models")

    def barH(self, col, name, textVisible = True, title = " of Freezing Pre-Trained Models: 3 Epochs"):
        
        
        fig, ax = plt.subplots(figsize =(16, 8))
        bar_plot = ax.barh(self.benchmark_df.model_name, eval(col), height = 0.5, color = eval(self.coloring) )
        if textVisible:
            for idx, rect in enumerate(bar_plot):
                if self.benchmark_df["model_name"][idx] in ["EfficientNetV2B2", "DenseNet121"]:
                    ax.text(np.round_(eval(col)[idx], 3) + 0.01,  rect.get_y(), np.round_(eval(col)[idx], 3), ha = 'center', rotation = 0, size = 7)

        for s in ['top', 'left', 'right']:
            ax.spines[s].set_visible(False)

            
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_tick_params(pad = 5)
        ax.yaxis.set_tick_params(pad = 0.5)

        ax.xaxis.grid(b = True, color = 'grey',
                linestyle = '-.', linewidth = 0.5,
                alpha = 0.2)


        plt.xlabel( name )
        plt.ylabel('Model')
        plt.title( name + title, pad = 10, size = 15)
    
        plt.show()

    def scaterplot(self, y = "accuracy", x = "params"):
        fig, ax = plt.subplots(figsize =(16, 9))
        scatter_plot = ax.scatter(self.benchmark_df.num_model_, eval("self.benchmark_df.validation_" + y), color = eval(self.coloring))
        
        for i, txt in enumerate(self.benchmark_df["model_name"]):
            if self.benchmark_df["model_name"][i] in ["EfficientNetV2B2", "DenseNet121"]:
                ax.annotate(txt, (self.benchmark_df.num_model_params[i], (eval("self.benchmark_df.validation_" + y)[i] + 0.005)), ha = 'center', rotation = 0, size = 6)

        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_tick_params(pad = 5)
        ax.yaxis.set_tick_params(pad = 0.55)


        for s in ['top', 'right']:
            ax.spines[s].set_visible(False)
            ax.grid(b = True, color ='grey',
                    linestyle ='-.', linewidth = 0.3,
                    alpha = 0.5)

            plt.xscale('log')
            plt.xlabel('Parameters',  size = 10)
            plt.ylabel('Accuracy',  size = 10)
            plt.title('Accuracy vs Model Size: 3 Epochs', pad = 10, size = 15)
            plt.show()


    def plotResaluts(self, path, critaria = "accuracy"): 

        with open(path, 'r') as training:
            training = eval(training.read().splitlines()[0])


        plt.plot(training[critaria],  label = 'training')
        plt.plot(training["val_" + critaria], label = 'validatin') # shows overfitting

        plt.grid(color = 'grey', linewidth = 0.5, linestyle = '-.', alpha = 0.2, b = True)
        plt.xlabel("epoch",  size = 15 )
        plt.ylabel(critaria, size = 15)

        plt.legend();
        plt.title(label = "Trainig " + critaria + " vs Validation " + critaria , pad = 30, size = 30)
        plt.show() 

    def table(self, x = "accuracy") :

        self.benchmark_df.sort_values(["validation_" + x ], inplace = True, ascending = False)
        print(self.benchmark_df.head())
    




    