# Author: ArpiHunanyan
# Created: 18 May, 2022
# Email: arpi_hunanyan@edu.aua.am

from pickle import FALSE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Selection:

    def __init__(self, path = 'Results/selection/benchmark_df.csv'):
        # reading the data
        self.benchmark_df = pd.read_csv(path)

        # higlite "EfficientNetV2B2", "DenseNet121", "ResNet50", "MobileNetV3Large" models' info
        self.coloring = """ np.where(self.benchmark_df["model_name"] == "EfficientNetV2B2", '#756bb1', 
                                np.where(self.benchmark_df["model_name"] == "DenseNet121", '#4B8BBE', 
                                    np.where(self.benchmark_df["model_name"] == "ResNet50", '#238443',
                                        np.where(self.benchmark_df["model_name"] == "MobileNetV3Large",  '#FFE873',
                                            "#BFC2C5")
                                    )
                                )
                            ) """
        self.modelNames = ["EfficientNetV2B2", "DenseNet121", "ResNet50", "MobileNetV3Large"]



    def table(self, criteria = "accuracy") :
        #prints top 5 models' info sored by the input criteria
        self.benchmark_df.sort_values(["validation_" + criteria ], inplace = True, ascending = False)
        print(self.benchmark_df.head())

    def metricBar (self, criteria = "accuracy"):
        # bar plot for inpute criteria
        col = "self.benchmark_df." + criteria
        name = criteria.capitalize()
        self.benchmark_df.sort_values(by = [criteria], inplace = True, ascending = True)
        self.benchmark_df.reset_index(drop=True, inplace = True)
        self.barH(col = col, name =  name, title = " of Pre-Trained Models: 3 epochs with 5 top training layers")

    
    def parameterBar (self, criteria = "parameters"):
        # num_model_parameters, num_model_layers
         
        col = "self.benchmark_df.num_model_" + criteria
        name = criteria.capitalize()

        self.benchmark_df.sort_values(by = ["num_model_" + criteria], inplace = True, ascending = False)
        self.benchmark_df.reset_index(drop = True, inplace = True)

        self.barH(col = col, name =  name, textVisible = False, title = " of Pre-Trained Models")


    
    def barH(self, col, name, textVisible = True, title = "" ):
        
        
        fig, ax = plt.subplots(figsize =(16, 8))
        bar_plot = ax.barh(self.benchmark_df.model_name, eval(col), height = 0.5, color = eval(self.coloring) )

        if textVisible:
            for idx, rect in enumerate(bar_plot):
                if self.benchmark_df["model_name"][idx] in self.modelNames:
                    ax.text(np.round_(eval(col)[idx], 3) + 0.01,  rect.get_y(), str(int(np.round_(eval(col)[idx], 2) * 100 )) + "%", ha = 'center', rotation = 0, size = 7)

        for s in ['top', 'left', 'right']:
            ax.spines[s].set_visible(False)

            
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_tick_params(pad = 5)
        ax.yaxis.set_tick_params(pad = 0.5)
        
        # ax.set_xticklabels(["0%", "10%", "20%", "30%", "40%"])



        ax.xaxis.grid(b = True, color = 'grey',
                linestyle = '-.', linewidth = 0.5,
                alpha = 0.4)


        plt.xlabel( name )
        plt.ylabel('Model')
        plt.ylim(ymax = 33.5, ymin = -1)
        plt.title( name + title, pad = 10, size = 15)
    
        plt.show()

    def scaterplot(self, y = "accuracy", x = "params"):

            fig, ax = plt.subplots(figsize =(16, 9))
            ax.scatter(eval("self.benchmark_df.num_model_" + x), eval("self.benchmark_df." + y), color = eval(self.coloring))
            
            
            for i, txt in enumerate(self.benchmark_df["model_name"]):
                if self.benchmark_df["model_name"][i] in self.modelNames:
                    ax.annotate(txt, (eval("self.benchmark_df.num_model_" + x)[i], (eval("self.benchmark_df." + y)[i] + 0.005)), ha = 'center', rotation = 0, size = 6)

            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_ticks_position('none')
            ax.xaxis.set_tick_params(pad = 5)
            ax.yaxis.set_tick_params(pad = 0.55)


            for s in ['top', 'right']:
                ax.spines[s].set_visible(False)
                ax.grid(b = True, color ='grey',
                        linestyle ='-.', linewidth = 0.3,
                        alpha = 0.5)

            plt.xlabel(x.capitalize(),  size = 10)
            plt.ylabel(y.capitalize(),  size = 10)
            plt.title(y .capitalize() + ' vs Model\'s ' + x.capitalize() + ': 3 epochs with 5 top training layers', pad = 10, size = 15)
            plt.show()




class Execution:
    
    def __init__(self, path = "Results/5.ModelMobileNetV3Large_cont/Tuning"):

        with open(path, 'r') as results:
            self.results = eval(results.read().splitlines()[0])


    def plotResaluts(self, critaria = "accuracy"): 


        plt.plot(self.results[critaria],  label = 'training')
        plt.plot(self.results["val_" + critaria], label = 'validation') # shows overfitting

        plt.grid(color = 'grey', linewidth = 0.5, linestyle = '-.', alpha = 0.2, b = True)
        plt.xlabel("Epoch",  size = 15 )
        plt.ylabel(critaria.capitalize(), size = 15)

        plt.legend();
        plt.title(label = "Training " + critaria.capitalize()+ " vs Validation " + critaria.capitalize() , pad = 30, size = 30)
        plt.show() 


        
    def lastValues(self, matric ):

        for m  in matric:
            if (self.results.get(m) is not None):
                if ( not (isinstance(self.results.get(m), list) ) ):
                    
                        print(m ," : ",  np.round_(self.results.get(m), 3))
                    
                else:
                    print(m ," : ", np.round_(self.results.get(m)[-1], 3) )


    




    