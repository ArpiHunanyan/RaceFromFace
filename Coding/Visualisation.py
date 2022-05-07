from inspect import trace
import pandas as pd
import matplotlib.pyplot as plt

class Plot:
    
    def __init__(self, path = 'Results/benchmark_df_topLayer.csv'):
        self.benchmark_df = pd.read_csv(path)

    def metricBar (self, x = "accuracy"):
        col = "self.benchmark_df.validation_" + x
        name = x.capitalize()
        self.benchmark_df.sort_values(["validation_" + x ], inplace = True, ascending = True)
        self.barH(col, name)

    def parameterBar (self):
        col = "self.benchmark_df.num_model_params" 
        self.benchmark_df.sort_values(['num_model_params'], inplace = True, ascending = False)
        self.barH(col,  "Paramenter")

    def barH(self, col, name):
        
        
        fig, ax = plt.subplots(figsize =(16, 8))
        ax.barh(self.benchmark_df.model_name, eval(col), height = 0.5)

        for s in ['top', 'left', 'right']:
            ax.spines[s].set_visible(False)

            
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_tick_params(pad = 5)
        ax.yaxis.set_tick_params(pad = 10)

        ax.xaxis.grid(b = True, color = 'grey',
                linestyle = '-.', linewidth = 0.5,
                alpha = 0.2)


        plt.xlabel( name )
        plt.ylabel('Models')
        plt.title("Validation " +  name + "after 3 Epochs with Freezed the convolutional base.")
        plt.legend(bbox_to_anchor=(1, 1), loc='upper left');
        plt.show()

    def scaterplotWithParameters(self, x = "accuracy"):
        fig, ax = plt.subplots(figsize =(16, 9))
        ax.scatter(self.benchmark_df.num_model_params, eval("self.benchmark_df.validation_" + x))
        for s in ['top', 'right']:
            ax.spines[s].set_visible(False)
            ax.grid(b = True, color ='grey',
                    linestyle ='-.', linewidth = 0.3,
                    alpha = 0.5)

            plt.xscale('log')
            plt.xlabel('Number of Parameters in Model')
            plt.ylabel('Validation Accuracy after 3 Epochs')
            plt.title('Accuracy vs Model Size')
            plt.legend(bbox_to_anchor=(1, 1), loc='upper left');
            plt.show()


    def plotResaluts(path = "Results/trainingResults_efficientnetv2_1", critaria = "accuracy"): 

        with open("Results/trainingResults_efficientnetv2_1", 'r') as training:
            training = eval(training.read().splitlines()[0])


        plt.plot(training[critaria],  label = 'training')
        plt.plot(training["val_" + critaria], label = 'validatin') # shows overfitting

        plt.grid(color = 'grey', linewidth = 0.5, linestyle = '-.', alpha = 0.2, b = True)
        plt.xlabel(critaria)
        plt.ylabel("score")

        plt.legend();
        plt.title(label = "Trainig " + critaria + " vs Validation " + critaria , pad = 30)
        plt.show() 



    