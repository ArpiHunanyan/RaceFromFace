from  Visualisation import Plot

plot = Plot()
# data = plot.table()
# print(data.validation_accuracy )
#plot.parameterBar()
#'accuracy', 'recall', 'precision', 'specificity_at_sensitivity'
# plot.metricBar()
#plot.scaterplotWithParameters()
# opptions 'loss', 'accuracy', 'recall', 'precision', 'specificity_at_sensitivity'
plot.plotResaluts( path = "Results/model 4_2 ResultsResNet50/tuning" )
