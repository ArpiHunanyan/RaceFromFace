# Author: ArpiHunanyan
# Created: 18 May, 2022
# Email: arpi_hunanyan@edu.aua.am


import sys
from ResultInterpretation import Selection, Execution





# Set up___________________________________________________________________________________________________________________________________________________

print()
print("Demo_Result.py started execution...")
print()

print()
print("Do you want to compare all Keras Application result(True)? Or analize spesific model result(False)?")
print()
isSelection = True if input() == "True" else False

if ( isSelection) :

    selection = Selection()
    print()
    print("Choose: Bar Plot for model's stucure(0), Bar Plot for metrics(1), Scaterplot(3)")
    print()
    plotIndex = input()
    if (plotIndex == "0"):
        print()
        print("Options: parameters, layers")
        print()
        selection.parameterBar( input() )

    elif (plotIndex == "1"):

        print()
        print("Options: accuracy, recall, specificity")
        print()
        selection.metricBar(input())


    elif (plotIndex == "3"):
        print()
        print("Options for x: parameters, layers")
        x = input()
        print()
        print("Options for y: accuracy, recall,  specificity")
        y = input()
        selection.scaterplot(x = x, y = y)

    else:
        print("InputMismuch!")
        sys.exit(0)

else :
    print()
    print('Enter the path of the existing results:')
    path = [ "Results/1.ResNet121/Training",
             "Results/1.ResNet121/Tuning",
             "Results/2.EfficientNetV2B2/Training",
             "Results/3.MobileNetV3Large/Training", 
             "Results/3.MobileNetV3Large/Tuning", 
             "Results/4.ResNet50/Training",
             "Results/4.ResNet50/Tuning" , 
             "Results/5.MobileNetV3Large_cont/Tuning",
             "Results/6.MobileNetV3LargeEvaluationMask/Evaluation",
             "Results/7.MobileNetV3LargeMasked/Tuning"]

    _ = [print(str) for str in path]

    path = input()
    execution = Execution( path = path )

    validation = ['val_loss', 'val_accuracy',  'val_recall', 'val_precision',  'val_specificity_at_sensitivity'] # for trainig 
    train = ['loss', 'accuracy',  'recall', 'precision',  'specificity_at_sensitivity']  # for trainig and evaluation

    if ("Evaluation" in path ):
        execution.lastValues(matric = train )
        sys.exit(0)

    print()
    print("Choose: Plot(0), Values(1)")
    print()
    output = input()

    if ( output == "0"):
        print()
        print("Options: loss, accuracy")
        print()
        execution.plotResaluts(input())

    elif (output == "1"):
        print()
        print("Options: train, validation")
        print()
        execution.lastValues( matric = eval(input()) )
    
    else:
        print("InputMismuch!")
        sys.exit(0)









