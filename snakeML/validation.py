import numpy


def matrix_max_error(m1,m2):
    print("Error: ",numpy.abs(m1 - m2).max())

def accuracy(predicted, test):
    correct=0
    for i in range(len(predicted)):
        if predicted[i]==test[i]:
            correct+=1
    #print("Accuracy: ",(correct/len(predicted))*100, "%")
    return correct/len(predicted)*100

def error(predicted, test):
    wrong=0
    for i in range(len(predicted)):
        if predicted[i]!=test[i]:
            wrong+=1
    #print("Error: ",(wrong/len(predicted))*100, "%")
    return wrong/len(predicted)*100