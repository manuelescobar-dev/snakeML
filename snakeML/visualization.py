import math
import matplotlib.pyplot as plt
from snakeML.numpy_transformations import mcol, mrow

def find_dimensions(total):
    rows=1
    cols=1
    alternate=True
    while rows*cols<total:
        if alternate:
            rows+=1
            alternate=False
        else:
            cols+=1
            alternate=True
    return rows, cols


def histogram_attributeVSfrequency(data, labels, features, label_names, is_label_dict=False,row_attributes=False, dense=False, save = False, center_data=True):
    plt.figure()
    rows, cols=find_dimensions(len(features))
    if center_data:
        if row_attributes:
            data=data-mcol(data.mean(axis=1))
        else:
            data=data-mrow(data.mean(axis=0))
    if is_label_dict:
        lab=list(label_names.keys())
    else:
        lab=label_names
    for i in range(len(features)):
        plt.subplot(rows,cols,i+1)
        plt.xlabel(features[i])
        for j in range(len(lab)):
            if row_attributes:
                plt.hist(data[:, labels==j][i, :], density = dense, label = lab[j])        
            else:
                plt.hist(data[labels==j, :][:, i], density = dense, label = lab[j])        
        plt.legend()
        plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
    if save:
        plt.savefig('hist_%d.png' % i)
    plt.show()

def scatter_attributeVSattribute(data, labels, features, label_names, is_label_dict=False,row_attributes=False, dense=False, save = False, center_data=False):
    plt.figure()
    rows, cols=find_dimensions(math.factorial(len(features)-1))
    if center_data:
        if row_attributes:
            data=data-mcol(data.mean(axis=1))
        else:
            data=data-mrow(data.mean(axis=0))
    if is_label_dict:
        lab=list(label_names.keys())
    else:
        lab=label_names
    counter=1
    for i in range(len(features)):
        for k in range(len(features)):
            if i >= k:
                continue
            plt.subplot(rows,cols,counter)
            counter+=1
            plt.xlabel(features[i])
            plt.ylabel(features[k])
            for j in range(len(lab)):
                if row_attributes:
                    plt.scatter(data[:, labels==j][i, :],data[:, labels==j][k, :], label = lab[j])        
                else:
                    plt.scatter(data[labels==j, :][:, i],data[labels==j, :][:, k], label = lab[j])        
            plt.legend()
            plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
    if save:
        plt.savefig('scatter_%d_%d.png' % i, k)
    plt.show()

def scatter_categories(data, labels, label_names, is_label_dict=False,row_attributes=False,  save = False):
    if is_label_dict:
        lab=list(label_names.keys())
    else:
        lab=label_names
    plt.figure()
    for j in range(len(lab)):
        if row_attributes:
            plt.scatter(data[:, labels==j],data[:, labels==j], label = lab[j])        
        else:
            plt.scatter(data[labels==j, :],data[labels==j, :], label = lab[j])        
    plt.legend()
    plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
    if save:
        plt.savefig('scatter_%d_%d.png')
    plt.show()