import numpy

def oneHotEncoding(labeled_data, return_dictionary= False):
    labels={}
    for i, name in enumerate(numpy.unique(labeled_data)):
        labels[name]=i
    new_labels=[]
    for i in labeled_data:
        new_labels.append(labels[i])
    if return_dictionary:
        return numpy.array(new_labels, dtype=numpy.int32), labels
    else:
        return numpy.array(new_labels, dtype=numpy.int32)