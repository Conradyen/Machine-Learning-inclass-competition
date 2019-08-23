import numpy as np
import operator
from collections import Counter
from sklearn.preprocessing import StandardScaler
import math

def scale_by_group(indata,group,ngroup,file_name):

    _group = np.concatenate((group,indata),axis = 1)
    row,col = indata.shape
    afterscal = np.zeros((1,col))
    for i in group[0].unique():
        data = _group[_group[:,0] == i]
        standard_scaler = StandardScaler()
        groupscal_train = standard_scaler.fit(data[:,1:col+1]).transform(data[:,1:col+1])
        afterscal = np.concatenate((afterscal,groupscal_train),axis = 0)
    afterscal = afterscal[0:row,:]

    np.savetxt(file_name, afterscal, delimiter=",")


def max_class(label_list):

    '''
    conut number of class label
    '''

    class_count = Counter(label_list)
    max_class_ = max(class_count.items(), key=operator.itemgetter(1))[0]

    return max_class_

def dense_to_one_hot(labels_dense,num_classes = 2):

    '''
    return one not encodeing
    '''

    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()-1] = 1

    return labels_one_hot

def thresholding(pred,threshold):

    for i in range(len(pred)):
        floor = math.floor(pred[i])
        ceilng = math.ceil(pred[i])
        if pred[i] > (floor*(1-threshold) + ceilng*threshold):
            pred[i] = ceilng
        else:
            pred[i] = floor
    return pred


def to_submission(pred,file_name):

    with open(file_name, 'x') as f :
            f.write('ID,Prediction\n')
            for i in range(len(pred)) :
                f.write("".join([str(i+1),',',str(int(pred[i])),'\n']))
