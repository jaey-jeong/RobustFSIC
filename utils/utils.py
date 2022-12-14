import pickle
import torch
import numpy as np
def key_to_class(dict):
    keys = list(dict.keys())
    for key in keys :
        dict[str(keys.index(key))] = dict.pop(key)
    #print(dict)
    return dict

def value_to_class(dict) :
    
    labels = []
    keys = list(dict.keys())
    for key in keys :
        size = len(dict[key])
        if len(key) > 1 :
            for _ in range(size) :
                labels.append(key)
        else :
            labels.extend(key*size)
    return labels

def load_pickle(path) :
    with open(path, 'rb') as f :
        data = pickle.load(f)
    return data

def adversarial_reweighting(loss, T):
    w = []
    alpha = 0.5
    beta = 0.5
    loss = loss.tolist()
    for V in loss :
        w.append((alpha * (1 + np.tanh(4 - 10 * (1 + V.index(max(V)) / (T + 1)))) / 2 
                            + beta * (max(V) / max(map(max, loss)))))
    w = np.array(w)
    w = np.expand_dims(w, axis =1)
    return w

def cross_entropy(targets, pred, class_num):
    delta = 1e-15
    pred = pred + delta
    targets = torch.eye(class_num)[targets].long()
    pred = torch.log(pred)
    loss = -torch.sum(targets*pred)

    return loss/float(pred.shape[0])

def adversarial_reweighting_crossentropy(targets, pred, weight, class_num):
    delta = 1e-15
    pred = pred + delta
    targets = torch.eye(class_num)[targets].long()
    tmp = targets * torch.log(pred)
    tmp = torch.sum(tmp, axis = 0)
    loss = -torch.sum(weight * tmp)

    return loss/float(pred.shape[0])

def get_adv_cross_entropy(targets, pred, class_num):
    delta = 1e-15
    pred = pred + delta
    targets = torch.eye(class_num)[targets].long()
    tmp = []
    for t, p in zip(targets, pred) :
        tmp.append(-torch.sum((t * torch.log(p))))
    tmp = torch.Tensor(tmp)
    tmp = torch.t(tmp)
    #print(type(tmp))
    loss = torch.unsqueeze(tmp, 1)

    return loss