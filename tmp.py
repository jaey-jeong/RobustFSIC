import pickle

train_path = '../miniimagenet/mini-imagenet-cache-train.pkl'
val_path = '../miniimagenet/mini-imagenet-cache-val.pkl'
test_path = '../miniimagenet/mini-imagenet-cache-test.pkl'

with open(val_path, 'rb') as f :
    data = pickle.load(f)

print(len(data['image_data']), len(data['class_dict']))
print(data['image_data'], data['class_dict'])