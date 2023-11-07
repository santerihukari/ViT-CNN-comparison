import os
import yaml
import random
from torchvision.datasets import CIFAR10
from torchvision import transforms



def generate_random_subset(n_samples_total, n_samples_to_return):
    return random.sample(range(0, n_samples_total), n_samples_to_return), str(n_samples_to_return)+" samples"

def generate_sequential_subset(sample_begin, sample_end):
    return list(range(sample_begin, sample_end)), "Sequential from "+str(sample_begin)
def generate_full_subset(dataset_name, n_samples_total, n_samples_train, n_samples_val, n_samples_test=0):
    pass
def generate_subset_file(dataset_name, dataset_size, n_samples_train, n_samples_val, n_samples_test=0, method='random'):
    if method=='random':
        subset_combined, method = generate_random_subset(dataset_size, n_samples_train + n_samples_val + n_samples_test)
    elif method=='sequential':
        subset_combined, method = generate_sequential_subset(0, n_samples_train+n_samples_val+n_samples_test)
    data = {'Original dataset':
                dataset_name,
            'Method of sampling': str(method),
            'Train_sample_ids':
                subset_combined[0:n_samples_train],
            'Val_sample_ids':
                subset_combined[n_samples_train:n_samples_train+n_samples_val],
            'Test_sample_ids':
                subset_combined[n_samples_train+n_samples_val:n_samples_train+n_samples_val+n_samples_test]}
    path = 'subsets/' + dataset_name + '/'

    for i in range(0, 100):
        filename = dataset_name+'_'+str(i)
        if (not os.path.exists(path+filename+".yml")):
            if(not os.path.exists(path)):
                print("Creating a new directory because it didn't exist")
                os.makedirs(path)
            print("Creating yml-file "+path+filename+".yml")
            with open(path+filename+".yml", 'w') as file:
                yaml.dump(data, file, default_flow_style=False)
            return data

train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784]),])
train_dataset = CIFAR10(root="data/", train=True, transform=train_transform, download=True)
#print(list(range(0,10)))
#print(len(train_dataset))
print(generate_subset_file("CIFAR-10_full_sequential",50000, 35000, 15000, 0, method='sequential'))