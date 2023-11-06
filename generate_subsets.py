import os
import yaml
#train_sample_ids = list(range(0,1000))
#val_sample_ids = list(range(1000,2000))
#train_sample_ids = list(range(2000,3000))
#val_sample_ids = list(range(3000,4000))
train_sample_ids = list(range(4000,5000))
val_sample_ids = list(range(5000,6000))
n_samples_train = 1000
n_samples_val = 1000

print("First sample of train:", train_sample_ids[0])
print("First sample of val:", val_sample_ids[0])

data = {'Train_sample_ids': train_sample_ids,
        'Val_sample_ids': val_sample_ids}

path='subsets/'+str(n_samples_train)+'train+'+str(n_samples_val)+'val_samples/'
if (len(val_sample_ids) == n_samples_train):
    if(not os.path.exists(path)):
        print("Creating new directory because it didn't exist")
        os.makedirs(path)
    print("Correct subset size, creating yml-file")
    with open(path+'Subset3.yml', 'w') as file:
        yaml.dump(data, file, default_flow_style=False)
else:
    print("incorrect size:",len(sample_ids))