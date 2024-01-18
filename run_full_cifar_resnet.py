#implement try-catch
#



import lightning as L
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
import torch.utils.data as data
from pytorch_lightning import loggers as pl_loggers
from train import *
import yaml
import traceback
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


cfgs_vit_2 = [
    'configs_second_narvi_run/resnet18_narvi_1.yml',
    'configs_second_narvi_run/ViT_narvi_1.yml',
    'configs_second_narvi_run/resnet18_narvi_2.yml',
    'configs_second_narvi_run/ViT_narvi_2.yml',
    'configs_second_narvi_run/resnet18_narvi_3.yml',
    'configs_second_narvi_run/ViT_narvi_3.yml',
    'configs_second_narvi_run/resnet18_narvi_4.yml',
    'configs_second_narvi_run/ViT_narvi_4.yml',
    'configs_second_narvi_run/resnet18_narvi_5.yml',
    'configs_second_narvi_run/ViT_narvi_5.yml',
    'configs_second_narvi_run/resnet18_narvi_6.yml',
    'configs_second_narvi_run/ViT_narvi_6.yml',
    'configs_second_narvi_run/resnet18_narvi_7.yml',
    'configs_second_narvi_run/ViT_narvi_7.yml',
    'configs_second_narvi_run/resnet18_narvi_8.yml',
    'configs_second_narvi_run/ViT_narvi_8.yml',
    'configs_second_narvi_run/resnet18_narvi_9.yml',
    'configs_second_narvi_run/ViT_narvi_9.yml',
    'configs_second_narvi_run/resnet18_narvi_10.yml',
    'configs_second_narvi_run/ViT_narvi_10.yml'
    ]
cfgs_vit_2 = [
    'configs_second_narvi_run/resnet18_narvi_10.yml'
    ]

cfg_0 = 'configs_second_narvi_run/resnet18_narvi_1.yml'

cfgs = cfgs_vit_2

train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.49139968, 0.48215841, 0.44653091],
                                                                                  [0.24703223, 0.24348513,
                                                                                   0.26158784]), ])
test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.49139968, 0.48215841, 0.44653091],
                                                                                 [0.24703223, 0.24348513,
                                                                                  0.26158784]), ])
cifar_train = CIFAR10(root="data/", train=True, transform=train_transform, download=True)
run_counter = 0

for cfgpath in cfgs:
    print("Run counter:", run_counter)
    with open(cfgpath, 'r') as file:
        cfg = yaml.safe_load(file)
    if torch.cuda.is_available() and (list(range(torch.cuda.device_count() > 1))):
        print("Using Narvi settings with ", cfg["TRAINER"]["MAX_EPOCHS"], " epochs.")
        pass
    else:
        cfg["TRAINER"]["MAX_EPOCHS"] = 5
        print("Using test settings with ", cfg["TRAINER"]["MAX_EPOCHS"], " epochs.")

    train_dataset = cifar_train
    with open(cfg["DATASET"]["DATA_SAMPLESET_NAME"], 'r') as file:
        sampleset = yaml.load(file, Loader=yaml.FullLoader)
    train_set = torch.utils.data.Subset(train_dataset, sampleset['Train_sample_ids'])
    val_set = torch.utils.data.Subset(train_dataset, sampleset['Val_sample_ids'])

    # We define a set of data loaders that we can use for various purposes later.
    train_loader = data.DataLoader(train_set, batch_size=cfg["TRAIN_LOADER"]["BATCH_SIZE"], shuffle=False, drop_last=True, pin_memory=True, num_workers=4)
    val_loader = data.DataLoader(val_set, batch_size=cfg["TRAIN_LOADER"]["BATCH_SIZE"], shuffle=False, drop_last=False, num_workers=4)

    logger = pl_loggers.TensorBoardLogger(cfg["LOGGER"]["logger_path"], name=cfg["LOGGER"]["logger_name"])
    try:
        run_counter = run_counter+1
#        if run_counter < 3:
#            print(int('e'))
        model, trainer = train_model(cfg, logger, train_loader, val_loader)
        trainer.logger.finalize("ok")
    except KeyboardInterrupt:
        print("KeyboardInterrupt. Aborting.")
        raise
    except Exception as err:
        print(type(err))  # the exception type
        print(err)  # actual error
        traceback.print_exc()

