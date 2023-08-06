# Session 12 Assignment - (Google Colab Notebook)

## S12.ipynb
The notebook clones mainErav1.git repo using -

```
!git clone https://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@github.com/piygr/mainErav1.git
```

The repo has a models folder which contains the [custom_resnet_lightning_s10.py](https://github.com/piygr/mainErav1/blob/main/models/custom_resnet_lightning_s10.py) file containing
**S10LightningModel** a LightningModule class.



If the repo is cloned, then move the mainErav1 folder

```
%cd mainErav1
```

To fetch the latest code from the mainErav1 repo do -

```
!git pull origin main
```

We might have to install few packages for eg.
```
!pip install torch_lr_finder
!pip install grad-cam
!pip install pytorch-lightning
```

Once the packages are installed, we have to import functionalities from main.py & utils.py
can train & experiment with the model.

### LRFinder graph- 
<img width="770" alt="Screenshot 2023-08-06 at 9 47 31 AM" src="https://github.com/piygr/s12erav1/assets/135162847/698edd3b-0f06-4ac5-9e6d-b58b9957a5ba">

### The brief model summary & the training logs -
```
INFO:pytorch_lightning.callbacks.model_summary:
  | Name       | Type        | Params
-------------------------------------------
0 | prep_layer | Sequential  | 1.9 K 
1 | x1         | Sequential  | 74.1 K
2 | R1         | ResnetBlock | 295 K 
3 | layer2     | Sequential  | 295 K 
4 | x2         | Sequential  | 1.2 M 
5 | R2         | ResnetBlock | 4.7 M 
6 | pool       | MaxPool2d   | 0     
7 | fc         | Linear      | 5.1 K 
-------------------------------------------
6.6 M     Trainable params
0         Non-trainable params
6.6 M     Total params
26.301    Total estimated model params size (MB)
0.0003853528593710531
/usr/local/lib/python3.10/dist-packages/pytorch_lightning/trainer/connectors/data_connector.py:480: PossibleUserWarning: Your `val_dataloader`'s sampler has shuffling enabled, it is strongly recommended that you turn shuffling off for val/test dataloaders.
  rank_zero_warn(
Epoch 23: 100%
98/98 [00:24<00:00, 4.00it/s, v_num=0]
Epoch  0
Train Loss:  2.1498637199401855  Accuracy:  24.746%  [ 12373 / 50000 ]
Validation Loss:  1.6777385473251343  Accuracy:  38.78809869375907%  [ 4276 / 11024 ]

Epoch  1
Train Loss:  1.442869782447815  Accuracy:  47.968%  [ 23984 / 50000 ]
Validation Loss:  1.1849777698516846  Accuracy:  57.89%  [ 5789 / 10000 ]

Epoch  2
Train Loss:  1.128368854522705  Accuracy:  60.086%  [ 30043 / 50000 ]
Validation Loss:  1.0484631061553955  Accuracy:  64.31%  [ 6431 / 10000 ]

Epoch  3
Train Loss:  0.9051997065544128  Accuracy:  68.09%  [ 34045 / 50000 ]
Validation Loss:  0.8215187191963196  Accuracy:  71.9%  [ 7190 / 10000 ]

Epoch  4
Train Loss:  0.7344226241111755  Accuracy:  74.482%  [ 37241 / 50000 ]
Validation Loss:  0.7320188879966736  Accuracy:  75.46%  [ 7546 / 10000 ]

Epoch  5
Train Loss:  0.6181399822235107  Accuracy:  78.328%  [ 39164 / 50000 ]
Validation Loss:  0.6262689828872681  Accuracy:  78.38%  [ 7838 / 10000 ]

Epoch  6
Train Loss:  0.5381202101707458  Accuracy:  81.46%  [ 40730 / 50000 ]
Validation Loss:  0.6052538156509399  Accuracy:  79.17%  [ 7917 / 10000 ]

Epoch  7
Train Loss:  0.4857393801212311  Accuracy:  83.202%  [ 41601 / 50000 ]
Validation Loss:  0.6124098896980286  Accuracy:  79.23%  [ 7923 / 10000 ]

Epoch  8
Train Loss:  0.44854506850242615  Accuracy:  84.564%  [ 42282 / 50000 ]
Validation Loss:  0.5080311894416809  Accuracy:  82.27%  [ 8227 / 10000 ]

Epoch  9
Train Loss:  0.40878310799598694  Accuracy:  86.12%  [ 43060 / 50000 ]
Validation Loss:  0.4635935425758362  Accuracy:  84.08%  [ 8408 / 10000 ]

Epoch  10
Train Loss:  0.38101285696029663  Accuracy:  86.804%  [ 43402 / 50000 ]
Validation Loss:  0.5572211146354675  Accuracy:  81.27%  [ 8127 / 10000 ]

Epoch  11
Train Loss:  0.3528030216693878  Accuracy:  88.058%  [ 44029 / 50000 ]
Validation Loss:  0.4275229573249817  Accuracy:  85.81%  [ 8581 / 10000 ]

Epoch  12
Train Loss:  0.319828599691391  Accuracy:  89.134%  [ 44567 / 50000 ]
Validation Loss:  0.4077281653881073  Accuracy:  85.96%  [ 8596 / 10000 ]

Epoch  13
Train Loss:  0.2897335886955261  Accuracy:  90.112%  [ 45056 / 50000 ]
Validation Loss:  0.41690197587013245  Accuracy:  85.74%  [ 8574 / 10000 ]

Epoch  14
Train Loss:  0.26105326414108276  Accuracy:  91.234%  [ 45617 / 50000 ]
Validation Loss:  0.39770808815956116  Accuracy:  86.64%  [ 8664 / 10000 ]

Epoch  15
Train Loss:  0.23679150640964508  Accuracy:  92.032%  [ 46016 / 50000 ]
Validation Loss:  0.35208040475845337  Accuracy:  87.93%  [ 8793 / 10000 ]

Epoch  16
Train Loss:  0.20291069149971008  Accuracy:  93.328%  [ 46664 / 50000 ]
Validation Loss:  0.33846718072891235  Accuracy:  88.88%  [ 8888 / 10000 ]

Epoch  17
Train Loss:  0.17984144389629364  Accuracy:  94.14%  [ 47070 / 50000 ]
Validation Loss:  0.29642051458358765  Accuracy:  89.97%  [ 8997 / 10000 ]

Epoch  18
Train Loss:  0.15271352231502533  Accuracy:  95.068%  [ 47534 / 50000 ]
Validation Loss:  0.287870317697525  Accuracy:  90.48%  [ 9048 / 10000 ]

Epoch  19
Train Loss:  0.12519571185112  Accuracy:  96.118%  [ 48059 / 50000 ]
Validation Loss:  0.26041147112846375  Accuracy:  91.34%  [ 9134 / 10000 ]

Epoch  20
Train Loss:  0.1021689921617508  Accuracy:  97.018%  [ 48509 / 50000 ]
Validation Loss:  0.2544329762458801  Accuracy:  92.21%  [ 9221 / 10000 ]

Epoch  21
Train Loss:  0.0890466645359993  Accuracy:  97.42%  [ 48710 / 50000 ]
Validation Loss:  0.2417704313993454  Accuracy:  92.17%  [ 9217 / 10000 ]

Epoch  22
Train Loss:  0.07858018577098846  Accuracy:  97.794%  [ 48897 / 50000 ]
Validation Loss:  0.23496107757091522  Accuracy:  92.42%  [ 9242 / 10000 ]

Epoch  23
Train Loss:  0.0734831914305687  Accuracy:  97.994%  [ 48997 / 50000 ]
Validation Loss:  0.23382949829101562  Accuracy:  92.44%  [ 9244 / 10000 ]

INFO:pytorch_lightning.utilities.rank_zero:`Trainer.fit` stopped: `max_epochs=24` reached.
```

### Missclassified images

<img width="640" alt="Screenshot 2023-08-06 at 9 49 11 AM" src="https://github.com/piygr/s12erav1/assets/135162847/d5b7ccd2-fc82-4565-aa1a-fb508b329594">

### GradCAM images

<img width="682" alt="Screenshot 2023-08-06 at 9 49 23 AM" src="https://github.com/piygr/s12erav1/assets/135162847/fc688a76-471b-446b-86b4-589127f15ad2">

### Model performance plots

<img width="1011" alt="Screenshot 2023-08-06 at 9 49 57 AM" src="https://github.com/piygr/s12erav1/assets/135162847/61622c4a-fff0-4296-b42f-d9fe836ae957">
<img width="1000" alt="Screenshot 2023-08-06 at 9 50 06 AM" src="https://github.com/piygr/s12erav1/assets/135162847/2c614ceb-f2c1-4202-994e-7f2e75571da0">




Happy Modeling :-) 
 
