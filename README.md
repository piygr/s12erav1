# Session 12 Assignment - (Google Colab Notebook)

## S12.ipynb
The notebook clones mainErav1.git repo using -

```
!git clone https://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@github.com/piygr/mainErav1.git
```

The repo has a models folder which contains the custom_resnet_lightning_s10.py file containing
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


Happy Modeling :-) 
 
