This repo provides Python implementation of the work "Deep Clustering with Incomplete Noisy Pairwise Annotations: A Geometric Regularization Approach".


Before running, please install all package listed in requirements.txt. 


we have set a default configution. Without any change, you should able to perform:

- Training the experiment on ImageNet10 in noiseless pairwise setting:
```
python our_model__training_imagenet10.py
```
This will load a pairwise labels dataset that have been drawn randomly and stored to datasets/. This dataset containing 10k pairs drawn randomly from the training part of ImageNet10. 

- Evaluate performance in terms of ACC, NMI, and ARI:
```
python our_model__eval.py
```
This will evaluate the learned mapping f using 2k test dataset.


Other options:
- You can create different pairwise dataset by inspecting file `imagenet10_create_pair.py`, similarly for stl10 and cifar10.
