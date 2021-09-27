
For basic usage, please refer to the [wiki page](https://github.com/dinhhuy2109/python-cope/wiki)

<!---
# Basic usage
## On the covariance of X in the AX=XB
The classical hand-eye calibration problem consists in identifying the rigidbody
transformation eTc between a camera mounted on the end-effector of
a robot and the end-effector itself (see the below figure). The problem is usually framed as the AX=XB problem. In this functionality, we provide a solution not only solving for X but also predicting the covariance of X from those of A and B, where A and B are now randomly perturbed transformation matrices. 

For more details, please refer to the accompanying paper [On the covariance of X in the AX=XB](https://arxiv.org/pdf/1706.03498.pdf).

<p align="center">
  <img src="medias/hand-eye.png" width="200"/>
</p>

The following code snippets shows basic usage of `cope` in finding the covariance of X:

First, import necessary functions
```python