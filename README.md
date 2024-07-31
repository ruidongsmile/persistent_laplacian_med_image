# Persistent Laplacian and medical images
This is a project to apply persistent Laplacian to analyse medical images.
In this project,
we take the pneumoniamnist dataset from [MedMNIST](https://medmnist.com/) to demonstrate that the spectrum of 
persistent Laplacian associated with a [cubical complex filtration](https://gudhi.inria.fr/python/latest/cubical_complex_user.html) encode the geometric information of an image.

## Cubical complex filtration in one sentence
Roughly speaking,
for a given image,
the cubical complex filtration is a process of filtering out the image(by the the values of pixels, for instance).
We can then study about the persistent behavior of the topological structure (persistent homology),
or define a persistent Laplacian and study about the corresponding eigenvelues.

In this project,
we focus on two cases,
i.e.,
the first eigenvalue of persistent $1$-Laplacian and the smallest non-zero eigenvalue of persistent $1$-up Laplacian.

<p>
  <img src="/images/animations_10.png" width="160" />
  <img src="/images/animations_13.png" width="160" />
  <img src="/images/animations_15.png" width="160" />
  <img src="/images/animations_17.png" width="160" />
  <img src="/images/animations_20.png" width="160" />  
</p>

## The files
In this project, the files "pneum_Lap.ipynb" and "pneum_up_Lap.ipynb" exhibit the process of 
computing the first eigenvalues of persistent Laplacian and the smallest non-zero eigenvalues of 
persistent up-Laplacian respectively,
and the eigenvalues are saved in the folder "eigenvalues".
These eigenvalues are the input object for training.
The files "pneum_Lap_dpl.ipynb" and "pneum_up_Lap_dpl.ipynb" show the training process of the eigenvalues produced 
by the previous two files.
In this project we only take a simple fully connected neural network with 5 hidden layers.

# More words about cubical complex structure
We take a $ 3\times 3$



