# Persistent Laplacian and medical images
This is a project to apply persistent Laplacian to analyse medical images.
In this project,
we take the pneumoniamnist dataset from [MedMNIST](https://medmnist.com/) to demonstrate that the spectrum of 
persistent Laplacian associated with a [cubical complex filtration](https://gudhi.inria.fr/python/latest/cubical_complex_user.html) encode the geometric information of an image.

Roughly speaking,
for a given image,
the process is to first filter out the image according to the values of pixels and then studying about the persistent behavior of the 
eigenvalues of the persistent Laplacian.

In this project,
we focus on two cases,
one is the first eigenvalue of persistent $1$-Laplacian 

<p>
  <img src="/images/animations_10.png" width="160" />
  <img src="/images/animations_13.png" width="160" />
  <img src="/images/animations_15.png" width="160" />
  <img src="/images/animations_17.png" width="160" />
  <img src="/images/animations_20.png" width="160" />  
</p>



