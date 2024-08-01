# Persistent Laplacian and medical images
This is a project to apply [persistent Laplacian](https://epubs.siam.org/doi/10.1137/21M1435471) to analyse medical images.
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
the eigenvalues of persistent $1$-Laplacian and that of persistent $1$-up Laplacian.

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
We take a $3\times 3$ pixel image as an example to explain the basic idea of cubical complex filtration.
There are two ways to construct cubical complexes. 

One is to regard each pixel as a vertex,
which is also referred to as a $0$-dim cube.
We call an edge between two adjacent vertexes a $1$-dim cube,
the value of an edge is set to be the minimal value of the two adjacent vertexes,
and we call an square between four adjacent vertexes a $2$-dim cube,
the value of a square is set to be the minimal value of the four adjacent vertexes.
We call the collection of all the cubes as a cubical complex,
and we denote it as $L$,
with the $0$-cubes, $1$-cubes and $2$-cubes as:
$$C_0^L = \\{(1), (2), (3), (4), (5), (6), (7), (8), (9)\\},$$
$$C_1^L = \\{(12), (23), (41), (52), (63), (45), (56), (74), (85), (96), (78), (89)\\},$$
$$C_2^L = \\{(4512), (5623), (7845), (8956)\\},$$
<p>
  <img src="/images/pic4.png" width="280" />
</p>


Then we set a filtration threshold,
keep only all the $1$-cubes with value less than or equal to $0.5$ and all the $0$-cubes (or vertexes),
which is denoted as $K$,
with:
$$C_0^{K} = \\{(1), (2), (3), (4), (5), (6), (7), (8), (9)\\},$$
$$C_1^{K} = \\{(12), (23), (41), (63), (45), (56), (74), (85), (96), (78), (89)\\}.$$
<p>
  <img src="/images/pic6.png" width="280" />
</p>

There is a linear map (always referred to as a boundary map) 
$\partial_2^{K, L}: C_2^L \to C_1^{K}$,
and a boundary map $\partial_1^K: C_1^K\to C_0^K$.

The persistent $1$-up-Laplacian $\triangle_{1, \mathrm{up}}^{K, L}: C_1^K\to C_1^K$ is defined as 

$$\triangle_{1, \mathrm{up}}^{K, L}:=\partial_{2}^{K, L}\cdot \left(\partial_{2}^{K, L}\right)^*,$$

and the persistent $1$-Laplacian $\triangle_1^{K, L}: C_1^K\to C_1^K$ is defined as 

$$\triangle_1^{K, L}:=\partial_{2}^{K, L}\cdot \left(\partial_{2}^{K, L}\right)^* + \left(\partial_{1}^K\right)^*\cdot\partial_{1}^K.$$

The other way is to regard each pixel as a square in a cubical complex,
and the value of edges and vertexes is taken to be the smallest value of the adjacent squares.
The process is similar to the previous case, 
thus we skip the details.
## Construct a cubical complex from an image 
Each image in the pneumoniamnist dataset is an $28\times 28$ image.
Take one image,
we construct the corresponding cubical complex $L$ by regarding each pixel as a square,
as a result,
such a cubical complex contains $841$ vertexes, $1624$ edges and $784$ squares.
On the other hand,
the pixels range from $0$ to $1$,
we evenly divide the interval $[0, 1]$ into $14$ intervals $0=t_0 \< t_1 \< \cdots \< t_{14}=1$,
let $L_{i}$ denote the sub-cubical complex with cubes whose values are less than $t_{i}$ for $1\\leq i\\leq 14$.
### Persistent $1$-Laplacian
We first compute the $14$ persistent $1$-Laplacians: $\triangle_{1}^{L_1, L_2}$, $\triangle_{1}^{L_2, L_3}$, ..., $\triangle_{1}^{L_{13}, L_{14}}$,
for each one we compute the smallest $3$ eigenvalues,
as a result we produce a $14\times 3$ tensor from each image, 
that is what the file "pneum_Lap.ipynb" is doing.
### Persistent $1$-up-Laplacian
We then compute the $14$ persistent $1$-up-Laplacians: $\triangle_{1, \mathrm{up}}^{L_1, L_2}$, $\triangle_{1, \mathrm{up}}^{L_2, L_3}$,
, ..., $\triangle_{1, \mathrm{up}}^{L_{13}, L_{14}}$,
for each one we compute the smallest non-zero eigenvalue,
as a result we come up with a $14\times 1$ tensor for each image,
that is what the file "pneum_up_Lap.ipynb" is doing.

# Remark
This project is mainly an experiment to show that the technique of persistent Laplacian can be applicable in medical imaging.
The algorithm is cited from [M\'{e}moli](https://epubs.siam.org/doi/10.1137/21M1435471)







