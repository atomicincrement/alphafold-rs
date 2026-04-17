# Create a rust version of Alphafold as a learning excercise.

See

https://github.com/google-deepmind/alphafold

The goal here is not to replicate Alphafold but to create a dependency-free demonstrator
of the technology that can be understood by the lay person such as myself.

We will translate just enough of the project from Python only using ndarray for linear algebra.

Steps (to be expanded)

* Fetch a small version of the model data, if such a thing exists using reqwest.
* Decode the model data into ndarray matrices.
* Run a simple forward inference step to predict a known model.
* Build a 3D visualiser using bevy.

If small model data does not exist, we will have to train the model ourselves on
a single PDB file to see if it can predict the sequence and 3D geometry.

