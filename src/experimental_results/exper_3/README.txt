Feb. 2017
A.Lons

After getting my code to work faster, see README in exper_2_fixing, I am here going to run some more appropriate
experiments using TF r1.0, on the Titan X Pascel, using a Res-net sized to handle 128x128, 256x256 and 512x512 inputs.
Also unlike what I did in experiment 2 (exper_2) I am going to use the optimizer suggested by the original authors.
That is simple gradient-descent with momentum and a learning rate decay.

Also I am going make a specific input-pipeline so that I do image flipping at random in the pipeline itself.