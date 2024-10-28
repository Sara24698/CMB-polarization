README file for making simulations (by L. Bonavera).

+++++ Move the folder Make_simulations_**** to your home and run and write everything in your account because it is faster than writing on the external HD. Once you have the final simulations you'll have to move them to the server for the NN and probably on the external HD of this PC.

+++++ Requirement: you have to install astropy, h5py, heaply and retroject. I usually create and environment with this stuff by running for example:
conda create -n HPRenv python
conda install -n HPRenv ipython astropy healpy h5py reproject

And then activate the environment with:
conda activate HPRenv 

+++++ To run the simulations first prepare the maps you need. 

If they are in T, Q and U you'll probably need to extract them with the function Extract_PlanckMap in prepare_maps.py. The simulator require the maps already extracted.
If you simulate just the map you need, you don't have to do this step. 

Put the maps you need for the simulations in the data folder and write the corresponding name in the parameter file cfreq***.par. 

The "n_sims" parameter should be set to the desired number of simulations. We usually produce 15000 for the train_dataset, 3000 for the test_dataset and 1000 for the validation_dataset.
The other parameters in the parfile should be already set to right value by me, revise them and ask if you have doubts.

+++++ Once you have everything set, you can run the simulations with simulate4***.py
If you open the simulate4***.py, in the defmain() you have an example how to run it. 
Basically, you have to repeat the process for each set of simulations. Remember to change the output name to train, test and validation dataset when you change the number of simulations.
I suggest to run first for a few simulations, e.g. "n_sims": 2 and check the output.

