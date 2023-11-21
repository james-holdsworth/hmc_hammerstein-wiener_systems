# hmc_hammerstein-wiener_systems
To run the code:
* Install  Python 3.7< & 3.11> 
* Create a virtual environment in the root directory (the directory of this file). 
* Install the required packages using `pip install -r requirements.txt`. 
* Run `MIMO Model/hammerstein-wiener.py`. Note that you can change the output directory at the top of `__main__`.

The runs that are used in the paper are stored using Git LFS in `MIMO Model/run 1` and `MIMO Model/run 2` as pickled stanfit objects. The plots can be recreated using `plot_figs.py` and changing the run variable to select a run.
