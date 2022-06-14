# Smart City project - Forecasting and optimizing HVAC system parameters in a large public building.

This is repository includes Python3 scripts for smart Building Automation system development during Smart City project. The Smart City project of Häme University of Applied Sciences focuses on promoting low-carbon developments through implementation of Artificial Intelligence. To learn more about the project, read [popular](https://blog.hamk.fi/hamk-smart/koneoppiminen-alykkaissa-rakennuksissa/) and [technical](https://blog.hamk.fi/hamk-smart/alykaupunki-hanke-edistaa-tekoalyn-tuotteistamista-rakennuksissa/) blog posts from HAMK Smart blog (in Finnish).

For HVAC optimization, see the main branch. Files related to optimization have been removed from this branch.

Package dependencies are listed in requirements.txt, and suitable .gitignore file is included in the repository.

### Installation
Install required python libraries by using requirements.txt
```
pip install -r requirements.txt
```

To use GPU in an Anaconda environment:

```
conda install cudnn
```

### Overview of files
* __data_example_offsets_new.csv__: Exemplary data set. Includes data for multiple HVAC parameters in addition to weather variables. The whole data set has been downsampled to hourly temporal granularity.
* __rnn.py__: Class structure used to train RNN models. Includes methods for data preprocessing, hyperparameter optimization, model training and visualizing the predictions. Saving and loading methods use folders in the the current directory to save training checkpoints and class attributes, including scaling and models objects. 
* __main.py__: Main script for training.
* __Scenario testing.ipynb__: Jupyter Notebook for scenario testing predictions.

### Authors
2021 Iivo Metsä-Eerola and Genrikh Ekkerman
2022 Olli Niemitalo

### Licence
Permissive Apache License 2.0
