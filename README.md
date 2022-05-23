# Smart City project - Forecasting and optimizing HVAC system parameters in a large public building.

This is repository includes Python3 scripts for smart Building Automation system development during Smart City project. The Smart City project of Häme University of Applied Sciences focuses on promoting low-carbon developments through implementation of Artificial Intelligence. To learn more about the project, read [popular](https://blog.hamk.fi/hamk-smart/koneoppiminen-alykkaissa-rakennuksissa/) and [technical](https://blog.hamk.fi/hamk-smart/alykaupunki-hanke-edistaa-tekoalyn-tuotteistamista-rakennuksissa/) blog posts from HAMK Smart blog (in Finnish). The content of this repository is focused on forecasting district heating energy consumption and optimization of HVAC system controls.

The branch 'main' includes scripts which include everything from data preprocessing to Particle Swarm Optimization of HVAC controls. Package dependencies are listed in requirements.txt, and suitable .gitignore file is included in the repository.

### Installation
Install required python libraries by using requirements.txt
```
pip install -r requirements.txt
```

### Overview of files
* __data_example.csv__: Exemplary data set. Includes data for multiple HVAC parameters in addition to weather variables. The whole data set has been downsampled to hourly temporal granularity.
* __rnn.py__: Class structure used to train RNN models. Can be used for modeling district heating energy consumption and average inside temperature. Includes methods for data preprocessing, training the model with hyperparameter optimization, calculating prediction intervals to quantify uncertainty and visualizing the predictions. Saving and loading methods use the current directory to save training checkpoints and class attributes, including scaling and models objects. 
* __optimization.py__: Particle Swarm Optimization of HVAC controls. Algorithm finds the HVAC network controls for which the energy consumption is minimal with inside temperature still being inside acceptable range. Saves control values of each iteration to disk. Single objective implementation with non-linear penalties for temperature deviations is based on Pareto optimality based multi-objective optimization presented in Wei et al. (2015) article in Energy. **This script is not functional at the moment as it has not been fitted to the class structure and main scripts.**
* __main.py__: Main script for running the analysis. Methods of the class rnn are demonstrated for the user.

### Authors
2021 Iivo Metsä-Eerola and Genrikh Ekkerman

### Licence
Permissive Apache License 2.0
