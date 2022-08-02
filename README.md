# Smart City project - Forecasting and optimizing HVAC system parameters in a large public building.

This is repository includes Python3 scripts for smart Building Automation system development during Smart City project. The Smart City project of Häme University of Applied Sciences focuses on promoting low-carbon developments through implementation of Artificial Intelligence. To learn more about the project, read [popular](https://blog.hamk.fi/hamk-smart/koneoppiminen-alykkaissa-rakennuksissa/) and [technical](https://blog.hamk.fi/hamk-smart/alykaupunki-hanke-edistaa-tekoalyn-tuotteistamista-rakennuksissa/) blog posts from HAMK Smart blog (in Finnish).

The current branch 'scenario-testing' contains data, trained models and scripts for scenario testing modeling. It was created in order to reproduce the results of the [article](https://doi.org/10.3390/en15145084) together with confidence intervals.

See also the ['main'](https://github.com/hamk-uas/HAMK_Smart_City/tree/main) branch.

Package dependencies are listed in requirements.txt, and suitable .gitignore file is included in the repository.

### Installation
Install required python libraries by using requirements.txt:
```
pip install -r requirements.txt
```

This works on Windows at least. The version numbers for packages scipy==1.7.3 and keras_tuner==1.0.2 are important, others probably not so you can remove the other version numbers if you can't find the packages with those version numbers.

To use GPU in an Anaconda environment:

```
conda install cudnn
```

### Overview of files
* __data_example_offsets_new.csv__: Exemplary data set. Includes data for multiple HVAC parameters in addition to weather variables. The whole data set has been downsampled to hourly temporal granularity.
* __rnn.py__: Class structure for RNN model and their training. Includes methods for data preprocessing, hyperparameter optimization, model training and visualization of predictions. Saving and loading methods use folders in the the current directory to save training checkpoints and class attributes, including scaling and models objects. 
* __main.py__: Main script for training.
* __Scenario testing.ipynb__: Jupyter Notebook for scenario testing predictions.
* __predictedEnergyConsumptionScenarioDisabled.csv__: Bootstrap statistics of scenario testing results for disabled scenario testing.
* __predictedEnergyConsumptionScenarioDisabled.csv__: Bootstrap statistics of scenario testing results for enabled scenario testing.

### Trained models

__fut_0__ refers to a model that makes a prediction at the time point of the last input. __fut_1__ refers to making a prediction one hour into the future compared to the time of the last input, the same as in the [thesis](http://urn.fi/URN:NBN:fi:aalto-202202061759) and the [article](https://doi.org/10.3390/en15145084).

The best models found in hyperparameter tuning:
* __GRU_Energy_consumption_2022-06-14_hyperparameter_tuning_fut_0__
* __GRU_Energy_consumption_2022-06-15_hyperparameter_tuning_fut_1__

The above models were then retrained, resulting in these models:
* __GRU_Energy_consumption_2022-06-14_trained_fut_0__
* __GRU_Energy_consumption_2022-06-15_trained_fut_1__

### Full workflow

The following steps should be done in order.

#### Model training and bootstrap statistics for scenario testing modeling results
By enabling code blocks disabled by `if False:` in __main.py__ and by running the script each time by `python main.py`, do the following:
* Tune hyperparameters. This will save the model with the best hyperparameters. The best hyperparameters are currently fixed (in __rnn.py__) to those found when the hyperparameter optimization was done for the first time for the thesis and the article.
* Find the result folder and append "_hyperparameter_tuning_fut_0" or "_hyperparameter_tuning_fut_1" to the folder name, depending on the value of the `fut` variable in your `MyGRU` constructor call.
* Calculate statistics on scenario testing results over bootstrapped model training on input sequences resampled with replacement. These will be stored in __predictedEnergyConsumptionScenarioDisabled.csv__ and __predictedEnergyConsumptionScenarioEnabled.csv__

#### Running the models and visualizing the results
* Open __Scenario testing.ipynb__ in Jupyter Notebook.
* Make sure you have the correct model folder name in `hvac_model.load(r'GRU_Energy_consumption_2022-06-15_trained_fut_1')`.
* Run the notebook.

### Authors
2021 Iivo Metsä-Eerola and Genrikh Ekkerman<br>
2022 Olli Niemitalo

### Licence
Permissive Apache License 2.0

### Article
Metsä-Eerola, I.; Pulkkinen, J.; Niemitalo, O.; Koskela, O. On Hourly Forecasting Heating Energy Consumption of HVAC with Recurrent Neural Networks. Energies 2022, 15, 5084. https://doi.org/10.3390/en15145084
