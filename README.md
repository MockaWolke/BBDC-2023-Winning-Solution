# BBDC-2023-Winning-Solutions

These are our winning solutions for both tracks.

### Data Handling

Regarding the data, we discarded all dates where all values were missing, as well as every weekend, even when there were measurements on the weekends. We then interpolated the remaining missing values based on their measurement time. To address the lack of $NO_x$ data before 2004, we used Scikit-learn's Gradient Boosting Regressor to predict $NO_x$ based on every other label measurement for each timestep. This approach worked well, given that $NO_x$ and $NO_2$ have a Pearson correlation of nearly 1, and $NO_x$ is simply a combination of $NO_2$ and $NO$.

Additionally, we collected daily and hourly weather reports from Helgoland and past salinity measurements from the unmanned fire ship UFS Deutsche Bucht, which we downloaded from the Deutscher Wetterdienst's archive and marineinsitu.eu, respectively.

### Models

Initially, we attempted various models using TensorFlow, but a simple Dense Layer, i.e., a linear regression trained with SGD, performed the best. We thus concluded that the dataset contained too few data points for a deep learning approach and focused on classical machine learning techniques.

We then turned to Darts, which was extremely helpful. We experimented with several different models, starting with Elastic Net linear regression models that performed well (~0.74).

Next, we switched to the XGBoost Regressor and used Optuna's Bayesian hyperparameter optimization to find the best hyperparameters. With this model, we achieved excellent results (~0.64).

Finally, we tried LightGBM and CatBoost, also searching for the best hyperparameters and improving slightly.

### Specifics

We divided the labels into separate groups.

To predict temperature, we trained LightGBM on the temperature and SECCI values from 1968 onwards, using daily weather as covariates, and achieved excellent results (validation RMSE of 0.11).

For nitrogen and other chemicals (SiO4, PO4), we used LightGBM and the daily weather records as covariates. Interestingly, using training data only from 1990 onwards improved our performance (from 0.623 to 0.616).

Since we had difficulty predicting salinity, we searched for additional data and fortunately found the data from the UFS Deutsche Bucht, which significantly helped us (from 0.643 to 0.623).

Finally, to improve our SECCI predictions, we trained a Variational Autoencoder to represent the daily weather reports of any given day in a smaller latent dimension. We then used these representations as covariates for our model and improved our score from 0.613 to our final score of 0.609.

(All scores reference the student track.)


## Recreation of results


All you have to do to recreate our results is:


1. Get our conda env: ``` conda env create -f  environment.yml```
2. Create the folder ```mkdir data/bbdc_student``` and instert the develop & evaluations csvs aswell as json dictionaries for the means and variances for the student tracks. 
3. Create the folder ```mkdir data/bbdc_prof``` and to the excact same things only for with data for the profesional track.
4. Download same salinity measurements from the sation "Deutsche Bucht" which we collected from http://www.marineinsitu.eu/dashboard/.

 ```
 mkdir data/salt
 wget -P data/salt https://data-marineinsitu.ifremer.fr/glo_multiparameter_nrt/history/MO/NO_TS_MO_UFSDeutscheBucht.nc
 ```

5. Run both these python scripts: ```python data_preparation_profesional/quick_data_prep.py``` and ```python data_preparation_student/quick_data_prep.py``` to get our trainings represenations.

6. Downlaod hourly weather data from https://drive.google.com/file/d/1cxt6fv4ERK1UL_6Jo2jQEUt-3moPxw-U/view?usp=sharing 
This can also be done via the dwd portal. These datapoints are later compressed using an autoencoder and used to predict solely SECCI

7. Have a look at the Notebooks in ```prediction_profesional``` and ```prediction_student```.