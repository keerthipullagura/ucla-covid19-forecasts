
# SuEIRV model for forecasting the COVID-19 related confirmed cases, deaths, and vaccinations.

## Vaccination
to include the latest vaccination data in this model, run the following script first

```
cd vaccination
python vaccination.py
```
The vaccination predictions based LinearRegression for every state are generated under /vaccination/predictions folder. Plots for every state which depict the vaccination rate over time is generated under /vaccination/plots

## SuEIRV model for forecasting confirmed cases, deaths, and vaccinations state level.

### How to get forecast results of confirmed cases, deaths at different levels?

Step 1: Go to directory spark_code and Run ```validation.py``` to generate validation file for selecting hyperparameters, e.g.,
```python
python validation.py --END_DATE 2020-07-07 --VAL_END_DATE 2020-07-14  --dataset NYtimes --level state
```
Step 2: The results are generated under folder /spark_code/val_results_state
Step 3: Go to directory spark_code and to Generate prediction results by running ```generate_predictions.py```, e.g.,
```python
python generate_predictions.py --END_DATE 2020-07-07 --VAL_END_DATE 2020-07-14 --dataset NYtimes --level state
```
Step 4: The results are generated under folder /spark_code/pred_results_state

Before runing ```generate_predictions.py```, one should make sure the corresponding validation file, i.e., with the same ```END_DATE```, ```VAL_END_DATE```, ```dataset```, and ```level```, has already be generated.


### Arguments:
*```END_DATE```: end date for training data

*```VAL_END_DATE```: end date for validation data

*```level```: can be state default: state

*```state```: validation/prediction for one specific state (```level``` should be set as state), default: all states in the US 

*```dataset```: select which data source to use. We have used NYtimes, default: NYtimes data

### Notice:
We consider two-stage training (sequentially training over two periods of data, determined by the ```mid_date``` variable determined in the code) for validation and generating predictions. Additionlly, the end date of training data ```END_DATE``` should guarantee that the length of the second period of data should be greater than 21, i.e., the length between ```mid_date``` and ```END_DATE``` should be greater than 21. For example, currently the ```mid_date``` for CA is 2020-06-07, then the ```END_DATE``` should be set at least after 2020-06-28.


## Reference
SuEIR referenced from https://github.com/uclaml/ucla-covid19-forecasts 
UCLA forecasts at county, state, and nation levels in their website [covid19.uclaml.org](https://covid19.uclaml.org/).


