#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import json
import matplotlib as plt
# from sklearn.metrics import mean_absolute_percentage_error

# Get SUV predictions.
# Columns of interest: 'date', 'lower_predicted_deaths', 'upper_predicted_deaths', 'pre_fata', 'Region'
suv_predictions = pd.read_csv("20210429_Shad_pred_state_END_DATE_2021-03-20_PRED_START_DATE_2021-04-03.csv").rename(columns={"Date":"date", "Region":"location_name", 'lower_pre_fata':'SUV_lower_predicted_deaths', 'upper_pre_fata':'SUV_upper_predicted_deaths', 'pre_fata':'SUV_predicted_deaths'})
suv_predictions = suv_predictions.loc[:,['date', 'SUV_lower_predicted_deaths', 'SUV_upper_predicted_deaths', 'SUV_predicted_deaths', 'location_name']]

# Get SuEIR prediction.
sueir_predictions = pd.read_csv("./pred_results_state/predSuEIR_state_END_DATE_2021-03-20_PRED_START_DATE_2021-04-03.csv").rename(columns={"Date":"date", "Region":"location_name", 'lower_pre_fata':'sueir_lower_predicted_deaths', 'upper_pre_fata':'sueir_upper_predicted_deaths', 'pre_fata':'sueir_predicted_deaths'})
sueir_predictions = sueir_predictions.loc[:,['date', 'sueir_lower_predicted_deaths', 'sueir_upper_predicted_deaths', 'sueir_predicted_deaths', 'location_name']]

# Get IHME predictions.
# Columns of interest: 'location_name', 'date', 'totdea_mean_smoothed', 'totdea_lower_smoothed', 'totdea_upper_smoothed'
ihme = pd.read_csv("20210429_IHME.csv", low_memory=False).rename(columns={'totdea_lower_smoothed':'ihme_lower_predicted_deaths', 'totdea_upper_smoothed':'ihme_upper_predicted_deaths', 'totdea_mean_smoothed':'ihme_predicted_deaths'})
ihme_predictions = ihme[ihme.location_name == "California"].loc[:,['location_name', 'date', 'ihme_lower_predicted_deaths', 'ihme_upper_predicted_deaths', 'ihme_predicted_deaths']]

# Get actual dates and deaths by sate
actual_state = pd.read_csv("./fetchedData/us-states.csv")
actual_state = actual_state[actual_state.state == "California"].rename(columns={'state':'location_name'})

predictions_merged = pd.merge(suv_predictions, sueir_predictions).merge(ihme_predictions).merge(actual_state, how = "outer").rename(columns = {"deaths":"reported_deaths"})
predictions_merged['date'] = pd.to_datetime(predictions_merged['date'])
predictions_merged.plot(x='date', y=["SUV_predicted_deaths", "sueir_predicted_deaths", "ihme_predicted_deaths", "reported_deaths"])

predictions_march = predictions_merged[(predictions_merged.date > "2021-03-01 00:00:00") & (predictions_merged.date < "2021-05-30 00:00:00")]
# predictions_march = predictions_merged[(predictions_merged.date > "2021-04-01 00:00:00")]
predictions_march.plot(x='date', y=["SUV_predicted_deaths", "sueir_predicted_deaths", "ihme_predicted_deaths", "reported_deaths"])

predictions_march.to_csv("./march_predictions_output.csv")