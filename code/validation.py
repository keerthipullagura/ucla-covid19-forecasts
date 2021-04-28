import numpy as np
import pandas as pd
from datetime import timedelta, datetime
import json
import argparse
import us

from model import *
from data import *
from rolling_train_modified import *
from util import *
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='validation of prediction performance for all states')
parser.add_argument('--END_DATE', default = "default",
                    help='end date for training models')
parser.add_argument('--VAL_END_DATE', default = "default",
                    help='end date for training models')
parser.add_argument('--level', default = "state",
                    help='state, county')
parser.add_argument('--state', default = "default",
                    help='state')
#parser.add_argument('--nation', default = "default",
#                    help='nation')
parser.add_argument('--county', default = "default",
                    help='county')
parser.add_argument('--dataset', default = "NYtimes",
                    help='nytimes')
parser.add_argument('--popin', type=float, default = 0,
                    help='popin')
args = parser.parse_args()

print(args)
START_nation = {"US": "2020-03-22"}


FR_nation = {"US": [0.75, 0.02]}

decay_state = {"Pennsylvania": [0.7, 0.024], "New York": [0.7, 0.042], "Illinois": [0.7, 0.035], "California": [0.5,0.016], "Massachusetts": [0.7,0.026], "New Jersey": [0.7,0.03], \
"Michigan": [0.8,0.035], "Virginia": [0.7,0.034], "Maryland": [0.7,0.024], "Washington": [0.7,0.036], "North Carolina": [0.7,0.018], "Wisconsin": [0.7,0.034], "Texas": [0.3,0.016], \
"New Mexico": [0.7,0.02], "Louisiana": [0.4,0.02], "Arkansas": [0.7,0.02], "Delaware": [0.7,0.03], "Georgia": [0.7,0.015], "Arizona": [0.7,0.02], "Connecticut": [0.7,0.026], "Ohio": [0.7,0.024], \
"Kentucky": [0.7,0.023], "Kansas": [0.7,0.02], "New Hampshire": [0.7,0.014], "Alabama": [0.7,0.024], "Indiana": [0.7,0.03], "South Carolina": [0.7,0.02], "Colorado": [0.7,0.02], "Florida": [0.4,0.016], \
"West Virginia": [0.7,0.022], "Oklahoma": [0.7,0.03], "Mississippi": [0.7,0.026], "Missouri": [0.7,0.02], "Utah": [0.7,0.018], "Alaska": [0.7,0.04], "Hawaii": [0.7,0.04], "Wyoming": [0.7,0.04], "Maine": [0.7,0.025], \
"District of Columbia": [0.7,0.024], "Tennessee": [0.7,0.027], "Idaho": [0.7,0.02], "Oregon": [0.7,0.036], "Rhode Island": [0.7,0.024], "Nevada": [0.5,0.022], "Iowa": [0.7,0.02], "Minnesota": [0.7,0.025], \
"Nebraska": [0.7,0.02], "Montana": [0.5,0.02]}

mid_dates_state = {"Alabama": "2020-06-03", "Arizona": "2020-05-28", "Arkansas": "2020-05-11", "California": "2020-05-30", "Georgia": "2020-06-05",
 "Nevada": "2020-06-01", "Oklahoma": "2020-05-31", "Oregon": "2020-05-29", "Texas": "2020-06-15", "Ohio": "2020-06-09",
     "West Virginia": "2020-06-08", "Florida": "2020-06-01", "South Carolina": "2020-05-25", "Utah": "2020-05-28", "Iowa": "2020-06-20", "Idaho": "2020-06-15",
     "Montana": "2020-06-15", "Minnesota": "2020-06-20", "Illinois": "2020-06-30", "New Jersey": "2020-06-30", "North Carolina": "2020-06-20" , "Maryland":  "2020-06-25",
     "Kentucky": "2020-06-30", "Pennsylvania": "2020-07-01", "Colorado": "2020-06-20", "New York": "2020-06-30", "Alaska": "2020-06-30", "Washington": "2020-06-01"
}
mid_dates_state_resurge = {"Colorado": "2020-09-10", "California": "2020-09-30", "Florida": "2020-09-20", "Illinois": "2020-09-10", "New York": "2020-09-10", "Texas": "2020-09-15"
}

mid_dates_county = {"San Joaquin": "2020-05-26", "Contra Costa": "2020-06-02", "Alameda": "2020-06-03", "Kern": "2020-05-20", \
 "Tulare": "2020-05-30", "Sacramento": "2020-06-02", "Fresno": "2020-06-07", "San Bernardino": "2020-05-25", \
 "Los Angeles": "2020-06-05", "Santa Clara": "2020-05-29", "Orange": "2020-06-12", "Riverside": "2020-05-26", "San Diego": "2020-06-02" \
}

mid_dates_nation = {"US": "2020-06-15"}

north_cal = ["Santa Clara", "San Mateo", "Alameda", "Contra Costa", "Sacramento", "San Joaquin", "Fresno"]



def validation(model, init, params_all, train_data, val_data,  new_sus, pop_in):
    val_data_confirm, val_data_fatality = val_data[0], val_data[1]
    val_size = len(val_data_confirm)

    pred_confirm, pred_fatality, _ = rolling_prediction(model, init, params_all, train_data, new_sus, pred_range=val_size, pop_in=pop_in)

    return  0.5*loss(pred_confirm, val_data_confirm, smoothing=0.1) + loss(pred_fatality, val_data_fatality, smoothing=0.1)

def get_county_list_for_state(cc_limit=200, pop_limit=50000, data=None, County_Pop=None):
    non_county_list = ["Puerto Rico", "American Samoa", "Guam", "Northern Mariana Islands", "Virgin Islands", "Diamond Princess", "Grand Princess"]

    county_list = []
    for region in County_Pop.keys():
        county, state = region.split("_")
        if County_Pop[region][0]>=pop_limit and not state in non_county_list:
            train_data = data.get("2020-03-22", args.END_DATE, state, county)
            confirm, death = train_data[0], train_data[1]
            start_date = get_start_date(train_data)
            if len(death) >0 and np.max(death)>=0 and np.max(confirm)>cc_limit and start_date < "2020-05-10" and not county=="Lassen":
                county_list += [region]

    return county_list


if __name__ == '__main__':
    
    
    # initial the dataloader, get region list 
    # get the directory of output validation files
    if args.level == "state":
        data = NYTimes(level='states')
        nonstate_list = ["American Samoa", "Diamond Princess", "Grand Princess", "Virgin Islands", "Northern Mariana Islands"]
        region_list = [state for state in data.state_list if not state in nonstate_list]

        mid_dates = mid_dates_state
        write_dir = "val_results_state/" + args.dataset + "_" 
        if not args.state == "default":
            region_list = [args.state]  
            write_dir = "val_results_state/test" + args.dataset + "_"
        
    elif args.level == "county":
        state = args.state
        data = NYTimes(level='counties')
        mid_dates = mid_dates_county
        with open("data/county_pop.json", 'r') as f:
            County_Pop = json.load(f)

        if not args.state == "default" and not args.county == "default":
            region_list = [args.county + "_" + args.state]
            write_dir = "val_results_county/test" + args.dataset + "_"
        else:
            region_list = get_county_list_for_state(cc_limit=2000, pop_limit=10, data=data, County_Pop=County_Pop)
            print("# feasible counties:", len(region_list))
            write_dir = "val_results_county/" + args.dataset + "_" 

    params_allregion = {}
   

    for region in region_list:

        # generate training data, validation data
        # get the population
        # get the start date, and second start date
        # get the parameters a and decay
        
        if args.level == "state":
            state = str(region)
            # can change to pyspark df.
            df_Population = pd.read_csv('data/us_population.csv')
            print(state)
            # change to pyspark df.
            Pop=df_Population[df_Population['STATE']==state]["Population"].to_numpy()[0]
            start_date = START_nation['US']
            #start_date = get_start_date(data.get("2020-03-22", args.END_DATE, state),100)
            if state in mid_dates.keys():
                second_start_date = mid_dates[state]
                train_data = [data.get(start_date, second_start_date, state), data.get(second_start_date, args.END_DATE, state)]
                reopen_flag = True
            else:
                second_start_date = "2020-08-30"
                train_data = [data.get(start_date, second_start_date, state), data.get(second_start_date, args.END_DATE, state)]
                reopen_flag = False

            if state in mid_dates.keys():
                resurge_start_date = mid_dates_state_resurge[state] if state in mid_dates_state_resurge.keys() else "2020-09-15"
                train_data = [data.get(start_date, second_start_date, state), data.get(second_start_date, resurge_start_date, state), \
                 data.get(resurge_start_date, args.END_DATE, state)]
                full_data = [data.get(start_date, second_start_date, state), data.get(second_start_date, resurge_start_date, state), \
                 data.get(resurge_start_date, args.VAL_END_DATE, state)]


            val_data = data.get(args.END_DATE, args.VAL_END_DATE, state)
            if state in decay_state.keys():
                a, decay = decay_state[state][0], decay_state[state][1]
            else:
                a, decay = 0.7, 0.3          
            # will rewrite it using json
            pop_in = 1/400
            if state == "California":
                pop_in = 0.01
        elif args.level == "county":
            county, state = region.split("_")
            region = county + ", " + state
            key = county + "_" + state

            Pop=County_Pop[key][0]
            start_date = START_nation['US']
            #start_date = get_start_date(data.get("2020-03-22", args.END_DATE, state, county))
            if state=="California" and county in mid_dates.keys():
                second_start_date = mid_dates[county]
                reopen_flag = True
            else:
                second_start_date = "2020-08-30"
                reopen_flag = False

            if start_date < "2020-05-10":
                train_data = [data.get(start_date, second_start_date, state, county), data.get(second_start_date, args.END_DATE, state, county)]
            else:
                train_data = [data.get(start_date, args.END_DATE, state, county)]
            val_data = data.get(args.END_DATE, args.VAL_END_DATE, state, county)
            if state in decay_state.keys():
                a, decay = decay_state[state][0], decay_state[state][1]
            else:
                a, decay = 0.7, 0.32
            if county in north_cal and state=="California":
                decay = 0.03
            pop_in = 1/400

            if state in mid_dates_state.keys():
                resurge_start_date = mid_dates_state_resurge[state] if state in mid_dates_state_resurge.keys() else "2020-09-15"
                train_data = [data.get(start_date, second_start_date, state, county), data.get(second_start_date, resurge_start_date, state, county), \
                 data.get(resurge_start_date, args.END_DATE, state, county)]
                full_data = [data.get(start_date, second_start_date, state, county), data.get(second_start_date, resurge_start_date, state, county), \
                 data.get(resurge_start_date, args.VAL_END_DATE, state, county)]

        print(len(train_data))
        mean_increase = 0
        if len(train_data)>1:
            last_confirm, last_fatality = train_data[-1][0], train_data[-1][1]
            daily_confirm = np.diff(last_confirm)
            mean_increase = np.median(daily_confirm[-7:] - daily_confirm[-14:-7])/2 + np.median(daily_confirm[-14:-7] - daily_confirm[-21:-14])/2
            # if mean_increase<1.1:
            #     pop_in = 1/5000
            if not reopen_flag or args.level == "county":
                if np.mean(daily_confirm[-7:])<12.5 or mean_increase<1.1:
                    pop_in = 1/5000
                elif mean_increase < np.mean(daily_confirm[-7:])/40:
                    pop_in = 1/5000
                elif mean_increase > np.mean(daily_confirm[-7:])/10 and np.mean(daily_confirm[-7:])>60:
                    pop_in = 1/500
                else:
                    pop_in = 1/1000
            if args.level=="state" and reopen_flag and (np.mean(daily_confirm[-7:])<12.5 or mean_increase<1.1):
                pop_in = 1/500
                if state == "California":
                    pop_in = 0.01
            if args.popin >0:
                pop_in = args.popin

        print("region: ", region, " start date: ", start_date, " mid date: ", second_start_date,
            " end date: ", args.END_DATE, " Validation end date: ", args.VAL_END_DATE, "mean increase: ", mean_increase, pop_in )    

        # print(train_data)
        # candidate choices of N and E_0, here r = N/E_0
        Ns = np.asarray([0.2])*Pop
        rs = np.asarray([30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 120, 150, 200, 400])
        if args.level == "county":
            rs = np.asarray([30,  40, 50, 60, 70, 80,  90, 100, 120, 150, 200, 400])

        A_inv, I_inv, R_inv, loss_list0, loss_list1, params_list, learner_list, I_list = [],[],[],[],[],[],[],[]
            
        val_log = []
        min_val_loss = 10 #used for finding the minimum validation loss
        for N in Ns:
            for r in rs:
                E_0 = N/r

                # In order to simulate the reopen, we assume at the second stage, there are N new suspectible individuals
                new_sus = 0 if reopen_flag else 0
                if args.level == "state" or args.level == "county":
                    bias = 0.025 if reopen_flag or (state=="Louisiana" or state=="Washington" or state == "North Carolina" or state == "Mississippi") else 0.005
                    if state == "Arizona" or state == "Alabama" or state == "Florida" or state=="Indiana" or state=="Wisconsin" or state == "Hawaii" or state == "California" or state=="Texas" or state=="Illinois":
                        bias = 0.025 if reopen_flag else 0.005
                    if state == "California":
                        bias = 0.01
                    if state == "Arkansas" or state == "Iowa" or state == "Minnesota" or state == "Louisiana" \
                     or state == "Nevada" or state == "Kansas" or state=="Kentucky" or state == "Tennessee" or state == "West Virginia":
                        bias = 0.05
                data_confirm, data_fatality = train_data[0][0], train_data[0][1]
                # print (bias)
                model = Learner_SuEIR(N=N, E_0=E_0, I_0=data_confirm[0], R_0=data_fatality[0], a=a, decay=decay, bias=bias)

                # At the initialization we assume that there is not recovered cases.
                init = [N-E_0-data_confirm[0]-data_fatality[0], E_0, data_confirm[0], data_fatality[0]]
                # train the model using the candidate N and E_0, then compute the validation loss
                params_all, loss_all = rolling_train(model, init, train_data, new_sus, pop_in=pop_in)
                val_loss = validation(model, init, params_all, train_data, val_data, new_sus, pop_in=pop_in)

                for params in params_all:
                    beta, gamma, sigma, mu = params
                    # we cannot allow mu>sigma otherwise the model is not valid
                    if mu>sigma:
                        val_loss = 1e6


                # using the model to forecast the fatality and confirmed cases in the next 100 days, 
                # output max_daily, last confirm and last fatality for validation
                pred_confirm, pred_fatality, _ = rolling_prediction(model, init, params_all, train_data, new_sus, pop_in=pop_in, pred_range=100, daily_smooth=True)
                max_daily_confirm = np.max(np.diff(pred_confirm))
                pred_confirm_last, pred_fatality_last = pred_confirm[-1], pred_fatality[-1]
                # print(np.diff(pred_fatality))
                # print(sigma/mu)
                #prevent the model from explosion
                if pred_confirm_last >  8*train_data[-1][0][-1] or  np.diff(pred_confirm)[-1]>=np.diff(pred_confirm)[-2]:
                    val_loss = 1e8

                # record the information for validation
                val_log += [[N, E_0] + [val_loss] + [pred_confirm_last] + [pred_fatality_last] + [max_daily_confirm] + loss_all  ]

                # plot the daily inc confirm cases
                confirm = train_data[0][0][0:-1].tolist() + train_data[-1][0][0:-1].tolist() + pred_confirm.tolist()
                true_confirm =  train_data[0][0][0:-1].tolist() + train_data[-1][0][0:-1].tolist() + val_data[0][0:-1].tolist()

                deaths = train_data[0][1][0:-1].tolist() + train_data[-1][1][0:-1].tolist() + pred_fatality.tolist()
                true_deaths =  train_data[0][1][0:-1].tolist() + train_data[-1][1][0:-1].tolist() + val_data[1][0:-1].tolist()
                # can remove all Remove all this
                min_val_loss = np.minimum(val_loss, min_val_loss)
                # print(val_loss)

        params_allregion[region] = val_log
        print (np.asarray(val_log))
        best_log = np.array(val_log)[np.argmin(np.array(val_log)[:,2]),:]
        print("Best Val loss: ", best_log[2], " Last CC: ", best_log[3], " Last FC: ", best_log[4], " Max inc Confirm: ", best_log[5] )

    # write all validation results into files
    write_file_name_all = write_dir + "val_params_" + "END_DATE_" + args.END_DATE + "_VAL_END_DATE_" + args.VAL_END_DATE
    write_file_name_best = write_dir + "val_params_best_" + "END_DATE_" + args.END_DATE + "_VAL_END_DATE_" + args.VAL_END_DATE

    write_val_to_json(params_allregion, write_file_name_all, write_file_name_best)
        