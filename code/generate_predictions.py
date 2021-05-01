import argparse
import os

from rolling_train_modified import *
from util import *
from datetime import timedelta, datetime

parser = argparse.ArgumentParser(description='validation of prediction performance for all states')
parser.add_argument('--END_DATE', default = "default",
                    help='end date for training models')
parser.add_argument('--VAL_END_DATE', default = "default",
                    help='end date for training models')
parser.add_argument('--level', default = "state",
                    help='state, nation or county')
parser.add_argument('--state', default = "default",
                    help='state')
#parser.add_argument('--nation', default = "default",
#                    help='nation')
parser.add_argument('--dataset', default = "NYtimes",
                    help='nytimes')
parser.add_argument('--popin', type=float, default = 0,
                    help='popin')
args = parser.parse_args()
PRED_START_DATE = args.VAL_END_DATE

print(args)

#team adding vaccination data, will need to update for output of predictions
vaccines = pd.read_csv("california_vaccinations.csv")
first_vaccine_date = list(vaccines.date)[0]
vaccines = list(vaccines.daily_vaccinations)



START_nation = {"US": "2020-03-22"}


FR_nation = {"US": [0.75, 0.02]}

#Adding back decay rates for all states in US.
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


if args.level == "state":
    data = NYTimes(level='states')
    nonstate_list = ["American Samoa", "Diamond Princess", "Grand Princess", "Virgin Islands"]
    mid_dates = mid_dates_state
    val_dir = "val_results_state/"
    pred_dir = "pred_results_state/"
    if not args.state == "default":
        region_list = [args.state]
        region_list = ["New York", "California"]
        val_dir = "val_results_state/test"

elif args.level == "county":
    state = args.state
    data = NYTimes(level='counties')
    mid_dates = mid_dates_county
    val_dir = "val_results_county/"
    pred_dir = "pred_results_county/"

json_file_name = val_dir + args.dataset + "_" + "val_params_best_END_DATE_" + args.END_DATE + "_VAL_END_DATE_" + args.VAL_END_DATE
if not os.path.exists(json_file_name):
    json_file_name = val_dir + "JHU" + "_" + "val_params_best_END_DATE_" + args.END_DATE + "_VAL_END_DATE_" + args.VAL_END_DATE




with open(json_file_name, 'r') as f:
    NE0_region = json.load(f)

prediction_range = 100
frame = []
region_list = list(NE0_region.keys())
region_list = [region for region in region_list if not region == "Independence, Arkansas"]
# region_list = ["France"]
for region in region_list:

    if args.level == "state":
        state = str(region)
        start_date = START_nation['US']
        mid_dates = mid_dates_state
        if state in mid_dates.keys():
            second_start_date = mid_dates[state]
            reopen_flag = True
        else:
            second_start_date = "2020-08-30"
            reopen_flag = False

        train_data = [data.get(start_date, second_start_date, state), data.get(second_start_date, args.END_DATE, state)]
        full_data = [data.get(start_date, second_start_date, state), data.get(second_start_date, PRED_START_DATE, state)]

        if state in mid_dates.keys():
            resurge_start_date = mid_dates_state_resurge[state] if state in mid_dates_state_resurge.keys() else "2020-09-15"
            train_data = [data.get(start_date, second_start_date, state), data.get(second_start_date, resurge_start_date, state), \
             data.get(resurge_start_date, args.END_DATE, state)]
            full_data = [data.get(start_date, second_start_date, state), data.get(second_start_date, resurge_start_date, state), \
             data.get(resurge_start_date, PRED_START_DATE, state)]



        #team added finding time between last section of dates starting and vaccines starting
        try:
            print("start date",start_date)
            last_start_day =start_date
        except:
            print("no start date")
        try:
            print("second_start_date",second_start_date)
            last_start_day = second_start_date
        except:
            print(" no second start date")
        try:
            print("resurge_start_date",resurge_start_date)
            last_start_day = resurge_start_date
        except:
            print("no resurge date")
        from datetime import datetime

        date_difference = (datetime.strptime(first_vaccine_date,"%Y-%m-%d")- datetime.strptime(last_start_day,"%Y-%m-%d")).days
        print(date_difference)





        if state in decay_state.keys():
            a, decay = decay_state[state][0], decay_state[state][1]
        else:
            a, decay = 0.7, 0.3

        # json_file_name = "val_results_state/" + args.dataset + "_val_params_best_END_DATE_" + args.END_DATE + "_VAL_END_DATE_" + args.VAL_END_DATE
        # with open(json_file_name, 'r') as f:
        #     NE0_region = json.load(f)
        pop_in = 1/400
        # will rewrite it using json

    elif args.level == "county":
        county, state = region.split(", ")
        region = county + ", " + state
        key = county + "_" + state
        start_date = START_nation['US']
        #start_date = get_start_date(data.get("2020-03-22", args.END_DATE, state, county))

        if state=="California" and county in mid_dates.keys():
            second_start_date = mid_dates[county]
            reopen_flag = True
        elif state in mid_dates_state.keys():
            second_start_date = mid_dates_state[state]
            reopen_flag = True
        else:
            second_start_date = "2020-08-30"
            reopen_flag = False

        train_data = [data.get(start_date, second_start_date, state, county), data.get(second_start_date, args.END_DATE, state, county)]
        full_data = [data.get(start_date, second_start_date, state, county), data.get(second_start_date, PRED_START_DATE, state, county)]

        # train_data = [data.get(start_date, second_start_date, state), data.get(second_start_date, args.END_DATE, state)]
        # full_data = [data.get(start_date, second_start_date, state), data.get(second_start_date, PRED_START_DATE, state)]

        if state in mid_dates_state.keys():
            resurge_start_date = mid_dates_state_resurge[state] if state in mid_dates_state_resurge.keys() else "2020-09-15"
            train_data = [data.get(start_date, second_start_date, state, county), data.get(second_start_date, resurge_start_date, state, county), \
             data.get(resurge_start_date, args.END_DATE, state, county)]
            full_data = [data.get(start_date, second_start_date, state, county), data.get(second_start_date, resurge_start_date, state, county), \
             data.get(resurge_start_date, PRED_START_DATE, state, county)]

        if state in decay_state.keys():
            a, decay = decay_state[state][0], decay_state[state][1]
        else:
            a, decay = 0.7, 0.32
        pop_in = 1/400


    # determine the parameters including pop_in, N and E_0
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
    N, E_0 = NE0_region[region][0], NE0_region[region][1]
    # print (N, E_0)
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


    #team changed call to Sueir model to reflect new inputs
    model = Learner_SuEIR(N, E_0=E_0, I_0=data_confirm[0], R_0=data_fatality[0], a=a, decay=decay, vac_data =vaccines,vac_date_diff =date_difference , bias=bias)


    #model = Learner_SuEIR(N=N, E_0=E_0, I_0=data_confirm[0], R_0=data_fatality[0], a=a, decay=decay, bias=bias)
    init = [N-E_0-data_confirm[0]-data_fatality[0], E_0, data_confirm[0], data_fatality[0]]

    params_all, loss_all = rolling_train(model, init, train_data, new_sus, pop_in=pop_in)
    loss_true = [NE0_region[region][-2], NE0_region[region][-1]]

    pred_true = rolling_prediction(model, init, params_all, full_data, new_sus, pred_range=prediction_range, pop_in=pop_in, daily_smooth=True)

    confirm = full_data[0][0][0:-1].tolist() + full_data[1][0][0:-1].tolist() + pred_true[0].tolist()

    print ("region: ", region, " training loss: ",  \
        loss_all, loss_true," maximum death cases: ", int(pred_true[1][-1]), " maximum confirmed cases: ", int(pred_true[0][-1]))

    _, loss_true = rolling_likelihood(model, init, params_all, train_data, new_sus, pop_in=pop_in)
    data_length = [len(data[0]) for data in train_data]

    prediction_list = []
    interval = 0.3
    params = params_all[1] if len(params_all)==2 else params_all[2]
    while interval >= -0.0001:
        interval -= 0.01
        beta_list = np.asarray([1-interval,1+interval])*params[0]
        gamma_list = np.asarray([1-interval,1+interval])*params[1]
        sigma_list = np.asarray([1-interval,1+interval])*params[2]
        mu_list = np.asarray([1-interval,1+interval])*params[3]
        for beta0 in beta_list:
            for gamma0 in gamma_list:
                for sigma0 in sigma_list:
                    for mu0 in mu_list:
                        temp_param = [params_all[0]] + [np.asarray([beta0,gamma0,sigma0,mu0])]
                        if len(params_all)==3:
                            temp_param = [params_all[0]] + [params_all[1]] + [np.asarray([beta0,gamma0,sigma0,mu0])]
                        temp_pred=rolling_prediction(model, init, temp_param, full_data, new_sus, pred_range=prediction_range, pop_in=pop_in, daily_smooth=True)

                        _, loss = rolling_likelihood(model, init, temp_param, train_data, new_sus, pop_in=pop_in)
                        if loss < (9.5/data_length[1]*4+loss_true): ###################### 95% tail probability of Chi square (4) distribution
                            prediction_list += [temp_pred]

    A_inv, I_inv, R_inv = [],[],[]

    prediction_list += [pred_true]

    for _pred in prediction_list:
        I_inv += [_pred[0]]
        R_inv += [_pred[1]]
        A_inv += [_pred[2]]

    I_inv=np.asarray(I_inv)
    R_inv=np.asarray(R_inv)
    A_inv=np.asarray(A_inv)

    #set the percentiles of upper and lower bounds
    maxI=np.percentile(I_inv,100,axis=0)
    minI=np.percentile(I_inv,0,axis=0)
    maxR=np.percentile(R_inv,100,axis=0)
    minR=np.percentile(R_inv,0,axis=0)
    maxA=np.percentile(A_inv,100,axis=0)
    minA=np.percentile(A_inv,0,axis=0)

    # get the median of the curves
    # meanI=I_inv[-1,:]
    # meanR=R_inv[-1,:]
    # meanA=A_inv[-1,:]
    meanI=np.percentile(I_inv,50,axis=0)
    meanR=np.percentile(R_inv,50,axis=0)
    meanA=np.percentile(A_inv,50,axis=0)

    diffR, diffI = np.zeros(R_inv.shape), np.zeros(I_inv.shape)
    diffR[:,1:], diffI[:,1:] = np.diff(R_inv), np.diff(I_inv)


    diffmR, diffmI = np.zeros(meanR.shape), np.zeros(meanI.shape)

    # diffmR[1:] = np.diff(meanR)
    # diffmI[1:] = np.diff(meanI)

    difflR = np.percentile(diffR,0,axis=0)
    diffuR = np.percentile(diffR,100,axis=0)

    difflI = np.percentile(diffI,0,axis=0)
    diffuI = np.percentile(diffI,100,axis=0)

    diffmR = np.percentile(diffR,50,axis=0)
    diffmI = np.percentile(diffI,50,axis=0)


    dates = [pd.to_datetime(PRED_START_DATE)+ timedelta(days=i) \
             for i in range(prediction_range)]

    # print(len(dates), len(meanI))
    results0 = np.asarray([minI, maxI, minR, maxR, meanI, meanR, diffmR, difflR, diffuR, minA, maxA, meanA, diffmI, difflI, diffuI])
    results0 = np.asarray(results0.T)

    pred_data=pd.DataFrame(data=results0, index = dates, columns=["lower_pre_confirm", "upper_pre_confirm", "lower_pre_fata", "upper_pre_fata",'pre_confirm', \
        'pre_fata','pre_fata_daily','lower_pre_fata_daily','upper_pre_fata_daily','lower_pre_act','upper_pre_act', 'pre_act', \
        'pre_confirm_daily','lower_pre_confirm_daily','upper_pre_confirm_daily'])

    if args.level == "state" or args.level == "nation":
        pred_data['Region'] = region
    elif args.level == "county":
        pred_data['Region'] = county
        pred_data["State"] = state

    pred_data=pred_data.reset_index().rename(columns={"index": "Date"})
    frame.append(pred_data[pred_data['Date']>=datetime.strptime(PRED_START_DATE,"%Y-%m-%d")])


result = pd.concat(frame)
save_name = pred_dir + "pred_" + args.level + "_END_DATE_" + args.END_DATE + "_PRED_START_DATE_" + PRED_START_DATE + ".csv"
result.to_csv(save_name, index=False)
