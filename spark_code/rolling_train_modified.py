import numpy as np
from scipy.optimize import minimize


def loss(pred, target, smoothing=10):
    # loss mean calculated
    return np.mean((np.log(pred+smoothing) - np.log(target+smoothing))**2)

#method to train the model
def train(model, init, prev_params, train_data, reg=0, lag=0):
    #confirmed and fatalities.
    data_confirm, data_fatality = train_data[0], train_data[1]
    #length 3 if mid-surge date is there calculating totalfatalities.
    if len(train_data)==3:
        data_fatality = train_data[1] + train_data[2]
    size = len(data_confirm)
    #fatality and cases per day and their avg.
    fatality_perday = np.diff(data_fatality)
    target_ave_fatality_perday = np.median(
        fatality_perday[np.maximum(0, len(fatality_perday)-7):])
    confirm_perday = np.diff(data_confirm)
    target_ave_confirm_perday = np.median(
        confirm_perday[np.maximum(0, len(confirm_perday)-7):])

    # function called by scipy.odeint.optimize minimizing scalar function params
    def loss_train(params):

        _, _, _, pred_remove, pred_confirm, pred_fatality = model(size, params, init, lag)

        pred_fatality = pred_fatality + data_fatality[0] - pred_fatality[0]
        reg = 0.5
        if len(train_data)==3:
            pred_fatality = pred_remove
            reg = 0
        pred_ave_confirm_perday = np.mean(np.maximum(0, np.diff(pred_confirm)[-7:]))
        pred_ave_fatality_perday = np.mean(np.maximum(0, np.diff(pred_fatality)[-7:]))


        return loss(pred_confirm, data_confirm) + 1*loss(pred_fatality, data_fatality)
        + 1*loss(pred_ave_confirm_perday, target_ave_confirm_perday) + 3 * \
            loss(pred_ave_fatality_perday, target_ave_fatality_perday)

    optimal = minimize(
        loss_train,
        [0.2, .5e-2, 2.5e-1, 0.01],
        method='L-BFGS-B',
        bounds=[(0.0001, .3), (0.001, 0.3), (0.01, 1), (0.001, 1.)]
    )

    return optimal.x, optimal.fun

#wrapper for training during validation
def rolling_train(model, init, train_data, new_sus, pop_in=1/500):

    lag = 0
    params_all = []
    loss_all = []
    prev_params = [0.2, .5e-2, 3e-1, 0.01]
    reg = 0
    model.reset()
    N = model.N
    ind = 0
    # print (mean_increase, pop_in)
    for _train_data in train_data:
        ind += 1


        #team added code to ensure adding vaccinations to latter dates
        if ind == len(train_data):
            model.last_segment_of_data =True
        else:
            model.last_segment_of_data=False


        #cases,fatalities
        data_confirm, data_fatality = _train_data[0], _train_data[1]
        #train to get params and loss.
        params, train_loss = train(model, init, prev_params, _train_data, reg=reg, lag=lag)
        # model differentials calculated using solver_ivp.
        pred_sus, pred_exp, pred_act, pred_remove, _, _ = model(len(data_confirm), params, init, lag=lag)
        # print(params)
        lag += len(data_confirm)-10
        reg = 0

        if len(_train_data)==3:
            true_remove = np.minimum(data_confirm[-1], np.maximum(_train_data[1][-1] + _train_data[2][-1], pred_remove[-1]))
        else:
            true_remove = np.minimum(data_confirm[-1], pred_remove[-1])

        init = [pred_sus[-1], pred_exp[-1], data_confirm[-1]-true_remove, true_remove]
        init[0] = init[0] + new_sus
        model.N += new_sus
        if ind == 1:
            model.pop_in = pop_in
            model.bias=60
        else:
            # model.pop_in = pop_in
            model.bias = 50
        # print (params, train_loss)
        prev_params = params
        params_all += [params]
        loss_all += [train_loss]




    init[0] = init[0] - new_sus
    model.reset()
    pred_sus, pred_exp, pred_act, pred_remove, pred_confirm, pred_fatality = model(7, params, init, lag=lag)

    # print (pred_remove)

    return params_all, loss_all

def rolling_prediction(model, init, params_all, train_data, new_sus, pred_range, pop_in=1/500, daily_smooth=False):
    lag = 0
    model.reset()
    ind = 0
    # model.bias = len(train_data[0])+30
    for _train_data, params in zip(train_data, params_all):
        ind += 1
        data_confirm, data_fatality = _train_data[0], _train_data[1]
        pred_sus, pred_exp, pred_act, pred_remove, _, _ = model(len(data_confirm), params, init, lag=lag)

        if len(_train_data)==3:
            true_remove = np.minimum(data_confirm[-1], np.maximum(_train_data[1][-1] + _train_data[2][-1], pred_remove[-1]))
        else:
            true_remove = np.minimum(data_confirm[-1], pred_remove[-1])

        lag += len(data_confirm)-10
        #initialization params.
        init = [pred_sus[-1], pred_exp[-1], data_confirm[-1]-true_remove, true_remove]
        init[0] = init[0] + new_sus
        model.N += new_sus
        if ind == 1:
            model.pop_in = pop_in
            model.bias=60
        else:
            # model.pop_in = pop_in
            model.bias = 50


    model.bias = 60-len(data_confirm)
    if len(train_data)==3:
        model.bias = 50-len(data_confirm)
        # print(init)
    # print(model.N)
    # if len(train_data)==1:
    init[0] = init[0] - new_sus
    model.N -= new_sus

    # model.bias = 14
    pred_sus, pred_exp, pred_act, pred_remove, pred_confirm, pred_fatality = model(pred_range, params, init, lag=lag)
    pred_fatality = pred_fatality + train_data[-1][1][-1] - pred_fatality[0]

    # print(data_fatality)
    # pred_fatality = pred_remove
    fatality_perday = np.diff(np.asarray(data_fatality))
    ave_fatality_perday = np.mean(fatality_perday[-7:])

    confirm_perday = np.diff(np.asarray(data_confirm))
    ave_confirm_perday = np.mean(confirm_perday[-7:])

    slope_fatality_perday  = np.mean(fatality_perday[-7:] -fatality_perday[-14:-7] )/7
    slope_confirm_perday  = np.mean(confirm_perday[-7:] -confirm_perday[-14:-7] )/7

    smoothing = 1. if daily_smooth else 0



    temp_C_perday = np.diff(pred_confirm.copy())
    slope_temp_C_perday = np.diff(temp_C_perday)
    modified_slope_gap_confirm = (slope_confirm_perday - slope_temp_C_perday[0])*smoothing

    modified_slope_gap_confirm = np.maximum(np.minimum(modified_slope_gap_confirm, ave_confirm_perday/40), -ave_confirm_perday/100)
    slope_temp_C_perday = [slope_temp_C_perday[i] + modified_slope_gap_confirm * np.exp(-0.05*i**2) for i in range(len(slope_temp_C_perday))]
    # print (modified_slope_gap_confirm)
    temp_C_perday = [np.maximum(0, temp_C_perday[0] + np.sum(slope_temp_C_perday[0:i])) for i in range(len(slope_temp_C_perday)+1)]
    # print(np.array(temp_C_perday)[1:7])

    # temp_C_perday = np.diff(pred_confirm)
    modifying_gap_confirm = (ave_confirm_perday - temp_C_perday[0])*smoothing
    temp_C_perday  = [np.maximum(0, temp_C_perday[i] + modifying_gap_confirm * np.exp(-0.1*i)) for i in range(len(temp_C_perday))]
    temp_C =  [pred_confirm[0] + np.sum(temp_C_perday[0:i])  for i in range(len(temp_C_perday)+1)]
    pred_confirm = np.array(temp_C)



    temp_F_perday = np.diff(pred_fatality.copy())
    slope_temp_F_perday = np.diff(temp_F_perday)
    smoothing_slope = 0 if np.max(fatality_perday[-7:])>4*np.median(fatality_perday[-7:]) or np.median(fatality_perday[-7:])<0 else 1

    # print (smoothing)

    modified_slope_gap_fatality = (slope_fatality_perday - slope_temp_F_perday[0])*smoothing_slope
    modified_slope_gap_fatality = np.maximum(np.minimum(modified_slope_gap_fatality, ave_fatality_perday/10), -ave_fatality_perday/20)
    slope_temp_F_perday = [slope_temp_F_perday[i] + modified_slope_gap_fatality * np.exp(-0.05*i**2) for i in range(len(slope_temp_F_perday))]
    temp_F_perday = [np.maximum(0, temp_F_perday[0] + np.sum(slope_temp_F_perday[0:i])) for i in range(len(slope_temp_F_perday)+1)]


    modifying_gap_fatality = (ave_fatality_perday - temp_F_perday[0])*smoothing_slope
    temp_F_perday  = [np.maximum(0, temp_F_perday[i] + modifying_gap_fatality * np.exp(-0.05*i)) for i in range(len(temp_F_perday))]
    temp_F =  [pred_fatality[0] + np.sum(temp_F_perday[0:i])  for i in range(len(temp_F_perday)+1)]
    pred_fatality = np.array(temp_F)

    model.reset()
    return pred_confirm, pred_fatality, pred_act

def rolling_likelihood(model, init, params_all, train_data, new_sus, pop_in):
    lag = 0
    model.reset()
    loss_all = []
    N = model.N
    ind = 0
    for _train_data, params in zip(train_data, params_all):
        ind += 1
        data_confirm, data_fatality = _train_data[0], _train_data[1]
        pred_sus, pred_exp, pred_act, pred_remove, pred_confirm, pred_fatality = model(len(data_confirm), params, init, lag=lag)
        pred_fatality = pred_fatality + data_fatality[0] - pred_fatality[0]


        est_perday_confirm, data_perday_confirm = np.diff(pred_confirm), np.diff(data_confirm)
        est_perday_fatality, data_perday_fatality = np.diff(pred_fatality), np.diff(data_fatality)

        loss_all += [np.mean(((est_perday_confirm) - (data_perday_confirm))**2/2/est_perday_confirm**2) \
             + np.mean(((est_perday_fatality) - (data_perday_fatality))**2/2/est_perday_fatality**2)]

        if len(_train_data)==3:
            true_remove = np.minimum(data_confirm[-1], np.maximum(_train_data[1][-1] + _train_data[2][-1], pred_remove[-1]))
        else:
            true_remove = np.minimum(data_confirm[-1], pred_remove[-1])



        lag += len(data_confirm)-10
        init = [pred_sus[-1], pred_exp[-1], data_confirm[-1]-true_remove, true_remove]
        init[0] = init[0] + new_sus
        model.N += new_sus
        model.pop_in = pop_in
        if ind == 1:
            model.pop_in = pop_in
            model.bias=60
        else:
            # model.pop_in = pop_in
            model.bias = 50

    model.reset()
    return loss_all[0], loss_all[-1]
