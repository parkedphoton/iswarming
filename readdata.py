
import datetime
import pytz

import numpy as np
import pandas as pd


def load_data(tt, vv, filepath):
    data = pd.read_csv(filepath, 
                error_bad_lines=False, 
                encoding='utf-8', 
                na_values=['na'], quotechar='"')

    t = data.DATE.tolist()

    data.HourlyDryBulbTemperature = pd.to_numeric(data.HourlyDryBulbTemperature,errors='coerce')
    v = data.HourlyDryBulbTemperature.to_numpy().squeeze()

    tt = tt+t
    vv = np.concatenate((vv, v))
    
    return tt, vv    



def string_to_unixtime(s):
    
    string_format = '%Y-%m-%dT%H:%M:%S'
    
    dt = datetime.datetime.strptime(s, string_format).replace(microsecond=0)
        
    #incorrect approaches:
    #dt = pytz.utc.localize(dt)
    #dt.replace(tzinfo=pytz.timezone('EST'))
    
    dt = pytz.utc.localize(dt).astimezone(pytz.timezone('EST'))
    dt_origin = datetime.datetime(1970,1,1).replace(tzinfo=pytz.timezone('EST'))
    #dt_origin = datetime.datetime(1970,1,1)
    

    return (dt - dt_origin).total_seconds()
    


def string_to_unixtime_array(ss):
    t = []
    for s in ss:
        t.append(string_to_unixtime(s))
        
    return np.array(t)
    


def find_hod_doy(s):
    string_format = '%Y-%m-%dT%H:%M:%S'
    dt = datetime.datetime.strptime(s, string_format).replace(microsecond=0)
    dt = pytz.utc.localize(dt).astimezone(pytz.timezone('EST'))

    return dt.hour, dt.timetuple().tm_yday


def find_hod_doy_array(ss):
    hods = []
    doys = []
    for s in ss:
        hod, doy = find_hod_doy(s)
        hods.append(hod)
        doys.append(doy)
        
    return hods, doys




