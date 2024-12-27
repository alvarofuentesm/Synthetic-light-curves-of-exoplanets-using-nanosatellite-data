from pytransit import QuadraticModel
from PyAstronomy import pyasl
from astropy import units as u
from astropy.constants import au, R_sun, R_jup
import numpy as np
from scipy.interpolate import Akima1DInterpolator
import matplotlib.pyplot as plt
from wotan import flatten

def generate_artificial_transit(times, k, t0, p, a, i, e, w, seed = 22):
    
    if (seed != None):
        np.random.seed(seed)
    
    # limb darkening coefficient vector
    ldc = [np.round(np.random.normal(0.3, 0.05, 1)[0],2), np.round(np.random.normal(0.1, 0.02, 1)[0], 2) ]
    
    tm = QuadraticModel()
    tm.set_data(times)
    
    print(f"t0:{t0}, p:{p}, a:{a}, i:{i}, e:{e}, w:{w}, ldc:{ldc}")  
    flux = tm.evaluate_ps(k, ldc, t0, p, a, i, e, w)  
    
    return tm, ldc, flux
    
def calculate_transit_duration(sma, rp, rs, i, period): # i in rad
    '''
    sma:The semi-major axis in AU. (float)

    rp: The planetary radius in Jovian radii. (float)

    rs: The stellar radius in solar radii. (float)

    inc: The orbital inclination in degrees. (float)

    period: The orbital period. (float)
    '''
    inc = i*(180)/np.pi # degree 

    return pyasl.transitDuration(sma, rp, rs, inc, period)

#########  Inject transit + Transform magnitude to relative brightness #########
def inject_transit(data, k, p, a, i, e, w, star_radius, t0 = None, seed = 44, plot = False):
    
    if (plot):
        fig, ax = plt.subplots(figsize=(9,5))
        plt.title(f"Decorrelated light curve", fontsize = 15)
        plt.ylabel("Relative Magnitude", fontsize = 13)
        plt.xlabel("HJD - 2456000.0", fontsize = 13)
        plt.plot(data["Time"],  data["Magnitude"], 'b.', alpha = 1.0)
        plt.show()
        print("STD (decorrelated): ", np.std(data["Magnitude"].copy() ) ) 

    
    # detrend using wotan
    flatten_flux, trend = flatten(data["Time"].copy().values, data["Magnitude"].copy().values + 1, window_length=0.5, return_trend=True, method='biweight')

    if (plot):
        fig, ax = plt.subplots(figsize=(9,5))
        plt.title(f"Decorrelated light curve (flatten)", fontsize = 15)
        plt.ylabel("Relative Magnitude", fontsize = 13)
        plt.xlabel("HJD - 2456000.0", fontsize = 13)
        plt.plot(data["Time"], flatten_flux, 'b.', alpha = 1.0)
        plt.show() 
 
    ## now injection            
    tm, ldc, flux_transit = generate_artificial_transit(data["Time"], k, t0, p, a, i, e, w, seed = seed )

    flux_transit -= 1 # since pytransit.QuadraticModel use 1 as base for flux values 

    if ( np.count_nonzero(np.isinf(flux_transit)) == len(flux_transit) ): ## all values in flux_transit are inf 
        return [], [], t0, 0, flux_transit, []
    
    '''
    sma:The semi-major axis in AU. (float)
    planet_radius: The planetary radius in Jovian radii. (float)
    star_radius: The stellar radius in solar radii. (float)
    '''
    sma = (star_radius*R_sun*a).to(u.au).value # sma = sr * aos [AU]
    planet_radius = k*star_radius*R_sun/R_jup # k = planet_radius/star_radius => planet_radius = k*radius_star
    print("Radius planet (Rjup): ", planet_radius)
    transit_duration = calculate_transit_duration(sma = sma, rp = planet_radius, rs = star_radius, i = i, period = p).value

    times = data["Time"].copy().values
    index_to_replace = []
    
    t_i = t0
    max_time = np.max(times) 
    
    delta = 0.5*transit_duration
    while(t_i <= max_time): # time > t0
        in_transit = [( data["Time"] >= (t_i - delta) )&( data["Time"] <= (t_i + delta))][0]
        # points: used to add noise to the injected signal
        points = data["Magnitude"][( data["Time"] >= (t_i - delta) )&( data["Time"] <= (t_i + delta) )] 
        bool_list = list(in_transit)
        indexs = [i for i, b in enumerate(bool_list) if b == True]
        
        for idx in indexs:
            index_to_replace.append(idx)
        t_i += p 
        
    min_time = min(times)
    t_i = t0 
    while(t_i >= min_time): #  time <= t0
        in_transit = [( data["Time"] >= (t_i - delta) )&( data["Time"] <= (t_i + delta))][0]
        points = data["Magnitude"][( data["Time"] >= (t_i - delta) )&( data["Time"] <= (t_i + delta) )]
        bool_list = list(in_transit)
        indexs = [i for i, b in enumerate(bool_list) if b == True]  
        
        for idx in indexs:
            index_to_replace.append(idx)
        t_i -= p 
        
    flux = flatten_flux
    
    times_transit = times[index_to_replace]
    only_transit = []
     
    for i, idx in enumerate(index_to_replace):
        flux[idx] = flux[idx] +  np.absolute(flux_transit[idx])
        only_transit.append(flux[idx])
        
    if (plot):
        fig, ax = plt.subplots(figsize=(9,5))
        plt.title(f"Light curve (flatten & injected transit)", fontsize = 15)
        plt.ylabel("Relative Magnitude", fontsize = 13)
        plt.xlabel("HJD - 2456000.0", fontsize = 13)
        plt.plot(times, flux, 'b.', alpha = 1.0, label = "Decorrelated & flatten")
        plt.plot(times_transit, only_transit, '.', color = 'orange', alpha = 1.0, label = "Injected transit")
        plt.gca().invert_yaxis() # invert y axis
        plt.legend()
        plt.show()

        fig, ax = plt.subplots(figsize=(9,5))
        plt.plot(times, flux_transit+1, '.', color = 'red', alpha = 0.5)
        plt.show()
    
    return times, flux, t0, transit_duration, flux_transit, len(times_transit), ldc