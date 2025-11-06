import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#from scipy.integrate import trapezoid as trapz, cumulative_trapezoid as cumtrapz
from scipy.optimize import curve_fit
from scipy.interpolate import *
from scipy.integrate import *
from scipy import special
import pandas as pd
from tqdm import tqdm
from time import time
import os
import sys
from scipy.special import erfc

import joblib
from joblib import Parallel, delayed

from colossus.lss import mass_function
from colossus.cosmology import cosmology

import warnings
warnings.filterwarnings("ignore")

from modules.input_parameters import *
def read_input_params(input_filename):
    input_params = read_input_parameters(input_filename)
    return input_params

cosmo = cosmology.setCosmology('planck18')
h=cosmo.H0/100.
h70=cosmo.H0/70.

c=3e8 #[m/s]
Msun=2e30 #[kg]
erg_to_J=1e-7
sec_to_yr=31536000
logLsun=np.log10(3.846e33) #[erg/s]
scatter_mhdotlog_mhlog=0.194



lb_time=np.arange(0.001,13.421,0.1)#Gyr
z=cosmo.lookbackTime(lb_time,inverse=True)
nz=z.size
logz=np.log10(z)
cosmic_time=cosmo.age(z)

grid_width = 0.005
lt_grid = np.arange(lb_time[0],lb_time[-1]+grid_width, grid_width)
z_grid = cosmo.lookbackTime(lt_grid, inverse=True)
nz_grid = z_grid.size

z_SatGen = np.array([0.00000000e+00, 4.30881684e-03, 8.64468153e-03, 1.30078715e-02, 1.73987469e-02, 2.18175685e-02, 2.62646870e-02, 3.07404080e-02, 3.52450566e-02, 3.97789780e-02, 4.43424954e-02, 4.89359668e-02, 5.35597393e-02, 5.82141550e-02, 6.28996116e-02, 6.76164259e-02, 7.23650328e-02, 7.71457928e-02, 8.19590590e-02, 8.68052899e-02, 9.16848731e-02, 9.65981966e-02, 1.01545665e-01, 1.06527762e-01, 1.11544910e-01, 1.16597540e-01, 1.21686098e-01, 1.26811040e-01, 1.31972836e-01, 1.37171965e-01, 1.42408918e-01, 1.47684196e-01, 1.52998310e-01, 1.58351780e-01, 1.63745134e-01, 1.69178908e-01, 1.74653643e-01, 1.80169890e-01, 1.85728199e-01, 1.91329178e-01, 1.96973445e-01, 2.02661573e-01, 2.08394140e-01, 2.14171867e-01, 2.19995341e-01, 2.25865218e-01, 2.31782198e-01, 2.37746915e-01, 2.43760126e-01, 2.49822499e-01, 2.55934782e-01, 2.62097734e-01, 2.68312059e-01, 2.74578596e-01, 2.80898134e-01, 2.87271457e-01, 2.93699378e-01, 3.00182755e-01, 3.06722490e-01, 3.13319459e-01, 3.19974562e-01, 3.26688719e-01, 3.33462873e-01, 3.40297982e-01, 3.47195064e-01, 3.54155159e-01, 3.61179288e-01, 3.68268482e-01, 3.75423834e-01, 3.82646514e-01, 3.89937582e-01, 3.97298290e-01, 4.04729761e-01, 4.12233298e-01, 4.19810069e-01, 4.27461446e-01, 4.35188733e-01, 4.42993248e-01, 4.50876366e-01, 4.58839535e-01, 4.66884238e-01, 4.75011958e-01, 4.83224219e-01, 4.91522613e-01, 4.99908793e-01, 5.08384406e-01, 5.16951115e-01, 5.25610674e-01, 5.34364944e-01, 5.43215667e-01, 5.52164870e-01, 5.61214421e-01, 5.70366321e-01, 5.79622704e-01, 5.88985658e-01, 5.98457351e-01, 6.08040031e-01, 6.17736012e-01, 6.27547668e-01, 6.37477422e-01, 6.47527775e-01, 6.57701430e-01, 6.68000950e-01, 6.78429179e-01, 6.88988832e-01, 6.99682979e-01, 7.10514592e-01, 7.21486762e-01, 7.32602701e-01, 7.43865729e-01, 7.55279258e-01, 7.66846775e-01, 7.78571998e-01, 7.90458684e-01, 8.02510705e-01, 8.14732040e-01, 8.27126957e-01, 8.39699719e-01, 8.52454762e-01, 8.65396679e-01, 8.78530187e-01, 8.91860362e-01, 9.05392232e-01, 9.19131146e-01, 9.33082510e-01, 9.47252054e-01, 9.61645722e-01, 9.76269661e-01, 9.91130292e-01, 1.00623414e+00, 1.02158812e+00, 1.03719925e+00, 1.05307510e+00, 1.06922335e+00, 1.08565200e+00, 1.10236955e+00, 1.11938455e+00, 1.13670621e+00, 1.15434394e+00, 1.17230760e+00, 1.19060749e+00, 1.20925427e+00, 1.22825936e+00, 1.24763442e+00, 1.26739172e+00, 1.28754407e+00, 1.30810486e+00, 1.32908829e+00, 1.35050909e+00, 1.37238268e+00, 1.39472523e+00, 1.41755394e+00, 1.44088666e+00, 1.46474215e+00, 1.48914014e+00, 1.51410167e+00, 1.53964861e+00, 1.56580402e+00, 1.59259252e+00, 1.62003977e+00, 1.64817309e+00, 1.67702114e+00, 1.70658071e+00, 1.73644201e+00, 1.76660857e+00, 1.79708395e+00, 1.82787171e+00, 1.85897567e+00, 1.89039957e+00, 1.92214703e+00, 1.95422174e+00, 1.98662777e+00, 2.01936884e+00, 2.05244865e+00, 2.08587146e+00, 2.11964102e+00, 2.15376128e+00, 2.18823646e+00, 2.22307037e+00, 2.25826737e+00, 2.29383146e+00, 2.32976682e+00, 2.36607778e+00, 2.40276841e+00, 2.43984327e+00, 2.47730642e+00, 2.51516258e+00, 2.55341582e+00, 2.59207101e+00, 2.63113230e+00, 2.67060459e+00, 2.71049221e+00, 2.75080004e+00, 2.79153262e+00, 2.83269478e+00, 2.87429133e+00, 2.91632695e+00, 2.95880682e+00, 3.00173553e+00, 3.04511844e+00, 3.08896037e+00, 3.13326641e+00, 3.17804188e+00, 3.22329164e+00, 3.26902133e+00, 3.31523607e+00, 3.36194106e+00, 3.40914207e+00, 3.45684427e+00, 3.50505326e+00, 3.55377477e+00, 3.60301414e+00, 3.65277731e+00, 3.70307005e+00, 3.75389787e+00, 3.80526698e+00, 3.85718325e+00, 3.90965241e+00, 3.96268084e+00, 4.01627462e+00, 4.07043967e+00, 4.12518247e+00, 4.18050941e+00, 4.23642659e+00, 4.29294055e+00, 4.35005804e+00, 4.40778538e+00, 4.46612906e+00, 4.52509630e+00, 4.58469364e+00, 4.64492768e+00, 4.70580572e+00, 4.76733476e+00, 4.82952164e+00, 4.89237346e+00, 4.95589786e+00, 5.02010191e+00, 5.08499274e+00, 5.15057816e+00, 5.21686575e+00, 5.28386289e+00, 5.35157704e+00, 5.42001662e+00, 5.48918926e+00, 5.55910265e+00, 5.62976484e+00, 5.70118436e+00, 5.77336916e+00, 5.84632727e+00, 5.92006731e+00, 5.99459795e+00, 6.06992751e+00, 6.14606438e+00, 6.22301766e+00, 6.30079629e+00, 6.37940894e+00, 6.45886437e+00, 6.53917216e+00, 6.62034154e+00, 6.70238159e+00, 6.78530145e+00, 6.86911110e+00, 6.95382020e+00, 7.03943819e+00, 7.12597464e+00, 7.21343991e+00, 7.30184412e+00, 7.39119714e+00, 7.48150893e+00, 7.57279023e+00, 7.66505169e+00, 7.75830361e+00, 7.85255640e+00, 7.94782110e+00, 8.04410897e+00, 8.14143077e+00, 8.23979738e+00, 8.33922011e+00, 8.43971091e+00, 8.54128102e+00, 8.64394178e+00, 8.74770475e+00, 8.85258265e+00, 8.95858721e+00, 9.06573029e+00, 9.17402385e+00, 9.28348103e+00, 9.39411431e+00, 9.50593605e+00, 9.61895874e+00, 9.73319575e+00, 9.84866044e+00, 9.96536572e+00, 1.00833246e+01, 1.02025507e+01, 1.03230584e+01, 1.04448610e+01, 1.05679722e+01, 1.06924059e+01, 1.08181772e+01, 1.09453002e+01, 1.10737894e+01, 1.12036588e+01, 1.13349241e+01, 1.14676005e+01, 1.16017028e+01, 1.17372460e+01, 1.18742456e+01, 1.20127182e+01, 1.21526793e+01, 1.22941444e+01, 1.24371295e+01, 1.25816519e+01, 1.27277280e+01, 1.28753742e+01, 1.30246069e+01, 1.31754437e+01, 1.33279024e+01, 1.34819999e+01, 1.36377536e+01, 1.37951809e+01, 1.39543012e+01, 1.41151323e+01, 1.42776921e+01, 1.44419988e+01, 1.46080720e+01, 1.47759310e+01, 1.49455945e+01, 1.51170816e+01, 1.52904118e+01, 1.54656060e+01, 1.56426839e+01, 1.58216652e+01, 1.60025701e+01, 1.61854202e+01, 1.63702366e+01, 1.65570399e+01, 1.67458510e+01, 1.69366918e+01, 1.71295850e+01, 1.73245522e+01, 1.75216153e+01, 1.77207962e+01, 1.79221193e+01, 1.81256073e+01, 1.83312830e+01, 1.85391694e+01, 1.87492909e+01, 1.89616722e+01, 1.91763370e+01, 1.93933095e+01, 1.96126139e+01, 1.98342772e+01, 2.00583242e+01])

"""nz=50
z=np.linspace(0.01,5.01,nz)
logz=np.log10(z)
cosmic_time=np.flip(cosmo.age(z))
lb_time=cosmo.lookbackTime(z)"""

dzdt=1.02276e-10*h*(1.+z)*np.sqrt(1.-cosmo.Om0+cosmo.Om0*(1.+z)**3)

nms=350
mslog_min=5.
mslog_max=12.
mslog=np.linspace(mslog_min,mslog_max,nms)
ms=10.**mslog

nmhdot=1000
mhdotlog=np.linspace(0.5,17.,nmhdot)-9.

sfrlog = np.arange(-6, 5, 0.1)
nsfr=sfrlog.size


nmh=1000
mhlog=np.linspace(5.,15.5,nmh)
mh=10.**mhlog
mhlog_min=11.; mhlog_min_cat=11.


nhsar=100
hsarlog=np.linspace(-5.,2.,nhsar)
hsar=10.**hsarlog


sbin=0.1
logsigma=np.arange(1., 3., sbin)
nsigma=logsigma.size


mbhdotlog=np.arange(-6, 2, 0.1)
nmbhdot=mbhdotlog.size



def nearest(vector,value):
    #res=(np.abs(vector - value)).argmin()
    #return res
    return np.argmin(np.abs(vector - value))


def gaussian(x,m,s):
    return np.exp( -0.5 * (x-m)**2. / s**2 ) / (s * np.sqrt(2.*np.pi) )

def straight_line(x,m,q):
    return m*x+q

def line_across_2_points(x, x1, y1, x2, y2):
    return (x-x1)/(x1-x2) * (y1-y2) + y1



def compute_dyn_friction_timescale(z):
    x=cosmo.Om(z)-1.
    Dvir=18.*np.pi**2.+82.*x-39.*x**2.
    return 1.628 / h / np.sqrt(Dvir/178.) * (cosmo.H0/cosmo.Hz(z))



def integrate_accretion_rates_across_time(z,log_acc_rate_cat):
    mass = np.flip( np.transpose( cumtrapz(np.flip(10.**np.transpose(log_acc_rate_cat)), np.flip(cosmo.age(z))*10.**9., axis=0, initial=0.) ) )
    return mass, np.log10(mass)



def mhdotlog_mhlog_relation(z,Z,mhlog):
    params = np.array([[  1.10320981,   1.04189297,   1.05466866,   1.01121661, 1.03165295,   0.99784522,   1.01528466,   0.98234759,
              1.00045836,   0.98121872,   0.99934336,   0.97855251, 0.99174069,   0.99147979,   0.97325701,   0.986431  ,
              0.99394462,   0.97616078,   0.98280507,   0.97281999, 0.97521068,   0.97400765,   0.97877229,   0.96974662,
              0.97969836,   0.98136448,   0.96579367,   0.97107004, 0.97137628,   0.96412803,   0.94359286,   0.95533298,
              0.95484576,   0.95249929,   0.93367736,   0.92886104, 0.92186463,   0.92433048,   0.92918776,   0.91255546,
              0.8868027 ,   0.88575913,   0.88388416,   0.87907199, 0.8759381 ,   0.88405567,   0.87797213,   0.87348   ,
              0.87810947,   0.87648979],
              [-11.67642727, -10.77454112, -10.87843318, -10.23739005, -10.42739899,  -9.92310931, -10.08444426,  -9.61259494,
             -9.78200797,  -9.47563309,  -9.65115942,  -9.34054795, -9.46072264,  -9.42756152,  -9.15990431,  -9.27898178,
             -9.33309742,  -9.08339908,  -9.13010439,  -8.97306045, -8.98114725,  -8.9410926 ,  -8.97140367,  -8.83409055,
             -8.91913663,  -8.91377223,  -8.71184186,  -8.74639581, -8.74285867,  -8.64603941,  -8.40088859,  -8.50008721,
             -8.48270876,  -8.44779874,  -8.22080449,  -8.15948178, -8.0692837 ,  -8.08691714,  -8.12882938,  -7.92934301,
             -7.6608463 ,  -7.63523838,  -7.60814981,  -7.5414963 , -7.4935035 ,  -7.56230647,  -7.50432552,  -7.43920168,
             -7.47667834,  -7.44697678]])
    m = interp1d(z,params[0,:],fill_value="extrapolate")(Z)
    q = interp1d(z,params[1,:],fill_value="extrapolate")(Z)
    return interp1d(mhlog, mhlog*m+q,fill_value="extrapolate")


def Mbh_sigma_relation(logsigma, z, velocity_dispersion):
    #logsigma in [km/s]
    #alpha=3.83; beta=8.21 #fit from Agnese's thesis
    if velocity_dispersion=="Ferrarese+2002":
        alpha=4.58; beta=8.22 #Merritt&Ferrarese+2001, sigma_c computed at r_e/8, with r_e the half light radius
        return alpha*(logsigma-np.log10(200))+beta
    if velocity_dispersion=="Marsden+2022":
        mm=np.loadtxt("Data/Marsden_2022/Mbh_sigma.txt")
        marsden=interp2d(np.array([0.1,1.,2.,3]), np.arange(2.0,2.441,0.02), mm)
        return np.array([marsden(z,logsigma[i])[0] for i in range(logsigma.size)])



def abundance_matching(z, nz, phisfrLF, red_har, phihar, mhdotlog, sigma_am=0.2, delay=False):

    def derivative(x,y):
        func = interp1d(x,y,fill_value="extrapolate")
        dx = 0.1
        x_calc = np.arange(x[0],x[-1]+dx,dx)
        y_calc = func(x_calc)
        dydx = np.diff(y_calc)/np.diff(x_calc)
        dydx = np.append(dydx, dydx[-1])
        dydx_low = np.mean(dydx[:10])
        dydx[0] = dydx_low
        dydx[1] = dydx_low
        dydx_high = np.mean(dydx[-10:])
        return interp1d(x_calc,dydx,fill_value=(dydx_low,dydx_high),bounds_error = False)
    def integrals(SMF , HMF , sig):
        M_s , phi_s = SMF[0] , SMF[1]
        M_h , phi_h = HMF[0] , HMF[1]
        phi_s , phi_h = 10.0**phi_s , 10.0**phi_h
        I_phi_s = np.flip(cumtrapz(np.flip(phi_s), M_s))
        I_phi_h = np.array([])
        for m in M_h:
            I = np.trapz(phi_h*0.5*erfc((m-M_h)/(np.sqrt(2)*sig)) , M_h)
            I_phi_h = np.append(I_phi_h,I)
        M_s , M_h = M_s[:-1]+0.025 , M_h+0.025
        return I_phi_s , I_phi_h , M_s , M_h
    def SFRFfromSFRHAR(Sfr , Har , sig , z , HARF):
        bins = 0.1
        volume = (200*cosmo.h)**3
        mhdot_harf = HARF[:,0]
        phi = 10**HARF[:,1]
        cum_phi = np.cumsum(phi)
        max_number = np.floor(np.trapz(phi*volume, mhdot_harf))
        if (np.random.uniform(0,1) > np.trapz(phi, mhdot_harf)-max_number):
            max_number += 1
        int_cum_phi = interp1d(cum_phi, mhdot_harf)
        range_numbers = np.random.uniform(np.min(cum_phi), np.max(cum_phi), int(max_number))
        halo_acc_rates = int_cum_phi(range_numbers)
        sfr_smf = np.arange(-5., 4 , 0.1) #SMF histogram bins
        SFRHARinterp = interp1d(Har , Sfr , fill_value="extrapolate")
        star_form_rates = SFRHARinterp(halo_acc_rates) + np.random.normal(0., sig, halo_acc_rates.size)
        phi_sfr = np.histogram(star_form_rates , bins = sfr_smf)[0]/0.1/volume
        return sfr_smf[:-1]+0.05 , np.log10(phi_sfr)

    matrix = []
    sfr_har_z0=np.loadtxt("../../SatGen/Daniel/HARfunctions/SFR_HAR_z0.txt")
    derivative_z0 = np.diff(sfr_har_z0) / np.diff(mhdotlog)


    for iz in range(nz):
        idx_harf=nearest(red_har, z[iz])
        sig=sigma_am
        phi_s = phisfrLF[:,iz]

        deri = interp1d(mhdotlog[:-1]+0.01651652/2., derivative_z0, fill_value="extrapolate")(phihar[idx_harf][:,0])
        n=0; e=1.

        while n < 10:
            I_phi_s , I_phi_h , sfr_temp , mhdot_temp = integrals(np.array([sfrlog , phisfrLF[:,iz]]) , np.array([phihar[idx_harf][:,0] , phihar[idx_harf][:,1]]) , sig/deri)
            int_I_phi_s = interp1d(I_phi_s , sfr_temp , fill_value="extrapolate")
            sfr_match = np.array([ int_I_phi_s(I_phi_h[h]) for h in range(mhdot_temp.size) ])

            sfr_iter , phi_sfr_iter = SFRFfromSFRHAR(sfr_match , mhdot_temp , sig , z , phihar[idx_harf])
            int_phi_s_iter = interp1d(sfr_iter , phi_sfr_iter , fill_value="extrapolate")
            e_array = np.abs((phi_s - int_phi_s_iter(sfrlog))/phi_s)
            mask = np.where(np.isfinite(e_array))
            if e_array[mask].size < 1:
                e_temp = +np.inf
            else:
                e_temp = max(e_array[mask])

            if e_temp < e:
                e = e_temp
                deri = interp1d(mhdot_temp[:-1]+np.diff(mhdot_temp)[0]/2., np.diff(sfr_match)/np.diff(mhdot_temp) , fill_value="extrapolate" )(phihar[idx_harf][:,0])
            n += 1
        matrix.append( interp1d(mhdot_temp, sfr_match, fill_value="extrapolate")(mhdotlog) )

        percent = float(iz+1)*100./float(nz)
        sys.stdout.write("\r{per:.2f}%".format(per = percent))
        sys.stdout.flush()
    matrix=np.transpose(matrix)

    return 10.**matrix, matrix

def abundance_matching_LB(z,nz,mhdotlog,nmhdot, phisfr,sfrlog,nsfr, dNdVdlogmhdot_active, sigma_am=0.2, delay=False):
    if delay:
        tau_dyn=np.array([compute_dyn_friction_timescale(Z) for Z in z])
        z_after_dyn=cosmo.age( cosmo.age(z)+tau_dyn, inverse=True)

    phisfr_cum=np.zeros((nsfr,nz))
    for iz in range(nz):
        phisfr_cum[:,iz]=trapz(phisfr[:,iz],sfrlog)-cumtrapz(phisfr[:,iz],sfrlog,initial=0.)
    phisfr_cum[phisfr_cum<0.]=10.**(-66.)

    if sigma_am==0.:
        dNdVdlogmhdot_active_cum=np.zeros((nz,nmhdot))
        for iz in range(nz):
            dNdVdlogmhdot_active_cum[iz,:]=trapz(dNdVdlogmhdot_active[iz,:],mhdotlog)-cumtrapz(dNdVdlogmhdot_active[iz,:],mhdotlog,initial=0.)
        dNdVdlogmhdot_active_cum[dNdVdlogmhdot_active_cum<0.]=10.**(-66.)
        sfrlog_am=np.zeros((nmhdot,nz))
    elif sigma_am>0.:
        dNdVdlogmhdot_active_cum=np.zeros((nz,nmhdot))
        integrand_2nd_part = np.array([ 0.5*special.erfc((mhdotlog[imhdot]-mhdotlog)/np.sqrt(2.)/sigma_am) for imhdot in range(nmhdot) ])
        for iz in range(nz):
            if delay:
                z_before_delay=interp1d(z_after_dyn,z,fill_value="extrapolate")(z[iz])
                dNdVdlogmhdot_active_cum[iz,:] = np.trapz(dNdVdlogmhdot_active[nearest(z,z_before_delay),:]*integrand_2nd_part, mhdotlog, axis=1)
            elif not delay:
                dNdVdlogmhdot_active_cum[iz,:] = np.trapz(dNdVdlogmhdot_active[iz,:]*integrand_2nd_part, mhdotlog, axis=1)
        dNdVdlogmhdot_active_cum[dNdVdlogmhdot_active_cum<0.]=10.**(-66.)
        sfrlog_am=np.zeros((nmhdot,nz))
    for iz in range(nz):
        sfrlog_am[:,iz] = interp1d(np.log10(np.flip(phisfr_cum[:,iz])),np.flip(sfrlog),fill_value="extrapolate")(np.log10(dNdVdlogmhdot_active_cum[iz,:]))
        if sfrlog_am[np.isfinite(sfrlog_am[:,iz]),iz].size >0:
            sfrlog_am[np.logical_not(np.isfinite(sfrlog_am[:,iz])),iz] = sfrlog_am[np.isfinite(sfrlog_am[:,iz]),iz][-1]
    return 10.**sfrlog_am, sfrlog_am



def smooth_sfr_har_relation(nz,sfrlog_am,nmhdot,mhdotlog):
    for iz in range(nz):
        dsfrlog_dmhdotlog=np.diff(sfrlog_am[:,iz]) / np.diff(mhdotlog)
        for imhdot in range(nmhdot-1):
            if dsfrlog_dmhdotlog[imhdot] > 0.:
                idx=imhdot
                break
        if idx<nmhdot:
            sfrlog_am[:idx,iz]=interp1d(mhdotlog[idx:], sfrlog_am[idx:,iz], fill_value="extrapolate")(mhdotlog[:idx])
    return sfrlog_am



def SMHM_double_pl(mhlog,N,logM,b,g):
    return np.log10( np.power(10, mhlog) * 2*N* np.power( (np.power(np.power(10,mhlog-logM), -b) + np.power(np.power(10,mhlog-logM), g)), -1) )



def Mstar_recycle(z, nz, nhalo, sfrlog_cat):
    sfrlog_cat_real = np.zeros((nhalo,nz))
    for iz in range(nz):
        if z[iz]<=3.:
            sfrlog_cat_real[:,iz]=sfrlog_cat[:,iz].copy()+np.log10(1.-0.44)
        else:
            sfrlog_cat_real[:,iz]=sfrlog_cat[:,iz].copy()
    return sfrlog_cat_real

def sfr_catalogue(z_start, lb_time, z,nz,nhalo,sfrlog_am,sigma_sfr,mhdotlog,dmhdtlog_arr,mhlog_arr, sfr_delay_alpha, mhlog_crit, scatter_mhlog_crit=0., frac_quenched_byhalo=1., include_SNfeedback=True, logMh_SNfeedback=11., delay=False):
    if delay:
        tau_dyn=np.array([compute_dyn_friction_timescale(Z) for Z in z])
        z_after_dyn=cosmo.age( cosmo.age(z)+tau_dyn, inverse=True)
    sfrlog_cat=np.zeros((nhalo,nz))
    jquench_cat=np.zeros((nhalo,nz),dtype=bool)
    
    disp_mhlog_crit=np.random.normal(0,scatter_mhlog_crit,nhalo)
    
    for iz in range(nz):
        if delay:
            z_before_delay=interp1d(z_after_dyn,z,fill_value="extrapolate")(z[iz])
            sfrlog_cat[:,iz]=interp1d(mhdotlog,sfrlog_am[:,iz],bounds_error=False, fill_value='extrapolate')(dmhdtlog_arr[:,nearest(z,z_before_delay)])
        else:
            sfrlog_cat[:,iz]=interp1d(mhdotlog,sfrlog_am[:,iz],bounds_error=False, fill_value='extrapolate')(dmhdtlog_arr[:,iz])
        jquench_cat[:,iz]=mhlog_arr[:,iz]>mhlog_crit(z[iz]) + disp_mhlog_crit

        if frac_quenched_byhalo<1.:
            for ihalo in range(nhalo):
                if jquench_cat[ihalo,iz] and np.random.uniform(0.,1.)>frac_quenched_byhalo:
                    jquench_cat[ihalo,iz]=False

        sfrlog_cat[:,iz] += np.random.normal(0, sigma_sfr,nhalo)
        #sc=interp1d(mhdotlog, gaussian(mhdotlog,1.5,0.7)*1.1+0.6, fill_value="extrapolate")(dmhdtlog_arr[:,iz])
        #scatt=np.array([ np.random.normal(0,sc[ihalo]) for ihalo in range(nhalo)])
        #sfrlog_cat[:,iz] += scatt

        sfrlog_cat[jquench_cat[:,iz],iz]=-66.

    #correzione per riattivazioni
    #for ihalo in range(nhalo):
    #    for iz in range(0,nz):
    #        if sfrlog_cat[ihalo,iz]==-66.:
    #            sfrlog_cat[ihalo,:iz]=-66.

    #for ihalo in range(nhalo):
    #    iz_inst_quench=np.min(np.where(jquench_cat[ihalo,:]==False))
    #    z_delayed_quench = cosmo.age( cosmo.age(z[iz_inst_quench]) + 2., inverse=True )
    #    iz_delayed_quench = nearest(z,z_delayed_quench)
    #    jquench_cat[ihalo,iz_delayed_quench:]=False
    #
    #    sfrlog_cat[ihalo,jquench_cat[ihalo,:]]=-66.
    
    

    for ihalo in tqdm(range(nhalo)):
        if sfrlog_cat[ihalo,sfrlog_cat[ihalo,:]>-64.].size > 0:
            idx_quench=np.min(np.where(sfrlog_cat[ihalo,:]>-64.)[0])
            dt=lb_time[idx_quench] - lb_time[0:idx_quench]
            sfrlog_cat[ihalo,0:idx_quench] = sfrlog_cat[ihalo,idx_quench]+ np.log10(np.exp(-dt*sfr_delay_alpha)) + np.random.normal(0, sigma_sfr, sfrlog_cat[ihalo,0:idx_quench].size)

    if include_SNfeedback:
        for iz in range(nz):
            sfrlog_cat[mhlog_arr[:,iz] < logMh_SNfeedback,iz]=-65.

    sfrlog_cat[:,z>z_start]=-65.
    # impongo -65 perché voglio che la SFR sia 0, ma non deve essere contata come quenchata (le quenchate vengono classificate come logsfr==-66)

    sfrlog_cat_real = Mstar_recycle(z, nz, nhalo, sfrlog_cat)
    
    return sfrlog_cat, sfrlog_cat_real



def quench_sfr_MbhSigma(z, lb_time, nhalo, quenched_condition, sfrlog_cat, sigma_sfr, sfr_delay_alpha):
    for ihalo in tqdm(range(nhalo)):
        #sfrlog_cat[ihalo,quenched_condition[ihalo,:]]=-66.
        if z[quenched_condition[ihalo,:]].size>0 and np.max(z[quenched_condition[ihalo,:]])<2:
            z_q = cosmo.lookbackTime( cosmo.lookbackTime(np.max(z[quenched_condition[ihalo,:]])), inverse=True)
            if z_q > 0:
                sfrlog_cat[ihalo,z<z_q]=-66.

        if sfrlog_cat[ihalo,sfrlog_cat[ihalo,:]>-64.].size > 0:
            idx_quench=np.min(np.where(sfrlog_cat[ihalo,:]>-64.)[0])
            dt=lb_time[idx_quench] - lb_time[0:idx_quench]
            sfrlog_cat[ihalo,0:idx_quench] = sfrlog_cat[ihalo,idx_quench]+ np.log10(np.exp(-dt*sfr_delay_alpha)) + np.random.normal(0, sigma_sfr, sfrlog_cat[ihalo,0:idx_quench].size)
    return sfrlog_cat



def SMHM_scatter_from_logms_cat(logmstar_integrated, mhlog_arr, iz, mhlog_smhm, Mstar_range, b, volume, hmf, mhlog, mhlog_min):
    mslog_smhm = np.array([ np.mean(logmstar_integrated[np.logical_and.reduce((logmstar_integrated[:,iz]>0., mhlog_arr[:,iz]>m-b/2., mhlog_arr[:,iz]<=m+b/2.)), iz]) for m in mhlog_smhm ])
    scatter_smhm = np.array([ np.std(logmstar_integrated[np.logical_and.reduce((logmstar_integrated[:,iz]>0., mhlog_arr[:,iz]>m-b/2., mhlog_arr[:,iz]<=m+b/2.)), iz]) for m in mhlog_smhm ])
    if mslog_smhm[np.isfinite(mslog_smhm)].size<2 or scatter_smhm[np.isfinite(scatter_smhm)].size<2:
        return None, None
    scatter = np.mean(scatter_smhm[np.isfinite(scatter_smhm)])

    try:
        popt,pcov=curve_fit(SMHM_double_pl, mhlog_smhm[np.isfinite(mslog_smhm)], mslog_smhm[np.isfinite(mslog_smhm)], p0 = [0.032,12.,2.,0.608])
        mask=np.isfinite(mslog_smhm)
        chi=np.sum( (mslog_smhm[mask]-SMHM_double_pl(mhlog_smhm[mask],*popt) )**2/mslog_smhm[mask].size )
        if chi<0.5:
            smhm = interp1d(mhlog_smhm, SMHM_double_pl(mhlog_smhm,*popt), fill_value="extrapolate")
            return smhm, scatter
        else:
            smhm = interp1d(mhlog_smhm[np.isfinite(mslog_smhm)], mslog_smhm[np.isfinite(mslog_smhm)], fill_value="extrapolate")
            return smhm, scatter
    except:
        smhm = interp1d(mhlog_smhm[np.isfinite(mslog_smhm)], mslog_smhm[np.isfinite(mslog_smhm)], fill_value="extrapolate")
        return smhm, scatter



def compute_subHMF(z_, logmh, HMF):
    #HMF in normal units, not in log
    a = 1./(1.+z_)
    C = np.power(10., -2.415 + 11.68*a - 28.88*a**2 + 29.33*a**3 - 10.56*a**4)
    logMcutoff = 10.94 + 8.34*a - 0.36*a**2 - 5.08*a**3 + 0.75*a**4
    return HMF * C * (logMcutoff - logmh)


def compute_objs_from_mass_function(logm, mf, volume, mask):
    #HMF in normal units, not in log
    MF=mf*volume
    cum_hmf_tot = np.cumsum(MF[mask])
    max_number = np.floor(np.trapz(MF[mask], logm[mask]))
    if (np.random.uniform(0,1) > np.trapz(MF[mask], logm[mask])-max_number): #Calculating number of halos to compute
        max_number += 1
    int_cum_phi = interp1d(cum_hmf_tot, logm[mask])
    range_numbers = np.random.uniform(np.min(cum_hmf_tot), np.max(cum_hmf_tot), int(max_number))
    return int_cum_phi(range_numbers)



def correct_sfrlog_quenched_Sats_missingGals(iz, mhlog_arr, sfrlog_cat, sats_mhlog_cat, delta_mhlog_cat):
    b=0.1; mhlog_smhm=np.arange(10,15.5,0.1)
    f_quench_mhlog = np.zeros(mhlog_smhm.size)
    for i in range(mhlog_smhm.size):
        mask=np.logical_and(mhlog_arr[:,iz]>=mhlog_smhm[i]-b/2., mhlog_arr[:,iz]<mhlog_smhm[i]+b/2.)
        mask_quench=np.logical_and(mask, sfrlog_cat[:,iz]==-66.)
        if mhlog_arr[mask,iz].size>0:
            f_quench_mhlog[i] = mhlog_arr[mask_quench,iz].size / mhlog_arr[mask,iz].size
        else:
            f_quench_mhlog[i] = np.nan
    sats_prob_quench = interp1d(mhlog_smhm, f_quench_mhlog, fill_value="extrapolate") (sats_mhlog_cat[iz])
    delta_prob_quench = interp1d(mhlog_smhm, f_quench_mhlog, fill_value="extrapolate") (delta_mhlog_cat[iz])
    sats_randn = np.random.uniform(0,1,sats_prob_quench.size)
    delta_randn = np.random.uniform(0,1,delta_prob_quench.size)
    mask_sats = np.logical_and.reduce((sats_prob_quench>=0., sats_prob_quench<=1., sats_randn<sats_prob_quench))
    mask_delta = np.logical_and.reduce((delta_prob_quench>=0., delta_prob_quench<=1., delta_randn<delta_prob_quench))
    return mask_sats, mask_delta
