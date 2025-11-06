"""
@ kuv_correction_imf_Lapi.py

Provided by Andrea Lapi

"""

import numpy as np
from scipy.integrate import trapz, cumtrapz
from scipy.interpolate import interp1d


def kuv(agelogout,zetaout,imf_mscharlog,imf_slope,imf_type):

    #   Fit to Sandro 2023 stellar tracks

    nms=100

    mslog=np.linspace(np.log10(0.08),np.log10(600.),nms)

    ms=10.**mslog

    zeta=np.array([0.0005,0.001,0.01,0.02])

    nzeta=len(zeta)

    Euvmslog_temp=np.zeros((nms,nzeta))

    Euvmslog_coeff=np.array([3.56205222e+01,1.57081409e+00,-3.96603877e-02,-1.37531595e-01,3.22144582e-02])    # Z=0.0005

    Euvmslog_temp[:,0]=Euvmslog_coeff[0]+Euvmslog_coeff[1]*mslog+Euvmslog_coeff[2]*mslog**2.+Euvmslog_coeff[3]*mslog**3.+Euvmslog_coeff[4]*mslog**4.

    Euvmslog_coeff=np.array([35.77669905,0.72992959,1.24517121,-0.88772042,0.17807522])   # Z=0.001

    Euvmslog_temp[:,1]=Euvmslog_coeff[0]+Euvmslog_coeff[1]*mslog+Euvmslog_coeff[2]*mslog**2.+Euvmslog_coeff[3]*mslog**3.+Euvmslog_coeff[4]*mslog**4.

    Euvmslog_coeff=np.array([34.77322297,3.83082677,-2.34470324,0.84715905,-0.10909963])   # Z=0.01

    Euvmslog_temp[:,2]=Euvmslog_coeff[0]+Euvmslog_coeff[1]*mslog+Euvmslog_coeff[2]*mslog**2.+Euvmslog_coeff[3]*mslog**3.+Euvmslog_coeff[4]*mslog**4.

    Euvmslog_coeff=np.array([34.10637216,5.67966116,-4.19169518,1.62572847,-0.22886165])   # Z=0.02

    Euvmslog_temp[:,3]=Euvmslog_coeff[0]+Euvmslog_coeff[1]*mslog+Euvmslog_coeff[2]*mslog**2.+Euvmslog_coeff[3]*mslog**3.+Euvmslog_coeff[4]*mslog**4.

    Euvmslog=(interp1d(zeta,Euvmslog_temp,fill_value='extrapolate',kind='linear',axis=1))(zetaout)

    Euvms=10.**Euvmslog

    jms=ms<2.

    Euvms[jms]=0.

    if np.any(agelogout<0.):

        restemp=trapz(ms*Euvms*IMF(mslog,imf_mscharlog,imf_slope,imf_type),mslog)/trapz(ms**2.*IMF(mslog,imf_mscharlog,imf_slope,imf_type),mslog)

    else:

        agemslog=agems(mslog,zetaout)

        resnum=cumtrapz(ms*Euvms/10.**agemslog*IMF(mslog,imf_mscharlog,imf_slope,imf_type),mslog,initial=0)/trapz(ms**2.*IMF(mslog,imf_mscharlog,imf_slope,imf_type),mslog)

        res_temp=np.flip(cumtrapz(np.flip(np.abs(np.gradient(agemslog,mslog))*10.**agemslog*resnum),-np.flip(mslog),initial=0)*np.log(10.))

        jres=np.isfinite(res_temp)

        restemp=(interp1d(agemslog[jres],res_temp[jres],fill_value='extrapolate',kind='linear'))(agelogout)

    res=restemp/3.15e7

    return res



def IMF(mslogout,imf_mscharlog,imf_slope,imf_type):

    nms=100

    mslog=np.linspace(np.log10(0.08),np.log10(600.),nms)

    ms=10.**mslog

    restemp=np.zeros(nms)

    if imf_type == 'Salpeter':

        restemp=ms**(-2.35)

    if imf_type == 'Scalo':

        j1 = ms < 1.

        restemp[j1]=ms[j1]**(-1.8)

        j2= np.logical_and(ms>=1,ms<10)

        restemp[j2]=ms[j2]**(-3.25)

        j3 = ms>10

        restemp[j3]=0.16*ms[j3]**(-2.45)

    elif imf_type == 'Kroupa':

        j1 = ms < 0.5

        restemp[j1]=2.*ms[j1]**(-1.3)

        j2= np.logical_and(ms>=0.5,ms<10)

        restemp[j2]=ms[j2]**(-2.3)

        j3=ms>10

        restemp[j3]=10.**0.4*ms[j3]**(-2.7)

    elif imf_type == 'Chabrier':

        j1=ms<1.

        restemp[j1]=3.58/ms[j1]*np.exp(-1.050*(np.log10(ms[j1]/0.079))**2.)

        j2=ms>=1.

        restemp[j2]=ms[j2]**(-2.3)

    elif imf_type == 'SISSA':

        j1=ms<1.

        restemp[j1]=ms[j1]**(-1.4)

        j2=ms>=1.

        restemp[j2]=ms[j2]**(-2.35)

    elif imf_type == 'Larson':

        if imf_mscharlog <= -2.:

            restemp=ms**(imf_slope)

        else:

            restemp=ms**(imf_slope)*np.exp(-(10.**imf_mscharlog/ms))        # versione alla Larson

    elif imf_type == 'Wise':

        if imf_mscharlog <= -2.:

            restemp=ms**(imf_slope)

        else:

            restemp=ms**(imf_slope)*np.exp(-(10.**imf_mscharlog/ms)**1.6)  # versione alla Wise+12, anche usata da Goswami+22

    resnorm=trapz(restemp*ms**2.,mslog)*np.log(10.)

    res=restemp/resnorm

    return (interp1d(mslog,res,fill_value='extrapolate',kind='linear'))(mslogout)



def agems(mslog,zetaout):

    nms=len(mslog)

    zeta=np.array([0.0005,0.001,0.01,0.02])

    nzeta=len(zeta)

    agelog_temp=np.zeros((nms,nzeta))

    age_coeff=np.array([9.79557792,-3.23964917,0.77910623,0.08845161,-0.04035509])

    agelog_temp[:,0]=age_coeff[0]+age_coeff[1]*mslog+age_coeff[2]*mslog**2.+age_coeff[3]*mslog**3.+age_coeff[4]*mslog**4.

    age_coeff=np.array([9.8561528,-3.38621481,0.91448315,0.03416141,-0.03254432])

    agelog_temp[:,1]=age_coeff[0]+age_coeff[1]*mslog+age_coeff[2]*mslog**2.+age_coeff[3]*mslog**3.+age_coeff[4]*mslog**4.

    age_coeff=np.array([10.26292972,-4.21389943,1.49512165,-0.13787199,-0.01398241])

    agelog_temp[:,2]=age_coeff[0]+age_coeff[1]*mslog+age_coeff[2]*mslog**2.+age_coeff[3]*mslog**3.+age_coeff[4]*mslog**4.

    age_coeff=np.array([10.39792464,-4.4614218,1.62400155,-0.15687709,-0.01421212])

    agelog_temp[:,3]=age_coeff[0]+age_coeff[1]*mslog+age_coeff[2]*mslog**2.+age_coeff[3]*mslog**3.+age_coeff[4]*mslog**4.

    agelog=(interp1d(zeta,agelog_temp,fill_value='extrapolate',kind='linear',axis=1))(zetaout)

    return agelog
