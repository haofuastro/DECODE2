from modules.my_functions import *
from modules.dark_matter_haloes import *
from modules.compute_merger_rates_decode import *

from Mancuso_Lapi_SFRF.SFRF import *


def smf_active(z,nz,ms,nms, work="Weaver+2022"):
    #active
    if work=="Davidzon+2017":
        red_sample=np.array([0.35,0.65,0.95,1.3,1.75,2.25,2.75,3.25,3.75,5.])
        logm_star_sample=np.array([10.26,10.40,10.35,10.42,10.40,10.45,10.39,10.83,10.77,11.3])
        m_star_sample=10.**(logm_star_sample)
        alpha1_sample=np.array([-1.29,-1.32,-1.29,-1.21,-1.24,-1.5,-1.52,-1.78,-1.84,-2.12])
        phi1_sample=np.array([2.41,1.661,1.739,1.542,1.156,0.441,0.441,0.086,0.052,0.003])
        alpha2_sample=np.array([1.01,0.84,0.81,1.11,0.9,0.59,1.05])
        phi2_sample=np.array([1.3,0.86,0.95,0.49,0.46,0.38,0.13])
        smf_dav_ac=np.zeros((len(red_sample),nms))
        for iz in range (len(red_sample)):
            if(red_sample[iz]<3.):
                smf_dav_ac[iz,:]=0.001*np.log(10.)*(ms/m_star_sample[iz])*np.exp(-ms/m_star_sample[iz])*(phi1_sample[iz]*(ms/m_star_sample[iz])**(alpha1_sample[iz])+phi2_sample[iz]*(ms/m_star_sample[iz])**(alpha2_sample[iz]))
            else:
                smf_dav_ac[iz,:]=0.001*np.log(10.)*(ms/m_star_sample[iz])*np.exp(-ms/m_star_sample[iz])*phi1_sample[iz]*(ms/m_star_sample[iz])**(alpha1_sample[iz])
    if work=="Weaver+2022":
        red_sample=np.array([0.35,0.65,0.95,1.3,1.75,2.25,2.75,3.25,4.,5.])
        logm_star_sample=np.array([10.73,10.83,10.91,10.93,10.9,10.84,10.86,10.96,10.57,10.38])
        m_star_sample=10.**(logm_star_sample)
        alpha1_sample=np.array([-1.41,-1.4,-1.36,-1.36,-1.47,-1.47,-1.57,-1.57,-1.57,1.57])
        phi1_sample=np.array([0.8,0.7,0.74,0.66,0.29,0.24,0.19,0.12,0.13,0.1])
        alpha2_sample=np.array([-0.02,-0.31,-0.28,0.41,-0.41,0.01,0.31,0.,0.,0.])
        phi2_sample=np.array([0.49,0.36,0.29,0.11,0.31,0.13,0.04,0.,0.,0.])
        smf_dav_ac=np.zeros((len(red_sample),nms))
        for iz in range(len(red_sample)):
            smf_dav_ac[iz,:]=0.001*np.log(10.)*(ms/m_star_sample[iz])*np.exp(-ms/m_star_sample[iz])*(phi1_sample[iz]*(ms/m_star_sample[iz])**(alpha1_sample[iz])+phi2_sample[iz]*(ms/m_star_sample[iz])**(alpha2_sample[iz]))

    # interpolazione
    smflog_ac=np.zeros((nz,nms))
    for ims in range(nms):
        smflog_ac[:,ims]=interp1d(red_sample,np.log10(smf_dav_ac[:,ims]),bounds_error=False, fill_value="extrapolate")(z)
    # correzione ad alto z
    for iz in range(nearest(z,5)+1,nz):
        for ims in range(nms):
            if smflog_ac[iz,ims]>smflog_ac[iz-1,ims]:
                smflog_ac[iz,ims]=smflog_ac[iz-1,ims]
    smflog_ac[np.isnan(smflog_ac)]=-66.
    return smflog_ac



def smf_passive(z,nz,ms,nms, work="Weaver+2022"):
    #passive
    if work=="Davidzon+2017":
        red_sample=np.array([0.35,0.65,0.95,1.3,1.75,2.25,2.75,3.25,3.75])
        logm_star_sample=np.array([10.83,10.83,10.75,10.56,10.54,10.69,10.24,10.10,10.10])
        m_star_sample=10.**(logm_star_sample)
        alpha1_sample=np.array([-1.3,-1.46,-0.07,0.53,0.93,0.17,1.15,1.15,1.15])
        phi1_sample=np.array([0.098,0.012,1.724,0.757,0.251,0.068,0.028,0.01,0.004])
        alpha2_sample=np.array([-0.39,-0.21])
        phi2_sample=np.array([1.58,1.44])
        smf_dav_pas=np.zeros((len(red_sample),nms))
        for iz in range(len(red_sample)):
            if(red_sample[iz]<0.8):
                smf_dav_pas[iz,:]=0.001*np.log(10.)*(ms/m_star_sample[iz])*np.exp(-ms/m_star_sample[iz])*(phi1_sample[iz]*(ms/m_star_sample[iz])**(alpha1_sample[iz])+phi2_sample[iz]*(ms/m_star_sample[iz])**(alpha2_sample[iz]))
            else:
                smf_dav_pas[iz,:]=0.001*np.log(10.)*(ms/m_star_sample[iz])*np.exp(-ms/m_star_sample[iz])*(phi1_sample[iz]*(ms/m_star_sample[iz])**(alpha1_sample[iz]))

    if work=="Weaver+2022":
        red_sample=np.array([0.35,0.65,0.95,1.3,1.75,2.25,2.75,3.25,4.,5.])
        logm_star_sample=np.array([10.9,10.88,10.89,10.63,10.48,10.49,10.33,10.41,10.58,10.75])
        m_star_sample=10.**(logm_star_sample)
        alpha1_sample=np.array([-0.63,-0.47,-0.47,0.11,0.54,0.66,1.3,1.41,0.95,0.95])
        phi1_sample=np.array([90.92,94.15,95.88,69.9,35.36,11.48,5.65,2.13,1.35,0.29])*0.01
        alpha2_sample=np.array([-1.83,-2.02,-2.02,-2.02,0.,0.,0.,0.,0.,0.])
        phi2_sample=np.array([0.78,0.18,0.11,0.11,-2.02,0.,0.,0.,0.,0.,0.])*0.01
        smf_dav_pas=np.zeros((len(red_sample),nms))
        for iz in range(len(red_sample)):
            smf_dav_pas[iz,:]=0.001*np.log(10.)*(ms/m_star_sample[iz])*np.exp(-ms/m_star_sample[iz])*(phi1_sample[iz]*(ms/m_star_sample[iz])**(alpha1_sample[iz])+phi2_sample[iz]*(ms/m_star_sample[iz])**(alpha2_sample[iz]))

    smflog_pas=np.zeros((nz,nms))
    for ims in range(nms):
        smflog_pas[:,ims]=interp1d(red_sample,np.log10(smf_dav_pas[:,ims]),bounds_error=False, fill_value="extrapolate")(z)
    for iz in range(1,nz):
        for ims in range(nms):
            if smflog_pas[iz,ims]>smflog_pas[iz-1,ims]:
                smflog_pas[iz,ims]=smflog_pas[iz-1,ims]
    smflog_pas[np.isnan(smflog_pas)]=-66.
    smf_pas=10.**smflog_pas
    saved=smflog_pas[0,:]

    return smflog_pas



def sfr_main_sequence(z,nz,mslog,nms,work="popesso+2023"):
    sfrlog_ms=np.zeros((nz,nms))

    #==========================================main sequence speagle+2014
    if work=="speagle+2014":
        sigma_speagle=0.3
        for ired in range(nz):
            sfrlog_ms[ired,:]=(0.84-0.026*cosmo.age(z[ired]))*mslog-(6.51-0.11*cosmo.age(z[ired]))
    #==========================================

    #==========================================main sequence whitaker+2014
    if work=="whitaker+2014":
        red_sample2=np.array([0.75,1.25,1.75,2.25])
        a_sample2=np.array([-27.4,-26.03,-24.04,-19.99])
        b_sample2=np.array([5.02,4.62,4.17,3.44])
        c_sample2=np.array([-0.22,-0.19,-0.16,-0.13])
        a_wh=np.interp(z,red_sample2,a_sample2)
        b_wh=np.interp(z,red_sample2,b_sample2)
        c_wh=np.interp(z,red_sample2,c_sample2)
        for ired in range(nz):
            sfrlog_ms[ired,:]=a_wh[ired]+b_wh[ired]*mslog+c_wh[ired]*mslog*mslog
    #==========================================

    #==========================================main sequence ilbert+2015
    if work=="ilbert+2015":
        a_il=-1.02; beta_il=-0.201; b_il=3.09; sigma_il=0.22
        ssfrlog_il=np.zeros((nz,nms))
        for ired in range(nz):
            ssfrlog_il[ired,:]=a_il+beta_il*10.**(mslog-10.5)+b_il*np.log10(1.+z[ired])
            sfrlog_ms[ired,:]=ssfrlog_il[ired,:]+mslog-9.
    #==========================================

    #==========================================main sequence bethermin+12 (non so se e la stessa di sargent+12, sargent non e chiaro)
    if work=="bethermin+2012":
        ssfrlog_0_bet=-10.2
        beta_bet=-0.2; gamma_bet=3.; zevo=2.5
        ssfrlog_bet=np.zeros((nz,nms))
        for ired in range(nz):
            ssfrlog_bet[ired,:]=ssfrlog_0_bet+beta_bet*(mslog-11.)+gamma_bet*np.log10(1.+min(z[ired],zevo))
            sfrlog_ms[ired,:]=ssfrlog_bet[ired,:]+mslog
    #==========================================

    #==========================================main sequence leja+2015 (Pip's fit)
    if work=="leja+2015":
        for iz in range(nz):
            s0 = 0.6 + 1.22*z[iz] - 0.2*z[iz]*z[iz]
            M0 = np.power(10., 10.3 + 0.753*z[iz] - 0.15*z[iz]*z[iz])
            alpha = 1.3 - 0.1*z[iz]
            sfrlog_ms[iz,:] = s0 - np.log10(1. + np.power(10.**mslog/M0, -alpha))
    #==========================================

    #==========================================main sequence leja+2022
    if work=="leja+2022":
        a = -0.06707 + 0.3684*z - 0.1047*z*z
        b = 0.8552 - 0.101*z - 0.001816*z*z
        c = 0.2148 + 0.8137*z - 0.08052*z*z
        Mt = 10.29 - 0.1284*z + 0.1203*z*z
        for im in range(nms):
            sfrlog_ms[mslog[im]>Mt,im] = a[mslog[im]>Mt]*(mslog[im] - Mt[mslog[im]>Mt]) +c[mslog[im]>Mt]
            sfrlog_ms[mslog[im]<=Mt,im] = b[mslog[im]<=Mt]*(mslog[im] - Mt[mslog[im]<=Mt]) +c[mslog[im]<=Mt]
    #==========================================

    #==========================================main sequence popesso
    if work=="popesso+2023":
        for iz in range(nz):
            sfrlog_ms[iz,:]=(-27.58+0.26*cosmo.age(z[iz]))+(4.95-0.04*cosmo.age(z[iz]))*mslog-0.2*mslog*mslog
    #==========================================

    return np.transpose(sfrlog_ms)



def saunders(logx, lognorm, logxs, alpha, sigma):
    return 10**lognorm * (10**logx/10**logxs)**(1-alpha) * np.exp( - np.log10( 1 + 10**logx/10**logxs )**2 * 0.5 / sigma**2 )


def sfr_function(z,nz,sfrlog,nsfr):
    phisfr, phisfr_etg, phisfr_ltg = SFR_function_UV_IR_Mancuso_Lapi([nz,z,nsfr,sfrlog])# Mancuso, Lapi + 2016
    logphiS = [ [-4.709,0.017,0.018], [-4.688,0.021,0.021], [-5.044,0.027,0.027], [-3.425,0.007,0.007], [-3.414,0.052,0.051], [-4.148,0.111,0.104], [-3.406,0.099,0.102], [-3.917,0.128,0.125] ]
    logsfrS = [ [ 3.049,0.019,0.019], [ 2.753,0.024,0.024], [ 2.744,0.027,0.027], [ 0.712,0.004,0.004], [ 0.367,0.037,0.038], [ 0.336,0.055,0.058], [-0.116,0.069,0.066], [ 0.229,0.083,0.082] ]
    alpha  =  [ [ 1.829,0.002,0.002], [ 1.820,0.002,0.002], [ 1.917,0.002,0.002], [ 1.995,0.003,0.003], [ 2.191,0.007,0.007], [ 2.525,0.054,0.054], [ 2.288,0.028,0.026], [ 2.278,0.039,0.036] ]
    sigma  =  [ [ 0.268,0.006,0.006], [ 0.363,0.009,0.009], [ 0.425,0.012,0.012], [ 0.190,0.001,0.001], [ 0.305,0.014,0.014], [ 0.412,0.000,0.000], [ 0.519,0.019,0.019], [ 0.399,0.024,0.023] ]

    #UV only
    #logphiS[1][0]=logphiS[3][0]; logphiS[2][0]=logphiS[3][0]; logsfrS[1][0]=logsfrS[3][0]; logsfrS[2][0]=logsfrS[3][0]; alpha[1][0]=alpha[3][0]; alpha[2][0]=alpha[3][0]; sigma[1][0]=0.29; sigma[2][0]=0.22

    reds = np.arange(4,11.01,1)
    logphisfr_highz=np.zeros((nsfr,reds.size))
    for ired in range(reds.size):
        logphisfr_highz[:,ired] = np.log10( saunders(sfrlog, logphiS[ired][0], logsfrS[ired][0], alpha[ired][0], sigma[ired][0]) )

    corr=[1.08,1.15,1.25,1.38,1.45]
    idx=nearest(reds,6)
    fit=saunders(sfrlog, logphiS[idx][0], logsfrS[idx][0], alpha[idx][0], sigma[idx][0])
    for ired in range(reds.size):
        if reds[ired]>6.:
            logphisfr_highz[:,ired] = np.log10(fit)*1.2#corr[ired-4]

    for iz in range(nz):
        if z[iz]>3.:
            phisfr[:,iz] = 10**logphisfr_highz[:,nearest(reds,z[iz])]
    return phisfr

def compute_SFR_function_from_LF(nz, z, nLs, logLs, LF, nsfr, sfrlog, volume_mock, logLsmin):
    b=sfrlog[1]-sfrlog[0]
    phisfr=np.zeros((nsfr, nz))
    sfrlog_temp = np.arange(sfrlog[0]-b/2, sfrlog[-1]+b/2+0.1, b)
    #volume_mock = 550**3
    for iz in range(nz):
        logLs_mock = compute_objs_from_mass_function(logLs, LF[:,iz], volume_mock, mask=logLs>logLsmin) #[Lsun]
        #sfrlog_mock = np.log10(3.88e-44) + logLs_mock + logLsun #Murphy+2011
        sfrlog_mock = np.log10(4.55e-44) + logLs_mock + logLsun #Kennicutt+1998
        phisfr[:,iz] = np.histogram(sfrlog_mock, bins=sfrlog_temp)[0] / b / volume_mock
    return phisfr



"""
def sigma_Mstar_relation(mstarlog):
    sigma=np.zeros(mstarlog.size)
    logsigmab=2.073; mblog=10.26
    sigma[mstarlog<=mblog]=logsigmab+0.403*(mstarlog[mstarlog<=mblog]-mblog)
    sigma[mstarlog>mblog]=logsigmab+0.293*(mstarlog[mstarlog>mblog]-mblog)
    return sigma
"""



def assign_velocity_dispersion(nz,nhalo,mhlog_arr,logmstar_integrated,velocity_dispersion):
    if velocity_dispersion=="Ferrarese+2002":
        logVc_arr=np.log10(2.8e-2 * (10**mhlog_arr*h)**0.316) #[km/s] #Formula
        logsigma_arr=1.14*logVc_arr-0.534 #[km/s] #Ferrarese+2002
        #https://ned.ipac.caltech.edu/level5/March02/Ferrarese/Fer5_2.html
        #https://ui.adsabs.harvard.edu/abs/2002ApJ...578...90F/abstract
        #sigma_c normalized to an aperture of size 1/8 the bulge effective radius
    if velocity_dispersion=="Marsden+2022":
        mm=np.loadtxt("Data/Marsden_2022/sigma_evo.txt")
        marsden=interp2d(np.arange(9.5,11.51,0.5), np.arange(0,4.01,0.2), mm)
        logsigma_arr=np.array([[marsden(logmstar_integrated[igal,iz],z[iz])[0] for iz in range(nz)] for igal in range(nhalo)])
        logsigma_arr+=np.random.normal(0.,0.05,logsigma_arr.shape)
    #logsigma_arr = np.transpose([ sigma_Mstar_relation(logmstar_integrated[:,iz]) + np.random.normal(0,0.01,nhalo) for iz in range(nz) ])
    #logsigma_arr[:,z>3] = 100
    return logsigma_arr



def add_mergers_Decode(z,nz,mslog,logmstar_integrated,nhalo,mhlog_arr,sfrlog_cat):
    logMR_cat = compute_merger_rates(z,nz,mslog,logmstar_integrated,nhalo,mhlog_arr)
    logMR_cat[:,z>3]=-65.
    msdotlog_cat = np.log10( (1.-0.44) * 10.**sfrlog_cat + 0.8 * 10.** logMR_cat)
    mstar_integrated = np.flip( np.transpose( cumtrapz(np.flip(10.**np.transpose(msdotlog_cat)), np.flip(cosmo.age(z))*10.**9., axis=0, initial=0.) ) )
    return mstar_integrated, np.log10(mstar_integrated), logMR_cat



def initialize_satellites(nhalo, tree):
    tree["mstar"]=[]; tree["sfr"]=[]
    for ihalo in range(nhalo):
        tree["mstar"].append(np.zeros(tree["zinfall"][ihalo].size)-66.)
        tree["sfr"].append([])
        for isat in range(tree["zinfall"][ihalo].size):
            tree["sfr"][ihalo].append(np.zeros(tree["mhalo"][ihalo][isat][0,:].size )-66.)
    return tree


def add_mergers_SatGen_tree(z,nz,nhalo,mhlog_arr,logmstar_integrated,mstar_integrated,tree):
    mbin=0.1; Mhlog=np.arange(10,15.5,mbin)
    popts=[]
    for iz in range(nz):
        mean_smhm = np.array([ np.mean(logmstar_integrated[np.logical_and.reduce((logmstar_integrated[:,iz]>0., mhlog_arr[:,iz]>m-mbin/2., mhlog_arr[:,iz]<=m+mbin/2.)),iz]) for m in Mhlog ])
        try:
            popt,pcov=curve_fit(SMHM_double_pl, Mhlog[np.isfinite(mean_smhm)], mean_smhm[np.isfinite(mean_smhm)], p0 = [0.032,12.,2.,0.608])
            popts.append(popt)
        except:
            #print(z[iz])
            #popts.append([-10,1,1,1])
            try:
                popts.append(popts[iz-1])
            except:
                popts.append([0.032,12.,2.,0.608])
            #pass
    #print("fit SMHM finished")
    tree["mstar"]=np.array([ np.array([SMHM_double_pl(tree["mhalo"][ihalo][i],*popts[nearest(z,tree["zinfall"][ihalo][i])]) for i in range(tree["mhalo"][ihalo].size)]) for ihalo in range(nhalo) ])

    for ihalo in range(nhalo):
        tree["mstar"][ihalo][np.logical_not(np.isfinite(tree["mstar"][ihalo]))]=-66.

    tree["mratio"]=np.array([ np.array([ 10.**tree["mstar"][ihalo][i]/mstar_integrated[ihalo,nearest(z,tree["z_merge"][ihalo][i])] for i in range(tree["mhalo"][ihalo].size)]) for ihalo in range(nhalo) ])

    merger_history = np.array([[ np.sum(10**tree["mstar"][ihalo][np.logical_and(tree["z_merge"][ihalo]>Z,tree["order"][ihalo]<=1)]) for Z in z] for ihalo in range(nhalo)])
    mstar_integrated += merger_history * 0.8

    return mstar_integrated, np.log10(mstar_integrated), tree





def add_mergers_SatGen_tree_highz(z,nz,nhalo,mhlog_arr,logmstar_integrated,mstar_integrated,tree):
    mbin=0.1; Mhlog=np.arange(10,15.5,mbin)
    red_arr=np.array([5,6,7,8,9])
    smhm_list=[line_across_2_points(Mhlog, 11., 8.6, 12.5, 10.7),
               line_across_2_points(Mhlog, 11., 8.8, 12.5, 11.3),
               line_across_2_points(Mhlog, 10., 7.7, 11.5, 10.25),
               line_across_2_points(Mhlog, 10., 6.9, 11.5, 10.5),
               line_across_2_points(Mhlog, 10., 7.3, 11.5, 11.1)]
    sigma_list=[line_across_2_points(Mhlog, 11., 0.32, 13.6, 0.32),
                line_across_2_points(Mhlog, 11., 0.3, 13.6, 0.0),
                line_across_2_points(Mhlog, 11., 0.18, 13.6, 0.18),
                line_across_2_points(Mhlog, 10., 0.5, 11.6, 0.15),
                line_across_2_points(Mhlog, 10., 0.7, 11.5, 0.)]
    sigma_subs = np.array([ np.array([ interp1d(Mhlog, sigma_list[nearest(red_arr,tree["zinfall"][ihalo][i])], fill_value='extrapolate')(tree["mhalo"][ihalo][i][0,0]) for i in range(len(tree["mhalo"][ihalo]))]) for ihalo in range(nhalo) ])
    tree["mstar"]=np.array([ np.array([ interp1d(Mhlog, smhm_list[nearest(red_arr,tree["zinfall"][ihalo][i])], fill_value='extrapolate')(tree["mhalo"][ihalo][i][0,0]) + np.random.normal(0,sigma_subs[ihalo][i]) for i in range(len(tree["mhalo"][ihalo]))]) for ihalo in range(nhalo) ])
    print("Stellar masses computed")

    for ihalo in range(nhalo):
        tree["mstar"][ihalo][np.logical_not(np.isfinite(tree["mstar"][ihalo]))]=-66.
    
    tree["mratio"]=np.array([ np.array([ 10.**tree["mstar"][ihalo][i]/mstar_integrated[ihalo,nearest(z,tree["z_merge"][ihalo][i])] for i in range(len(tree["mhalo"][ihalo]))]) for ihalo in range(nhalo) ])
    
    merger_history = np.array([[ np.sum(10**tree["mstar"][ihalo][np.logical_and(tree["z_merge"][ihalo]>Z,tree["order"][ihalo]<=1)]) for Z in z] for ihalo in range(nhalo)])

    mstar_integrated_with_mergers = mstar_integrated + merger_history * 0.8
    
    return mstar_integrated_with_mergers, np.log10(mstar_integrated_with_mergers), tree





def apply_mergers_quenching(input_params, dNdVdlogmhdot_active, tree, nhalo, mstar_integrated, sfrlog_cat, z, nz, mhlog_arr, dmhdtlog_arr, mhdotlog, nmhdot, phisfrLF, sfrlog, nsfr, sfrlog_am, sfrlog_ms_sats, mhlog_crit, tnorm, tm, ks, alpha):

    mhdotlog_corr = np.arange(-8, 10, 1)
    phihar_corr = np.zeros(mhdotlog_corr.size) + 1
    dNdVdlogmhdot_active_updated = dNdVdlogmhdot_active.copy()

    tree=initialize_satellites(nhalo, tree)

    for z_am in tqdm(np.arange(3., input_params.z0-0.01, -0.1)):

        for ihalo in range(nhalo):
            for isat in range(tree["zinfall"][ihalo].size):
                if (tree["z_merge"][ihalo][isat] > z_am and tree["z_merge"][ihalo][isat]<=z_am+0.1 ) or z_am==input_params.z0:

                    tree["sfr"][ihalo][isat] = np.array([ sfrlog_am[int(tree["mhalo"][ihalo][isat][4,iz]), int(tree["mhalo"][ihalo][isat][3,iz])] for iz in range(tree["mhalo"][ihalo][isat][0,:].size) ]) + np.random.normal(0,input_params.sigma_sfr,tree["sfr"][ihalo][isat].size)
                    if input_params.include_SNfeedback:
                        tree["sfr"][ihalo][isat][tree["mhalo"][ihalo][isat][0,:] < input_params.logMh_SNfeedback] = -65.
                    tree["mstar"][ihalo][isat] = np.log10( trapz(10**tree["sfr"][ihalo][isat]*(1.-0.44), cosmo.lookbackTime(tree["mhalo"][ihalo][isat][1,:])*1e9) )

                    if tree["mstar"][ihalo][isat]>0.:
                        if input_params.include_sats_stripping:
                            stripped_logmstar = compute_stripped_stellar_mass(z_am, tree["mstar"][ihalo][isat], tree["mhalo"][ihalo][isat][0,0], tree["zinfall"][ihalo][isat], ks)
                        if input_params.include_sats_starformation:
                            tree["mstar"][ihalo][isat] = starformation_afterinfall(z_am, z, mslog, sfrlog_ms_sats, tree["mstar"][ihalo][isat], tree["zinfall"][ihalo][isat], tnorm, tm, alpha)
                        if input_params.include_sats_stripping:
                            tree["mstar"][ihalo][isat] = np.log10(10**tree["mstar"][ihalo][isat]-10**stripped_logmstar)

        merger_history = np.zeros(mstar_integrated.shape)
        for ihalo in range(nhalo):
            for isat in range(tree["zinfall"][ihalo].size):
                if tree["z_merge"][ihalo][isat]>z_am:
                    idx=nearest(z, tree["z_merge"][ihalo][isat])
                    if idx>0: idx-=1
                    merger_history[ihalo,idx:]+=10**tree["mstar"][ihalo][isat]
        mstar_integrated += merger_history * 0.8
        logmstar_integrated = np.log10(mstar_integrated)

        tree["mratio"]=np.array([ np.array([ 10.**tree["mstar"][ihalo][i]/mstar_integrated[ihalo,nearest(z,tree["z_merge"][ihalo][i])] for i in range(tree["zinfall"][ihalo].size)]) for ihalo in range(nhalo) ])

        for ihalo in range(nhalo):
            for isat in range(tree["zinfall"][ihalo].size):
                if tree["z_merge"][ihalo][isat]>z_am and tree["z_merge"][ihalo][isat]<z_am+0.1 and tree["mratio"][ihalo][isat]>0.25:
                    idx=nearest(z, tree["z_merge"][ihalo][isat])
                    if input_params.include_SNfeedback:
                        if mhlog_arr[ihalo,idx] > input_params.logMh_SNfeedback:
                            sfrlog_cat[ihalo,:idx]=-66.
                    else:
                        sfrlog_cat[ihalo,:idx]=-66.
                    break
        mstar_integrated, logmstar_integrated = integrate_accretion_rates_across_time(z, sfrlog_cat+np.log10(1.-0.44))
        logmstar_integrated_NoMerg = logmstar_integrated.copy()
        mstar_integrated += merger_history * 0.8
        logmstar_integrated = np.log10(mstar_integrated)

        for imdot in range(mhdotlog_corr.size):
            mask_tot = np.logical_and.reduce(( dmhdtlog_arr[:,nearest(z,z_am)]>mhdotlog_corr[imdot]-0.5,
                                               dmhdtlog_arr[:,nearest(z,z_am)]<mhdotlog_corr[imdot]+0.5))
            mask_quenched = np.logical_and( sfrlog_cat[:,nearest(z,z_am)]==-66., mask_tot )
            if dmhdtlog_arr[mask_tot,nearest(z,z_am)].size>0 and np.isfinite(dmhdtlog_arr[mask_quenched,nearest(z,z_am)].size / dmhdtlog_arr[mask_tot,nearest(z,z_am)].size):
                phihar_corr[imdot] = dmhdtlog_arr[mask_quenched,nearest(z,z_am)].size / dmhdtlog_arr[mask_tot,nearest(z,z_am)].size

        dNdVdlogmhdot_active_updated[nearest(z,z_am),:] *= 1. - interp1d(mhdotlog_corr, phihar_corr, fill_value="extrapolate")(mhdotlog)
        sfr_am, sfrlog_am = abundance_matching(z,nz,mhdotlog,nmhdot, phisfrLF,sfrlog,nsfr, dNdVdlogmhdot_active_updated, input_params.sigma_sfr, input_params.delay)

        if z_am >input_params.z0:
            sfrlog_cat_temp = sfr_catalogue(z,nz,nhalo,sfrlog_am,mhdotlog,dmhdtlog_arr,mhlog_arr,mhlog_crit, input_params.scatter_mhlog_crit, 1, input_params.delay) + np.random.normal(0, input_params.sigma_sfr,(nhalo,nz))
            sfrlog_cat[sfrlog_cat[:,nearest(z,z_am)]>-66.,nearest(z,z_am)] = sfrlog_cat_temp[sfrlog_cat[:,nearest(z,z_am)]>-66.,nearest(z,z_am)]
            #sfrlog_cat[:,z>4]=-65.
            if input_params.include_SNfeedback:
                for iz in range(nz):
                    sfrlog_cat[mhlog_arr[:,iz] < input_params.logMh_SNfeedback,iz]=-65.
            mstar_integrated, logmstar_integrated = integrate_accretion_rates_across_time(z, sfrlog_cat+np.log10(1.-0.44))
            logmstar_integrated_NoMerg = logmstar_integrated.copy()


    for ihalo in range(nhalo):
        for isat in range(tree["zinfall"][ihalo].size):
            if tree["z_merge"][ihalo][isat] < input_params.z0:

                tree["sfr"][ihalo][isat] = np.array([ sfrlog_am[int(tree["mhalo"][ihalo][isat][4,iz]), int(tree["mhalo"][ihalo][isat][3,iz])] for iz in range(tree["mhalo"][ihalo][isat][0,:].size) ]) + np.random.normal(0,input_params.sigma_sfr,tree["sfr"][ihalo][isat].size)
                if input_params.include_SNfeedback:
                    tree["sfr"][ihalo][isat][tree["mhalo"][ihalo][isat][0,:] < input_params.logMh_SNfeedback] = -65.
                tree["mstar"][ihalo][isat] = np.log10( trapz(10**tree["sfr"][ihalo][isat]*(1.-0.44), cosmo.lookbackTime(tree["mhalo"][ihalo][isat][1,:])*1e9) )

                if tree["mstar"][ihalo][isat]>0.:
                    if input_params.include_sats_stripping:
                        stripped_logmstar = compute_stripped_stellar_mass(input_params.z0, tree["mstar"][ihalo][isat], tree["mhalo"][ihalo][isat][0,0], tree["zinfall"][ihalo][isat], ks)
                    if input_params.include_sats_starformation:
                        tree["mstar"][ihalo][isat] = starformation_afterinfall(input_params.z0, z, mslog, sfrlog_ms_sats, tree["mstar"][ihalo][isat], tree["zinfall"][ihalo][isat], tnorm, tm, alpha)
                    if input_params.include_sats_stripping:
                        tree["mstar"][ihalo][isat] = np.log10(10**tree["mstar"][ihalo][isat]-10**stripped_logmstar)

    return mstar_integrated, logmstar_integrated, logmstar_integrated_NoMerg, sfrlog_cat, sfr_am, sfrlog_am, tree, dNdVdlogmhdot_active_updated




def f_gas_limit_check(z, Mstar):
    Mstar = 10**Mstar
    alpha = 0.59 * (1.+z)**0.45
    Mgas_Mstar = 0.04 * np.power(Mstar / (4.5*10**11), -alpha)
    f_gas = Mgas_Mstar / (Mgas_Mstar + 1.)
    if f_gas > 0.5:
        return True
    else:
        return False
def add_disc_instability(M_disc, z1,z2):
    dt = (cosmo.lookbackTime(z1) - cosmo.lookbackTime(z2)) * 1e9 #[yr]
    Mdot = 25. * (M_disc / 10**11) * np.power((1.+z2)/3., 1.5)
    return Mdot * dt



def evolve_one_bulge_disc(nz,z, tree_mstar,tree_z_merge,tree_mratio, mratio_threshold, logmstar_integrated_ihalo, sfrlog_cat_ihalo, F_discregrowth, add_disc_inst):
    f_discregrowth=F_discregrowth+np.random.normal(0,0.1)
    Mbulge=np.zeros(nz)+1e-66
    Mdisc=np.zeros(nz)+1e-66
    for i in range(19,nz-1):
        iz1=nz-i-1; iz2=nz-i-2
        majormergers=tree_mstar[np.logical_and.reduce((tree_z_merge>=z[iz2],tree_z_merge<z[iz1],tree_mratio>=mratio_threshold))]
        if majormergers.size>0:
            Mdisc[iz2]=f_discregrowth*10.**logmstar_integrated_ihalo[iz2]+1e-66
            Mbulge[iz2]=(1.-f_discregrowth)*10.**logmstar_integrated_ihalo[iz2]
        else:
            Mdisc[iz2] = Mdisc[iz1] + (1.-0.44)* (10.**sfrlog_cat_ihalo[iz1]+10.**sfrlog_cat_ihalo[iz2])/2. * (cosmo.age(z[iz2])-cosmo.age(z[iz1]))*1e9
            minormergers=tree_mstar[np.logical_and.reduce((tree_z_merge>=z[iz2],tree_z_merge<z[iz1],tree_mratio<mratio_threshold))]
            Mbulge[iz2] = Mbulge[iz1] + np.sum(10.**minormergers)
        if add_disc_inst and f_gas_limit_check(z[iz2], logmstar_integrated_ihalo[iz2]):
            Mbulge[iz2] += add_disc_instability(Mdisc[iz2], z[iz1], z[iz2])
            Mdisc[iz2] -= add_disc_instability(Mdisc[iz2], z[iz1], z[iz2])
    return [Mbulge,Mdisc]
def form_evolve_bulge_disc(nz,z,nhalo,mhlog_arr,logmstar_integrated,mstar_integrated,tree,sfrlog_cat,mratio_threshold, f_discregrowth, add_disc_inst):
    res = Parallel(n_jobs=-1)(
    delayed( evolve_one_bulge_disc )( nz,z,tree["mstar"][ihalo],tree["z_merge"][ihalo],tree["mratio"][ihalo],mratio_threshold,logmstar_integrated[ihalo,:],sfrlog_cat[ihalo,:],f_discregrowth,add_disc_inst ) for ihalo in range(nhalo))
    mbulge = np.array([ list(res[ihalo][0]) for ihalo in range(nhalo) ])
    mdisc = np.array([ list(res[ihalo][1]) for ihalo in range(nhalo) ])
    return mbulge,mdisc,np.log10(mbulge),np.log10(mdisc)


def f_gas(LogMs, Z):
    #Steward et al. 2009
    alpha=0.59*(1.+Z)**0.45
    return 0.04 * (10**LogMs/4.5e11)**(-alpha)


def assign_init_size(z, Z, logmstar, f_quenched, Mslog, Type, galaxy_quenched=0):
    if not bool([galaxy_quenched]):
        prob_quenched=interp1d(Mslog, f_quenched[:,nearest(z,Z)], fill_value="extrapolate")(logmstar)
        random_number=np.random.uniform(0,1)
        if random_number<prob_quenched:
            galaxy_quenched=True
        else:
            galaxy_quenched=False
    if Type=="sat":
        #Suess+2019 relation
        if not galaxy_quenched:
            return 0.05*(logmstar-1.2) +np.random.normal(0,0.22)
        elif galaxy_quenched and Z>1.5:
            return 0.3*(logmstar-11.5) +np.random.normal(0,0.22)
        elif galaxy_quenched and Z<=1.5:
            return 0.6*(logmstar-10.5) +np.random.normal(0,0.22)
        #Nedkova+2024 relation
        #if not galaxy_quenched and Z<0.5:
        #    return np.log10(10**0.84*(10**logmstar/5e10)**0.21) +np.random.normal(0,0.22)
        #if not galaxy_quenched and 0.5<=Z<1.:
        #    return np.log10(10**0.78*(10**logmstar/5e10)**0.23) +np.random.normal(0,0.22)
        #if not galaxy_quenched and Z>=1.:
        #    return np.log10(10**0.79*(10**logmstar/5e10)**0.21) +np.random.normal(0,0.22)
        #if galaxy_quenched and Z<0.5:
        #    #return 0.39 - 0.02*logmstar + (0.75+0.02)*np.log10(1+ 10**logmstar/10**10.71 ) +np.random.normal(0,0.22) #bulge
        #    return -0.24 + 0.04*logmstar + (1.82-0.04)*np.log10(1+ 10**logmstar/10**10.94 ) +np.random.normal(0,0.22)
        #if galaxy_quenched and 0.5<=Z<1.:
        #    #return 0.19 - 0.02*logmstar + (2.04+0.02)*np.log10(1+ 10**logmstar/10**10.86 ) +np.random.normal(0,0.22) #bulge
        #    return 0.48 - 0.03*logmstar + (1.6+0.03)*np.log10(1+ 10**logmstar/10**10.95 ) +np.random.normal(0,0.22)
        #if galaxy_quenched and Z>=1.5:
        #    return 3.24 - 0.33*logmstar + (1.25+0.33)*np.log10(1+ 10**logmstar/10**10.63 ) +np.random.normal(0,0.22)
        #if galaxy_quenched and Z>=1.:
        #    #return np.log10(10**(-0.01)*(10**logmstar/5e10)**0.79) +np.random.normal(0,0.22) #bulge
        #    return 3.24 - 0.33*logmstar + (1.25+0.33)*np.log10(1+ 10**logmstar/10**10.63 ) +np.random.normal(0,0.22)
    if Type=="disc":
        #Nedkova+2024 relation
        #return np.log10(10**0.64*(10**logmstar/5e10)**0.17) + +np.random.normal(0,0.22)
        if not galaxy_quenched and Z<0.5:
            return np.log10(10**0.81*(10**logmstar/5e10)**0.23) +np.random.normal(0,0.22)
        if not galaxy_quenched and Z>=0.5:
            return np.log10(10**0.76*(10**logmstar/5e10)**0.21) +np.random.normal(0,0.22)
        if galaxy_quenched and Z<0.5:
            return np.log10(10**0.73*(10**logmstar/5e10)**0.31) +np.random.normal(0,0.22)
        if galaxy_quenched and Z>=0.5:
            return np.log10(10**0.64*(10**logmstar/5e10)**0.30) +np.random.normal(0,0.22)

def update_size(M1,M2,R1,R2,f_orb=0.,c=0.45):
    # Shankar+2014
    return (M1+M2)**2 / ( M1**2/R1 + M2**2/R2 + f_orb/c*M1*M2/(R1+R2) )
    #return (M1*R1 + M2*R2)/(M1+M2)

def evolve_one_galaxy_size(nz,z, tree_mstar,tree_z_merge,tree_mratio, logmstar_integrated_ihalo, sfrlog_cat_ihalo, Mbulge,Mdisc, f_quenched,Mslog, add_gas_dissipation):
    # This function computes the size evolution of one single galaxy
    Rtot=np.zeros(nz)+1e-66
    Rb=np.zeros(nz)+1e-33
    Rd=np.zeros(nz)+1e-66
    first_merger_happened=False
    for i in range(19,nz-1):
        iz1=nz-i-1; iz2=nz-i-2
        mergers=tree_mstar[np.logical_and(tree_z_merge>=z[iz2],tree_z_merge<z[iz1])]
        zmerges=tree_z_merge[np.logical_and(tree_z_merge>=z[iz2],tree_z_merge<z[iz1])]
        mratios=tree_mratio[np.logical_and(tree_z_merge>=z[iz2],tree_z_merge<z[iz1])]
        Rd[iz2]=10.**assign_init_size(z,z[iz2],logmstar_integrated_ihalo[iz2],f_quenched,Mslog,"disc",sfrlog_cat_ihalo[iz2]<-60.)
        if mergers.size>0:
            mergers=np.flip(mergers[np.argsort(zmerges)])
            mratios=np.flip(mratios[np.argsort(zmerges)])
            for isat in range(mergers.size):
                Rsat=10.**assign_init_size(z,z[iz2],mergers[isat],f_quenched,Mslog,"sat")
                if not first_merger_happened:
                    Rb[iz2]=Rsat.copy()
                    first_merger_happened=True
                elif first_merger_happened and mratios[isat]>0.25:
                    if isat==0:
                        Rb[iz2]=update_size(Mbulge[iz2],10.**mergers[isat],Rb[iz1],Rsat)
                    else:
                        Rb[iz2]=update_size(Mbulge[iz2],10.**mergers[isat],Rb[iz2],Rsat)
                    Rd[iz2]=1e-66
                elif first_merger_happened and mratios[isat]<=0.25:
                    #if Rsat>1:
                    #if isat==0:
                    #    Rb[iz2]=update_size(Mbulge[iz2],10.**mergers[isat],Rb[iz1],Rsat)
                    #else:
                    #    Rb[iz2]=update_size(Mbulge[iz2],10.**mergers[isat],Rb[iz2],Rsat)
                    Rd[iz2]=update_size(Mdisc[iz2],10.**mergers[isat],Rd[iz2],Rsat)
                    Rd[iz2]=10.**assign_init_size(z,z[iz2], logmstar_integrated_ihalo[iz2],f_quenched,Mslog,"disc",sfrlog_cat_ihalo[iz2]<-60.)

                if add_gas_dissipation:
                    Mgas1=f_gas(logmstar_integrated_ihalo[iz2], z[iz2])*10**logmstar_integrated_ihalo[iz2]
                    Mgas2=f_gas(mergers[isat], z[iz2])*10.**mergers[isat]
                    F_gas_tot=(Mgas1+Mgas2) / (10**logmstar_integrated_ihalo[iz2]+10.**mergers[isat])
        else:
            Rb[iz2]=Rb[iz1].copy()
        Rtot[iz2]= (Rb[iz2]*Mbulge[iz2] + Rd[iz2]*Mdisc[iz2]) / (Mbulge[iz2]+Mdisc[iz2])
        if add_gas_dissipation and mergers.size>0:
            Rtot[iz2]=Rtot[iz2]/(1.+F_gas_tot/0.25)

    return [Rtot,Rb,Rd]

def evolve_sizes(nz,z,nhalo,mhlog_arr,logmstar_integrated,mstar_integrated,tree,sfrlog_cat,mbulge,mdisc,add_gas_dissipation=False):
    # This function computes the size evolution of the galaxies in the catalogue

    # stellar mass function and fraction of quenched
    Msbin=0.1; Mslog=np.arange(8,12.5,Msbin)
    phi_mslog_tot_cat = np.zeros((Mslog.size,nz))
    phi_mslog_tot_pas = np.zeros((Mslog.size,nz))
    for iz in range(nz):
        Mslog_bins=np.append(Mslog,Mslog[-1]+Msbin)-Msbin/2.
        phi_mslog_tot_cat[:,iz] = np.histogram(logmstar_integrated[:,iz], bins=Mslog_bins)[0]
        phi_mslog_tot_pas[:,iz] = np.histogram(logmstar_integrated[sfrlog_cat[:,iz]<-60,iz], bins=Mslog_bins)[0]
    f_quenched=phi_mslog_tot_pas/phi_mslog_tot_cat

    res = Parallel(n_jobs=-1)(
    delayed( evolve_one_galaxy_size )( nz,z,tree["mstar"][ihalo],tree["z_merge"][ihalo],tree["mratio"][ihalo],logmstar_integrated[ihalo,:],sfrlog_cat[ihalo,:],mbulge[ihalo,:],mdisc[ihalo,:],f_quenched,Mslog, add_gas_dissipation )
    for ihalo in range(nhalo))
    Rtotal = np.array([ list(res[ihalo][0]) for ihalo in range(nhalo) ])
    Rbulge = np.array([ list(res[ihalo][1]) for ihalo in range(nhalo) ])
    Rdisc = np.array([ list(res[ihalo][2]) for ihalo in range(nhalo) ])
    return Rtotal,Rbulge,Rdisc,np.log10(Rtotal),np.log10(Rbulge),np.log10(Rdisc)



def fration_stripped_mstar(f_DM, ks):
    return np.exp(ks * f_DM)

def compute_stripped_stellar_mass(z_am, tree_mstar_ihalo_isat, tree_mhalo_ihalo_isat_00, tree_zinfall_ihalo_isat, ks=-14.2):
    f_DM = 10.**mass_at_t(tree_mhalo_ihalo_isat_00, z_am, tree_zinfall_ihalo_isat)/ 10**tree_mhalo_ihalo_isat_00
    f_strip = fration_stripped_mstar(f_DM, ks)
    return tree_mstar_ihalo_isat+np.log10(f_strip)


def compute_quenching_delay_time(logmstar, tnorm, tm):
    return straight_line(logmstar, tm, tnorm)
    #t_delay=straight_line(11,tm,tnorm)


def sfr_exp_truncation(s0, t_age, t_infall, t_delay, alpha):
    if t_age<t_infall+t_delay:
        return s0
    elif t_age>=t_infall+t_delay:
        return s0* np.exp( (1-t_age/(t_infall+t_delay)) *alpha )


def starformation_afterinfall(z_am, z, mslog, sfrlog_ms_sats, tree_mstar_ihalo_isat, tree_zinfall_ihalo_isat, tnorm, tm, alpha):

    t_delay = compute_quenching_delay_time(tree_mstar_ihalo_isat, tnorm, tm)
    t_temp = np.arange(cosmo.age(tree_zinfall_ihalo_isat), cosmo.age(z_am), 0.1)
    tree_mstar_ihalo_isat=10**tree_mstar_ihalo_isat
    for it in range(t_temp.size):
        if t_temp[it]<cosmo.age(z_am):
            star_formation_rate=sfrlog_ms_sats[nearest(mslog, tree_mstar_ihalo_isat), nearest(z, cosmo.age(t_temp[it],inverse=True))] + np.random.normal(0,0.25,1)
            star_formation_rate_after_delay=star_formation_rate.copy()
        else:
            star_formation_rate=sfr_exp_truncation(star_formation_rate_after_delay, t_temp[it], cosmo.age(tree_zinfall_ihalo_isat), t_delay, alpha)
        tree_mstar_ihalo_isat += star_formation_rate*0.1e9

    return np.log10(tree_mstar_ihalo_isat)





def MANGa_Sizes_Fit(sm, z=0, incGamma = "Marsden"):
    """
    written by C. Marsden
    """
    args = [3.04965733e-03, 9.67029240e-02, -3.91008421e+01, 2.04401878e-01, -4.29464785e+00]
    def gammaFunc(sm, a1, a2, a3, a4, a5):
        return a1 * sm * ((sm * a2) ** a3 + (sm * a4) ** a5) ** -1

    smp = 10**(sm)
    res = (10**-1.02904886) * (smp**0.11888986) * (1 + smp/(10**10.14907206))**0.64171583
    # This is now just a modified version of the RN/Mosleh fit...

    isarray_sm = True if hasattr(sm, "__len__") else  False
    isarray_z = True if hasattr(z, "__len__")  else  False

    if isarray_sm and isarray_z:
        assert len(sm) == len(z), "sm and z are unequal lengths: {} and {} respectively".format(len(sm), len(z))


    if incGamma == "Marsden" or bool(incGamma) == True:
        gamma = gammaFunc(sm, *args)
    elif incGamma == "RN":
        gamma = (1/0.85) * (sm - 10.75)
        gamma[gamma < 0] = 0
    else:
        assert False, "Unregonised type for incGamma {}. Value is: {}".format(str(type(incGamma)), incGamma)

    res = res * (1.+z)**-gamma

    return res

def MANGa_Sersic_Fit(sm, minsm = 8, natmin = 1.6):
    """
    written by C. Marsden
    """
    res = 10**(-0.01903256 * sm**3 +\
                 0.57133231 * sm**2 +\
               -5.48853078 * sm +\
               17.22904271 )
    isarray_sm = True if hasattr(sm, "__len__") else  False

    if isarray_sm:
        res[res < natmin] = natmin
        res[sm < minsm]  = natmin
    else:
        if res < natmin:
            res = natmin
        elif sm < minsm:
            res = natmin

    return res

def LogSigma1kpc(n, ms, Re):
    """
    Xu+2024
    """
    #r=np.arange(0.001,100,0.01)
    r=np.concatenate((np.arange(0.001,10,0.01),np.arange(10,100,0.1)))
    bn = 2.*n - 1./3. + 4./(405.*n) + 46./(25515.*n**2)
    if n<2.5:
        I = np.exp( -bn * r**(1/n) )
        S1kpc = trapz(I[r<1],r[r<1])/trapz(I,r) * ms / np.pi
    else:
        #t=np.arange(1.0001,10,0.01)
        t=np.concatenate((np.arange(1.0001,2,0.01),np.arange(2,10,0.1)))
        rho = np.array([ (R/Re)**(1/n-1) * trapz( np.exp( -bn * (R/Re)**(1/n)*t ) / np.sqrt(t**(2*n) - 1.), t ) for R in r ])
        S1kpc = trapz(rho[r<1]*r[r<1]**2,r[r<1]) / trapz(rho*r**2,r) * ms / np.pi
    return np.log10(S1kpc)




def MgasSantini(Mgal,SFR):
    # P. Santini, R. Maiolino et al. 2014
    N=len(Mgal)
    alpha=np.empty(N)
    beta=np.empty(N)
    mask=np.where(SFR<=0.25)[0]
    if len(mask>0):
        alpha[mask]=-2.17
        beta[mask]=-1.04
    mask=np.where((SFR>0.25) & (SFR<0.50))[0]
    if len(mask>0):
        alpha[mask]=-1.53
        beta[mask]=-0.52
    mask=np.where((SFR>=0.50) & (SFR<0.75))[0]
    if len(mask>0):
        alpha[mask]=-1.34
        beta[mask]=-0.53
    mask=np.where((SFR>=0.75) & (SFR<1.00))[0]
    if len(mask>0):
        alpha[mask]=-1.58
        beta[mask]=-0.85
    mask=np.where((SFR>=1.00) & (SFR<1.20))[0]
    if len(mask>0):
        alpha[mask]=-1.38
        beta[mask]=-0.79
    mask=np.where((SFR>=1.20) & (SFR<1.40))[0]
    if len(mask>0):
        alpha[mask]=-1.34
        beta[mask]=-0.86
    mask=np.where((SFR>=1.40) & (SFR<1.60))[0]
    if len(mask>0):
        alpha[mask]=-1.22
        beta[mask]=-0.77
    mask=np.where((SFR>=1.60) & (SFR<1.80))[0]
    if len(mask>0):
        alpha[mask]=-1.06
        beta[mask]=-0.79
    mask=np.where((SFR>=1.80) & (SFR<2.00))[0]
    if len(mask>0):
        alpha[mask]=-0.96
        beta[mask]=-0.76
    mask=np.where((SFR>=2.00) & (SFR<2.25))[0]
    if len(mask>0):
        alpha[mask]=-0.85
        beta[mask]=-0.82
    mask=np.where((SFR>=2.25) & (SFR<2.50))[0]
    if len(mask>0):
        alpha[mask]=-0.75
        beta[mask]=-0.70
    mask=np.where(SFR>=2.50)[0]
    if len(mask>0):
        alpha[mask]=-0.54
        beta[mask]=-0.50
    logMgal=np.log10(Mgal)
    logfgas=alpha+beta*(logMgal-11.)#+np.random.normal(0,0.1,size=len(logMgal))
    logMgas=logfgas+logMgal-np.log10((1.-10.**logfgas))
    return logMgas

def MgasStewart(Mstar,SFRlog,z):
    alpha = 0.59 * (1. + z)**0.45
    return np.log10(Mstar * 0.04 * ( Mstar / 4.5e11 )**(-alpha))
