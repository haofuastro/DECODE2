from modules.my_functions import *



def Delta_vir(z):
    x = cosmo.Om(z) - 1
    return (18*np.power(np.pi, 2) + 82*x - 39*np.power(x, 2))
def mass_at_t(m, z, z_inf):
    m = np.power(10., m)
    A = 1.54
    dt = cosmo.age(z) - cosmo.age(z_inf)
    G_MPC_Msun_Gyr = -14.34693206746732
    tau = 1.628*(cosmo.h**-1)*np.power(Delta_vir(z)/178, -0.5)*np.power(cosmo.Hz(z)/cosmo.H0, -1)
    tau /= A
    return np.log10( m * np.exp(-dt/tau))



def load_SatGen_halo_catalogue(input_params):

    # read dark matter halo catalogue (parent haloes or central subhaloes)
    # nhalo: number of haloes
    # mhlog_arr: log Mhalo, 2D-array (index_redshift x index_halo)
    # dmhdtlog_arr: log HAR, 2D-array (index_redshift x index_halo)

    #results=np.load("../../SatGen/Daniel/FullCatalogsDifferentRedshifts/SatGen_reduced_catalogue_z=%.2f.npy"%z0,allow_pickle=True)

    results=[]
    loghalo_masses=np.array([])
    for i in range(10):
        if input_params.halo_quenching and (not input_params.mergers_quenching):
            if input_params.z0==6. or input_params.z0==7.:
                results+=list(np.load("../../SatGen/Daniel/FullCatalog_z={:.2f}/SatGen_catalogue_V4_chunck_{:.0f}.npy".format(input_params.z0,i+1), allow_pickle=True))
            else:
                results+=list(np.load("../../SatGen/Daniel/FullCatalog_z={:.2f}/SatGen_catalogue_V3_chunck_{:.0f}.npy".format(input_params.z0,i+1), allow_pickle=True))
        if input_params.mergers_quenching:
            results+=list(np.load("../../SatGen/Daniel/FullCatalog_z={:.2f}/SatGen_catalogue_V2_chunck_{:.0f}.npy".format(input_params.z0,i+1), allow_pickle=True))
        if input_params.z0==6. or input_params.z0==7.:
            loghalo_masses=np.concatenate(( loghalo_masses, np.loadtxt("../../SatGen/Daniel/FullCatalog_z={:.2f}/loghalo_masses_V4_chunck_{:.0f}.txt".format(input_params.z0,i+1)) ))
        else:
            loghalo_masses=np.concatenate(( loghalo_masses, np.loadtxt("../../SatGen/Daniel/FullCatalog_z={:.2f}/loghalo_masses_chunck_{:.0f}.txt".format(input_params.z0,i+1)) ))
    nhalo=len(results)

    #nhalo=results.size
    print(nhalo)

    if input_params.halo_quenching and (not input_params.mergers_quenching):
        Use_zarr = False
        zs = np.array([ results[ihalo][1][0] for ihalo in range(nhalo) ])
        if np.any(np.diff(zs,axis=0) > 0) or Use_zarr:
            print('Different z values - Remapping...')
            MAHs = np.array([interp1d(results[ihalo][1][0],results[ihalo][1][1],bounds_error=False,fill_value=np.NaN)(zarr) for ihalo in tqdm(range(nhalo))])
            MAHs = np.append(np.atleast_2d(zarr),MAHs,axis=0)
        else:
            print('Sampe z values - Combining...')
            MAHs = np.array([ results[ihalo][1][1] for ihalo in range(nhalo) ]) # Mass accretion histories of all centals at z=z0
            MAHs = np.append(np.atleast_2d(results[0][1][0]),MAHs,axis=0) # Include z values as the first row for ease

    if input_params.mergers_quenching:
        MAHs = np.array([ results[ihalo][0][1] for ihalo in range(nhalo) ]) # Mass accretion histories of all centals at z=z0
        MAHs = np.append(np.atleast_2d(results[0][0][0]),MAHs,axis=0) # Include z values as the first row for ease

    if input_params.reduced_catalogue:
        prob_reduced = np.random.uniform(0,1,nhalo)
        mask_reduced = np.array([ p<input_params.reduce_fraction for p in prob_reduced ])
        MAHs_temp=MAHs.copy()
        MAHs=[MAHs_temp[0,:]]
        Results=[]
        for ihalo in range(nhalo):
            if mask_reduced[ihalo]:
                MAHs.append( MAHs_temp[ihalo+1,:] )
                Results.append(results[ihalo])
        results=Results
        MAHs=np.array(MAHs)
        nhalo=MAHs[:,0].size-1
        print(nhalo)

    return nhalo, loghalo_masses, results, MAHs


def compute_SatGen_halo_acc_histories(z, nhalo, MAHs):
    mhlog_arr=np.array([ interp1d(MAHs[0,:],MAHs[ihalo+1,:],fill_value="extrapolate")(z) for ihalo in range(nhalo) ])

    #dmhdtlog_arr=np.loadtxt("../../SatGen/Daniel/FullCatalog_z={:.2f}/loghar_catalogue.txt".format(z0))

    lt_temp = np.arange(cosmo.lookbackTime(0.),13.5,0.4)
    z_temp = cosmo.lookbackTime(lt_temp, inverse=True)

    dmhdtlog_arr = []
    for ihalo in tqdm(range(nhalo)):

        logmh_temp = interp1d(cosmo.lookbackTime(MAHs[0,:]), MAHs[ihalo+1,:], fill_value="extrapolate")(lt_temp)
        #logmh_temp = interp1d(MAHs[0,:], MAHs[ihalo+1,:], fill_value="extrapolate")(z_temp)

        loghar = np.log10( np.flip( np.diff(np.flip(10**logmh_temp))/np.diff(1e9*cosmo.age(np.flip(z_temp))) ) )
        loghar_interp = interp1d( z_temp[:-1][np.isfinite(loghar)]-0.005, loghar[np.isfinite(loghar)] , fill_value="extrapolate" )

        loghar = loghar_interp(z)
        dmhdtlog_arr.append(loghar)
    dmhdtlog_arr=np.array(dmhdtlog_arr)

    return mhlog_arr, dmhdtlog_arr



def load_SatGen_halo_catalogue_highz(input_params, mhlog_Min=11., mhlog_Max=15.5):
    loghalo_masses=np.array([])
    for i in range(10):
        if input_params.z0 <=6:
            loghalo_masses=np.concatenate(( loghalo_masses, np.loadtxt("../../SatGen/Daniel/FullCatalog_z={:.2f}/loghalo_masses_chunck_{:.0f}.txt".format(input_params.z0,i+1)) ))
        else:
            loghalo_masses=np.concatenate(( loghalo_masses, np.loadtxt("../../SatGen/Daniel/FullCatalog_z={:.2f}/loghalo_masses_V4_chunck_{:.0f}.txt".format(input_params.z0,i+1)) ))
    nhalo=loghalo_masses.size
    print("Total number of haloes:", nhalo)
    print("Minimum halo mass in file:", np.min(loghalo_masses))
    
    masses = []
    orders = []
    files = os.listdir("../../SatGen/Daniel/FullCatalog_z={:.2f}/SatGen_catalogue_merger_tree/".format(input_params.z0))
    #files = ["../../SatGen/Daniel/FullCatalog_z={:.2f}/SatGen_catalogue_merger_tree/logmh>12.5.npy".format(input_params.z0)]
    ihalo=0
    Idel=0
    loghalo_masses=list(loghalo_masses)
    Nhalo_orig = 0
    if len(files)>1: chuncks=10; subs=16
    if len(files)==1: chuncks=1; subs=1
    for chunck in tqdm(range(chuncks)):
        for sub in range(subs):
            if len(files)>1: filename="chunck_{:d}_sub_{:d}.npy".format(chunck+1,sub+1)
            if len(files)==1: filename=files[0]
            if filename in files:
                if len(files)>1: df=np.load("../../SatGen/Daniel/FullCatalog_z={:.2f}/SatGen_catalogue_merger_tree/chunck_{:d}_sub_{:d}.npy".format(input_params.z0,chunck+1,sub+1), allow_pickle=True)
                if len(files)==1: df=np.load(filename, allow_pickle=True)
                Masses = list(df[0])
                Orders = list(df[1])
                Nhalo_sub = len(Masses)
                Nhalo_orig+=Nhalo_sub
                if input_params.reduced_catalogue:
                    prob_reduced = np.random.uniform(0,1,Nhalo_sub)
                    mask_reduced = np.array([ p<input_params.reduce_fraction for p in prob_reduced ])
                    idel=0
                    for ihalo_sub in range(Nhalo_sub):
                        if not mask_reduced[ihalo_sub] or Masses[idel].shape==(0,) or loghalo_masses[Idel]<mhlog_Min or loghalo_masses[Idel]>mhlog_Max:
                            del Masses[idel]
                            del Orders[idel]
                            del loghalo_masses[Idel]
                        else:
                            idel+=1
                            Idel+=1
                    masses=masses+Masses
                    orders=orders+Orders
    print("Initial number of haloes:", Nhalo_orig)
    loghalo_masses=np.array(loghalo_masses)
    nhalo=loghalo_masses.size
    print("Reduced number of haloes", nhalo)
    
    mass_accs=[]
    lt_temp = lt_grid#np.arange(cosmo.lookbackTime(0.),13.5,0.01)
    z_temp = cosmo.lookbackTime(lt_temp, inverse=True)
    for ihalo in tqdm(range(nhalo)):
        mass_accs.append([])
        for isub in range(masses[ihalo][:,0].size):
            logmh_temp = interp1d(cosmo.lookbackTime(z_SatGen), masses[ihalo][isub,:], fill_value="extrapolate")(lt_temp)
            loghar = np.log10( np.flip( np.diff(np.flip(10**logmh_temp))/np.diff(1e9*cosmo.age(np.flip(z_temp))) ) )
            if loghar[np.isfinite(loghar)].size>1:
                loghar_interp = interp1d( z_temp[:-1][np.isfinite(loghar)]-0.005, loghar[np.isfinite(loghar)] , fill_value="extrapolate" )
                loghar = loghar_interp(z_grid)
            else:
                loghar = np.zeros(z_grid.size)-66.
            mass_accs[ihalo].append(loghar)
        mass_accs[ihalo]=np.array(mass_accs[ihalo])

    mhlog_arr = np.array([ interp1d(z_SatGen,masses[ihalo][0,:],fill_value='extrapolate')(z_grid) for ihalo in range(nhalo) ])
    dmhdtlog_arr = np.array([ mass_accs[ihalo][0,:] for ihalo in range(nhalo) ])
    
    return nhalo, masses, orders, mass_accs, loghalo_masses, mhlog_arr, dmhdtlog_arr




def load_SatGen_halo_acc_rate_function(halo_quenching, mergers_quenching, mhdotlog, nz, nmh, z):
    z_temp = np.arange(0,11.1,0.25)
    dNdVdlogmhdot_active_temp=np.zeros((z_temp.size,nmhdot))
    for i,Z in enumerate(z_temp):
        if halo_quenching and (not mergers_quenching):
            logphihar = np.loadtxt("../../SatGen/Daniel/HARfunctions/HAR_function_haloquench_z_%.2f.txt"%Z)
        if mergers_quenching:
            #logphihar = np.loadtxt("../../SatGen/Daniel/HARfunctions/HAR_function_total_z_%.2f.txt"%Z)
            logphihar = np.loadtxt("../../SatGen/Daniel/HARfunctions/HAR_function_haloquench_z_%.2f.txt"%Z)

        #logphihar = np.loadtxt("../../SatGen/Daniel/HARfunctions/HAR_function_total_z_%.2f.txt"%Z)

        dNdVdlogmhdot_active_temp[i,:] = interp1d(logphihar[:,0], logphihar[:,1], fill_value="extrapolate")(mhdotlog)

    dNdVdlogmhdot_active=np.zeros((nz,nmh))
    for im in range(nmh):
        dNdVdlogmhdot_active[:,im] = 10**interp1d(z_temp, dNdVdlogmhdot_active_temp[:,im], fill_value="extrapolate")(z)

    return dNdVdlogmhdot_active



def dark_matter_halo_catalogue(reduced_catalogue,nz):
    if reduced_catalogue:
        nhalo=9908
        mhlog_arr=np.loadtxt("Dark_matter_halo_catalogue/mhlog_arr_z8_>11_reduced.txt").reshape(nhalo,nz)
        dmhdtlog_arr=np.loadtxt("Dark_matter_halo_catalogue/dmhdtlog_arr_z8_>11_reduced.txt").reshape(nhalo,nz)
    else:
        nhalo=100000
        mhlog_arr=np.loadtxt("Dark_matter_halo_catalogue/mhlog_arr_z8_>11.txt").reshape(nhalo,nz)
        dmhdtlog_arr=np.loadtxt("Dark_matter_halo_catalogue/dmhdtlog_arr_z8_>11.txt").reshape(nhalo,nz)
    return nhalo, mhlog_arr, dmhdtlog_arr



def halo_mass_function(z0,z,nz,nhalo,mhlog_arr,mhlog,nmh,mhlog_min,sigma_am=0.):
    mh=10.**mhlog
    hmflog=np.zeros((nmh,nz))
    for iz in range(nz):
        hmflog[:,iz] = np.log10(mass_function.massFunction(mh*h,z[iz],mdef='sp-apr-mn',model = 'diemer20',q_out = 'dndlnM')*np.log(10.)*h**3.)
        #hmflog[:,iz] = np.log10(hgm.halo_galaxies_massfunc(mhlog,z[iz])*mh*np.log(10.))
    #hmflog[:,99]=hmflog[:,98]
    hmf=10.**hmflog

    if sigma_am==0.:
        hmf_cum=np.zeros((nmh,nz))
        for iz in range(nz):
            hmf_cum[:,iz]=trapz(hmf[:,iz],mhlog)-cumtrapz(hmf[:,iz],mhlog,initial=0.)
        hmf_cum[hmf_cum<0.]=10.**(-66.)
    elif sigma_am>0.:
        hmf_cum=np.zeros((nmh,nz))
        for iz in range(nz):
            for imh in range(nmh):
                hmf_cum[imh,iz]=trapz(hmf[:,iz]*0.5*special.erfc((mhlog[imh]-mhlog)/np.sqrt(2.)/sigma_am),mhlog)
        hmf_cum[hmf_cum<0.]=10.**(-66.)

    #questa norm è il numero di aloni reali sopra mhlog_min ad ogni z
    norm=np.zeros((nz))
    for iz in range(nz):
        norm[iz]=np.trapz(hmf[mhlog>mhlog_min,iz],mhlog[mhlog>mhlog_min])
    norm=hmf_cum[nearest(mhlog,mhlog_min),:]

    volume = nhalo / trapz(hmf[nearest(mhlog,11):,nearest(z,z0)], mhlog[nearest(mhlog,11):]) #Mpc^-3
    cube_side = np.cbrt(volume) #Mpc
    print("Volume cube side: %f Mpc"%cube_side)

    hmf_cat = np.zeros((nmh,nz))
    for iz in range(nz):
        hmf_cat[:,iz] = interp1d(mhlog[:-1]+(mhlog[1]-mhlog[0])/2., np.histogram(mhlog_arr[:,iz],bins=mhlog)[0]/(mhlog[1]-mhlog[0])/volume, fill_value="extrapolate")(mhlog)

    return hmf, np.log10(hmf), hmf_cum, norm, volume, cube_side, hmf_cat



def halo_mass_crit_quench_givenparams(z_crit, m, mhlogcrit0, halo_quenching):
    if not halo_quenching:
        mhlogcrit0=16.
    z=np.arange(0,15,0.05)
    mhlogcrit=np.zeros(z.size)
    mhlogcrit[z<z_crit]=mhlogcrit0
    mhlogcrit[z>=z_crit] = m * (z[z>=z_crit]-z_crit) + mhlogcrit0
    return interp1d(z,mhlogcrit,fill_value="extrapolate")



def unpack_SatGen_merger_tree(z,nz,nhalo,results,mhlog_arr, mhdotlog, grow_satellites=False):
    # tree["mhalo"] cols: 1) logmsubhalo 2) z before infall 3) acc. rate before infall 4) z indexes
    tree={"mhalo":[], "zinfall":[], "order":[]}
    if grow_satellites:
        lt_temp = np.arange(cosmo.lookbackTime(0.),13.5,0.2)
        z_temp = cosmo.lookbackTime(lt_temp, inverse=True)
        zarr=results[0][0][0,:]
        for ihalo in tqdm(range(nhalo)):
            tree["mhalo"].append([])
            tree["zinfall"].append(np.array([]))
            for isat in range(results[ihalo][1][:,0].size):
                mask=np.isfinite( results[ihalo][1][isat,:] )
                if zarr[mask].size>2:
                    logmh_temp = interp1d(cosmo.lookbackTime(zarr[mask]), results[ihalo][1][isat,mask], fill_value="extrapolate")(lt_temp)
                    loghar = np.log10( np.flip( np.diff(np.flip(10**logmh_temp))/np.diff(1e9*cosmo.age(np.flip(z_temp))) ) )
                    if loghar[np.isfinite(loghar)].size>2:
                        loghar_interp = interp1d( z_temp[:-1][np.isfinite(loghar)], loghar[np.isfinite(loghar)], fill_value="extrapolate" )
                        if grow_satellites:
                            z_indexes = np.array([ nearest(z,Z) for Z in zarr[mask] ])
                            loghar_interpsss = loghar_interp(zarr[mask])
                            loghar_indexes = np.array([ nearest(mhdotlog, MHDOT) for MHDOT in loghar_interpsss ])
                            tree["mhalo"][ihalo].append( np.array([results[ihalo][1][isat,mask], zarr[mask], loghar_interpsss, z_indexes, loghar_indexes]) )
                        elif not grow_satellites:
                            tree["mhalo"][ihalo].append( np.array([results[ihalo][1][isat,mask], zarr[mask], loghar_interp(zarr[mask]) ]) )
                        tree["zinfall"][ihalo]=np.append(tree["zinfall"][ihalo], np.min(zarr[mask]))
    elif not grow_satellites:
        for ihalo in range(nhalo):
            tree["mhalo"].append(results[ihalo][0][0,:])
            tree["zinfall"].append(results[ihalo][0][1,:])
    for ihalo in range(nhalo):
        tree["order"].append(np.zeros(results[ihalo][0][0,:].size)+1)
    tree["tau_dyn"]=np.array([ np.array([ compute_dyn_friction_timescale(tree["zinfall"][ihalo][i]) for i in range(tree["zinfall"][ihalo].size)]) for ihalo in range(nhalo) ])
    print("Dynamical friction timescales computed.")
    if grow_satellites:
        tree["mhaloratio"]=np.array([ np.array([ 10.**(tree["mhalo"][ihalo][i][0,0]-mhlog_arr[ihalo,nearest(z,tree["zinfall"][ihalo][i])]) for i in range(tree["zinfall"][ihalo].size)]) for ihalo in range(nhalo) ])
    else:
        tree["mhaloratio"]=np.array([ np.array([ 10.**(tree["mhalo"][ihalo][i]-mhlog_arr[ihalo,nearest(z,tree["zinfall"][ihalo][i])]) for i in range(tree["zinfall"][ihalo].size)]) for ihalo in range(nhalo) ])
    print("Mass ratios computed.")
    tree["fudge"]=0.00035035*tree["mhaloratio"]+0.65
    print("Fudge factor computed.")
    tree["orb_circ"]=np.array([ np.array([ np.random.normal(0.5,0.23) for i in range(tree["zinfall"][ihalo].size)]) for ihalo in range(nhalo) ])
    tree["orb_energy"]=np.array([ np.array([ np.power(tree["orb_circ"][ihalo][i],2.17)/(1.-np.sqrt(1.-tree["orb_circ"][ihalo][i]**2.)) for i in range(tree["zinfall"][ihalo].size)]) for ihalo in range(nhalo) ])
    tree["tau_merge"]=np.array([ np.array([ tree["fudge"][ihalo][i]* tree["tau_dyn"][ihalo][i]*0.9*tree["mhaloratio"][ihalo][i] / np.log(1.+tree["mhaloratio"][ihalo][i]) * np.exp(0.6*tree["orb_circ"][ihalo][i]) * np.power(tree["orb_energy"][ihalo][i],0.1) for i in range(tree["zinfall"][ihalo].size)]) for ihalo in range(nhalo) ])
    print("Merging timescales computed")
    tree["age_merge"] = np.array([ np.array([ cosmo.age(tree["zinfall"][ihalo][i])+tree["tau_merge"][ihalo][i] for i in range(tree["zinfall"][ihalo].size)]) for ihalo in range(nhalo) ])
    tree["z_merge"] = np.array([ np.array([ -1. for i in range(tree["zinfall"][ihalo].size)]) for ihalo in range(nhalo) ])
    age_today=cosmo.age(0.)
    for ihalo in range(nhalo):
        tree["z_merge"][ihalo][tree["age_merge"][ihalo]<age_today] = cosmo.age(tree["age_merge"][ihalo][tree["age_merge"][ihalo]<age_today], inverse=True)
    print("Redshift at merging computed.")

    return tree



def unpack_SatGen_merger_tree_highz(z,nz,nhalo,masses,mass_accs,orders,mhlog_arr, mhdotlog, zarr):
    # tree["mhalo"] cols: 1) logmsubhalo 2) z before infall 3) acc. rate before infall 4) z indexes
    tree={"mhalo":[], "zinfall":[], "order":[]}
    lt_temp = lt_grid#np.arange(cosmo.lookbackTime(0.),13.5,0.2)
    z_temp = cosmo.lookbackTime(lt_temp, inverse=True)
    
    for ihalo in tqdm(range(nhalo)):
        tree["mhalo"].append([])
        tree["zinfall"].append(np.array([]))
        tree["order"].append([])
        
        for isat in range(masses[ihalo][:,0].size):
            
            mask=np.isfinite( masses[ihalo][isat,:] )
            if zarr[mask].size>2 and np.max(orders[ihalo][isat,:])!=0:
                
                logmh_temp = interp1d(cosmo.lookbackTime(zarr[mask]), masses[ihalo][isat,mask], fill_value="extrapolate")(lt_temp)
                loghar = np.log10( np.flip( np.diff(np.flip(10**logmh_temp))/np.diff(1e9*cosmo.age(np.flip(z_temp))) ) )
                if loghar[np.isfinite(loghar)].size>2:
                    loghar_interp = interp1d( z_temp[:-1][np.isfinite(loghar)], loghar[np.isfinite(loghar)], fill_value="extrapolate" )
                    z_indexes = np.array([ nearest(z,Z) for Z in zarr[mask] ])
                    loghar_interpsss = loghar_interp(zarr[mask])
                    loghar_indexes = np.array([ nearest(mhdotlog, MHDOT) for MHDOT in loghar_interpsss ])
                    tree["mhalo"][ihalo].append( np.array([masses[ihalo][isat,mask], zarr[mask], loghar_interpsss, z_indexes, loghar_indexes]) )
                    tree["zinfall"][ihalo]=np.append(tree["zinfall"][ihalo], np.min(zarr[mask]))
                    tree["order"][ihalo]=np.append(tree["order"][ihalo], np.max(orders[ihalo][isat,:]))

    for ihalo in tqdm(range(nhalo)):
        tree["order"].append( np.array([ np.max(orders[ihalo][isat,:]) for isat in range(orders[ihalo][:,0].size) ]) )
    tree["tau_dyn"]=np.array([ np.array([ compute_dyn_friction_timescale(tree["zinfall"][ihalo][i]) for i in range(tree["zinfall"][ihalo].size)]) for ihalo in range(nhalo) ])
    print("Dynamical friction timescales computed.")
    
    tree["mhaloratio"]=np.array([ np.array([ 10.**(tree["mhalo"][ihalo][i][0,0]-mhlog_arr[ihalo,nearest(z,tree["zinfall"][ihalo][i])]) for i in range(tree["zinfall"][ihalo].size)]) for ihalo in range(nhalo) ])
    print("Mass ratios computed.")
    
    tree["fudge"]=0.00035035*tree["mhaloratio"]+0.65
    tree["fudge"]*=1.5
    print("Fudge factor computed.")
    tree["orb_circ"]=np.array([ np.array([ np.random.normal(0.5,0.23) for i in range(tree["zinfall"][ihalo].size)]) for ihalo in range(nhalo) ])
    tree["orb_energy"]=np.array([ np.array([ np.power(tree["orb_circ"][ihalo][i],2.17)/(1.-np.sqrt(1.-tree["orb_circ"][ihalo][i]**2.)) for i in range(tree["zinfall"][ihalo].size)]) for ihalo in range(nhalo) ])
    tree["tau_merge"]=np.array([ np.array([ tree["fudge"][ihalo][i]* tree["tau_dyn"][ihalo][i]*0.9*tree["mhaloratio"][ihalo][i] / np.log(1.+tree["mhaloratio"][ihalo][i]) * np.exp(0.6*tree["orb_circ"][ihalo][i]) * np.power(tree["orb_energy"][ihalo][i],0.1) for i in range(tree["zinfall"][ihalo].size)]) for ihalo in range(nhalo) ])
    #tree["tau_merge"]=np.array([ np.array([ 0. for i in range(tree["zinfall"][ihalo].size)]) for ihalo in range(nhalo) ])
    print("Merging timescales computed")
    tree["age_merge"] = np.array([ np.array([ cosmo.age(tree["zinfall"][ihalo][i])+tree["tau_merge"][ihalo][i] for i in range(tree["zinfall"][ihalo].size)]) for ihalo in range(nhalo) ])
    tree["z_merge"] = np.array([ np.array([ -1. for i in range(tree["zinfall"][ihalo].size)]) for ihalo in range(nhalo) ])
    age_today=cosmo.age(0.)
    for ihalo in range(nhalo):
        tree["z_merge"][ihalo][tree["age_merge"][ihalo]<age_today] = cosmo.age(tree["age_merge"][ihalo][tree["age_merge"][ihalo]<age_today], inverse=True)
    print("Redshift at merging computed.")

    return tree



def read_SatGen_merger_tree(reduced_catalogue,nz,z,nhalo,mhlog_arr):
    if reduced_catalogue:
        tree=np.load("Dark_matter_halo_catalogue/SatGen_merger_trees/SatGen_tree_reduced.npy", allow_pickle=True)
    else:
        tree=np.load("Dark_matter_halo_catalogue/SatGen_merger_trees/SatGen_tree.npy", allow_pickle=True)
    tree={"mhalo":tree[0,:], "zinfall":tree[1,:], "order":tree[2,:]}

    #x=np.array([ np.array([ cosmo.Om(tree["zinfall"][ihalo][i]) for i in range(tree["mhalo"][ihalo].size)]) for ihalo in range(nhalo) ]) -1.
    #tree["Dvir"]=18.*np.pi**2.+82.*x-39.*x**2.
    #HZ=np.array([ np.array([ cosmo.Hz(tree["zinfall"][ihalo][i]) for i in range(tree["mhalo"][ihalo].size)]) for ihalo in range(nhalo) ])
    #sqrtDvir178=np.array([ np.array([ np.sqrt(tree["Dvir"][ihalo][i]/178.) for i in range(tree["mhalo"][ihalo].size)]) for ihalo in range(nhalo) ])
    #tree["tau_dyn"]=1.628 / h / sqrtDvir178 * (cosmo.H0/HZ)
    tree["tau_dyn"]=np.array([ np.array([ compute_dyn_friction_timescale(tree["zinfall"][ihalo][i]) for i in range(tree["mhalo"][ihalo].size)]) for ihalo in range(nhalo) ])
    print("Dynamical friction timescales computed.")
    tree["mhaloratio"]=np.array([ np.array([ 10.**(tree["mhalo"][ihalo][i]-mhlog_arr[ihalo,nearest(z,tree["zinfall"][ihalo][i])]) for i in range(tree["mhalo"][ihalo].size)]) for ihalo in range(nhalo) ])
    print("Mass ratios computed.")
    tree["fudge"]=0.00035035*tree["mhaloratio"]+0.65
    print("Fudge factor computed.")
    tree["orb_circ"]=np.array([ np.array([ np.random.normal(0.5,0.23) for i in range(tree["mhalo"][ihalo].size)]) for ihalo in range(nhalo) ])
    tree["orb_energy"]=np.array([ np.array([ np.power(tree["orb_circ"][ihalo][i],2.17)/(1.-np.sqrt(1.-tree["orb_circ"][ihalo][i]**2.)) for i in range(tree["mhalo"][ihalo].size)]) for ihalo in range(nhalo) ])
    tree["tau_merge"]=np.array([ np.array([ tree["tau_dyn"][ihalo][i]*0.9*tree["mhaloratio"][ihalo][i] / np.log(1.+tree["mhaloratio"][ihalo][i]) * np.exp(0.6*tree["orb_circ"][ihalo][i]) * np.power(tree["orb_energy"][ihalo][i],0.1) for i in range(tree["mhalo"][ihalo].size)]) for ihalo in range(nhalo) ])
    print("Merging timescales computed")
    tree["age_merge"] = np.array([ np.array([ cosmo.age(tree["zinfall"][ihalo][i])+tree["tau_merge"][ihalo][i] for i in range(tree["mhalo"][ihalo].size)]) for ihalo in range(nhalo) ])
    tree["z_merge"] = np.array([ np.array([ -1. for i in range(tree["mhalo"][ihalo].size)]) for ihalo in range(nhalo) ])
    age_today=cosmo.age(0.)
    for ihalo in range(nhalo):
        tree["z_merge"][ihalo][tree["age_merge"][ihalo]<age_today] = cosmo.age(tree["age_merge"][ihalo][tree["age_merge"][ihalo]<age_today], inverse=True)
    print("Redshift at merging computed.")

    return tree
