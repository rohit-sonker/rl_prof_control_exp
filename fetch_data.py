# This script is developped by Aza Jalalvand fetch DIII-D using gadata package (2025)


import MDSplus
from gadata import gadata
#import matplotlib.pyplot as plt
import h5py
import pickle
import numpy as np
import time
# from tqdm import tqdm
import sys
import os
from omfit_classes.omfit_eqdsk import OMFITgeqdsk
from glob import glob
import string
import random

def data2dict(shotn, signame, hf, atlconn) :
    dict_group = hf.create_group(str(signame))
    try:
        data = gadata(signame, shotn, connection=atlconn)
        dict_group['xdata'] = data.xdata
        dict_group['ydata'] = data.ydata
        dict_group['zdata'] = data.zdata
        dict_group['xunits'] = str(data.xunits)
        dict_group['yunits'] = str(data.yunits)
        dict_group['zunits'] = str(data.zunits)
    except: 
        print('%s not available, filled with NULL!' % (signame))
        dict_group['xdata'] = []
        dict_group['ydata'] = []
        dict_group['zdata'] = []
        dict_group['xunits'] = []
        dict_group['yunits'] = []
        dict_group['zunits'] = []
        del atlconn
        #global atlconn
        atlconn = MDSplus.Connection('atlas.gat.com')
        pass
    return atlconn

def generate_random_string(length):
    """
    Generates a random string of a specified length containing
    uppercase letters, lowercase letters, and digits.
    """
    characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(characters) for i in range(length))
    return random_string

atlconn = MDSplus.Connection('atlas.gat.com')
ech_gytname = ['lei','luk','r2d']

# shot_list = np.loadtxt('/cscratch/jalalvanda/shared/image_data/shotlist.txt',delimiter=',',dtype=np.int32)[200:]
# shot_list = np.random.choice(shot_list, size=100, replace=False)
# np.savetxt('DIII-D-CAKE-Shotnumbers_random100.txt',shot_list,fmt='%d', delimiter='\n')
# shot_list = np.load('tm-control-shots.npy');shot_list=np.unique(shot_list).astype(np.int)
shot_list = [np.int32(sys.argv[1])]
# shot_list=[200437]
# shot_list=np.arange(204367,204373)


beam_ptnames=[]
for pt in ['p','t']:
    for ang in [15,21,30,33]:
        for lr in ['l','r']:
            beam_ptnames.append(f'{pt}injf_{ang}{lr}')
            beam_ptnames.append(f'bms{pt}inj{ang}{lr}')
            beam_ptnames.append(f'{pt}inj_{ang}{lr}')

fs_ptnames=['fs%02d' % (k) for k in range(8)]+['pcphd02','pcphd03']

basic_list = ['aminor','alpha','bcoil', 'betan', 'bmspinj', 'bmspinj30l', 'bmspinj30r', 'bmspinj33l', 'bmspinj33r', 'bmspinj15l', 'bmspinj15r', 'bmspinj21l', 'bmspinj21r','bmstinj', 'bt', 'dssdenest', 'edensfit', 'etempfit', 'fzns', 'ip', 'ipsip', 'iptipp', 'neutronsrate', 'pcbcoil', 'pinj', 'plasticfix', 'pnbi', 'pres', 'q', 'q95', 'qmin','tinj', 'n1rms','dstdenp', 'n2rms', 'r0', 'kappa', 'tritop', 'tribot', 'triangularity_u','triangularity_l', 'gapin', 'psirz', 'psin', 'rhovn', 'irtvpitr2']

actu_list = {
'nbi':beam_ptnames
,'ech':['pech','echpwrc','echpwr']+['ec%sfpwrc' % (x) for x in ech_gytname]+['ec%sxmfrac' % (x) for x in ech_gytname]+['ec%spolang' % (x) for x in ech_gytname]
,'gas':[f'gas{x}' for x in ['a','b','c']]+[f'gas{x}_cal' for x in ['a','b','c']]
}

### gfile parameters begins ###

gfile_dict_keys=['RCENTR', 'NH', 'NW', 'RDIM', 'SIMAG', 'SIBRY', 'CURRENT', 'RLEFT', 'ZMID', 'RLIM', 'ZLIM', 'KVTOR', 'RVTOR', 'NMASS', 'ZDIM', 'RMAXIS', 'ZMAXIS', 'BCENTR', 'FPOL', 'PRES', 'FFPRIM', 'PPRIME', 'PSIRZ', 'QPSI', 'NBBBS', 'LIMITR', 'RHOVN']

gfile_aux_keys=['R', 'Z', 'PSI', 'PSI_NORM', 'PSIRZ', 'PSIRZ_NORM', 'RHOp', 'RHOpRZ', 'FPOLRZ', 'PRESRZ', 'QPSIRZ', 'FFPRIMRZ', 'PPRIMERZ', 'PRES0RZ', 'Br', 'Bz', 'Bp', 'Bt', 'Jr', 'Jz', 'Jt', 'Jp', 'Jt_fb', 'Jpar', 'PHI', 'PHI_NORM', 'PHIRZ', 'RHOm', 'RHO', 'RHORZ', 'Rx1', 'Zx1', 'Rx2', 'Zx2']

gfile_flux_keys=['R0','Z0','RCENTR','BCENTR','CURRENT','R0_interp','Z0_interp']

gfile_time = np.arange(0,6000,20)
efit_type = 'EFIT01'
### gfile parameters ends ###



basics = 0
actu = 0
ece= 1 
ts = 0
co2= 1
co2_pcs = 1
cer= 0
mse= 0
fs = 1
bes= 0
magnetics = 0
fida = 0
gfile = 0

main_path = '/cscratch/jalalvanda/outputs/'

for shotn in shot_list:

    t1=time.time()
    print('Shot #%d'%(shotn,))
    shot_path = os.path.join(main_path,f'{shotn}')
    os.makedirs(shot_path, exist_ok=True)
    if basics:
        hf = h5py.File(os.path.join(shot_path, f'{shotn}_basics.h5'),'w')
        for signame in basic_list:
            atlconn=data2dict(shotn,signame,hf,atlconn)
        hf.close()

    if actu:
        for grpname,signals in actu_list.items():
            hf = h5py.File(os.path.join(shot_path, f'{shotn}_{grpname}.h5'),'w')
            for signame in signals:
                atlconn=data2dict(shotn,signame,hf,atlconn)
            hf.close()
    if ece:
        hf = h5py.File(os.path.join(shot_path, f'{shotn}_ece.h5'),'w')
        pece_group = hf.create_group('pcece')	
        ece_group = hf.create_group('ece')
        rtece_group = hf.create_group('rtece')

        for k in range(48):
            pece_data = gadata('pcece%d' % (k+1), shotn, connection=atlconn)
            pece_group['pcece%02d' % (k+1)] = pece_data.zdata
            ece_data = gadata('tecef%02d' % (k+1), shotn, connection=atlconn)
            ece_group['tecef%02d' % (k+1)] = ece_data.zdata

            #rtece_data = gadata('rcsece%d' % (k+1), shotn, connection=atlconn)
            #rtece_group['rcsece%d' % (k+1)] = rtece_data.zdata

        pece_group['xdata'] = pece_data.xdata
        pece_group['ydata'] = pece_data.ydata
        pece_group['xunits'] = str(pece_data.xunits)
        pece_group['yunits'] = str(pece_data.yunits)
        pece_group['pceceunits'] = str(pece_data.zunits)

        ece_group['xdata'] = ece_data.xdata
        ece_group['ydata'] = ece_data.ydata
        ece_group['xunits'] = str(ece_data.xunits)
        ece_group['yunits'] = str(ece_data.yunits)
        ece_group['eceunits'] = str(ece_data.zunits)

        #rtece_group['xdata'] = rtece_data.xdata
        #rtece_group['ydata'] = rtece_data.ydata
        #rtece_group['xunits'] = rtece_data.xunits
        #rtece_group['yunits'] = rtece_data.yunits
        #rtece_group['rteceunits'] = rtece_data.zunits
        hf.close()

    if ts:
        hf = h5py.File(os.path.join(shot_path, f'{shotn}_ts.h5'),'w')
        for corediv in ['core','tan','div']:
            for nete in ['ne','te']:
                try: 
                    ts_data = gadata('ts%s_%s' % (nete,corediv), shotn, connection=atlconn)

                    ts_group = hf.create_group('ts%s_%s' % (nete,corediv))

                    ts_group['ts%s_%s_xdata' % (nete,corediv)] = ts_data.xdata
                    ts_group['ts%s_%s_ydata' % (nete,corediv)] = ts_data.ydata
                    ts_group['ts%s_%s_zdata' % (nete,corediv)] = ts_data.zdata

                    ts_group['ts%s_%s_xunits' % (nete,corediv)] = str(ts_data.xunits)
                    ts_group['ts%s_%s_yunits' % (nete,corediv)] = str(ts_data.yunits)
                    ts_group['ts%s_%s_zunits' % (nete,corediv)] = str(ts_data.zunits)

                    ts_data = gadata('ts%s_e_%s' % (nete,corediv), shotn, connection=atlconn)

                    ts_group['ts%s_e_%s_xdata' % (nete,corediv)] = ts_data.xdata
                    ts_group['ts%s_e_%s_ydata' % (nete,corediv)] = ts_data.ydata
                    ts_group['ts%s_e_%s_zdata' % (nete,corediv)] = ts_data.zdata

                    ts_group['ts%s_e_%s_xunits' % (nete,corediv)] = str(ts_data.xunits)
                    ts_group['ts%s_e_%s_yunits' % (nete,corediv)] = str(ts_data.yunits)
                    ts_group['ts%s_e_%s_zunits' % (nete,corediv)] = str(ts_data.zunits)
                except Exception as e:
                    print('Bad shot %d - ts%s_%s\n%s' % (shotn,nete,corediv,e))
                    with open('bad-shots.txt','a') as fid:
                        fid.write('%d - ts%s_%s\n' % (shotn,nete,corediv))

        hf.close()

####
    if co2_pcs:
        hf = h5py.File(os.path.join(shot_path, f'{shotn}_co2_pcs.h5'),'w')
        for co2_chn in ['r0','v1','v2','v3']:
            co2_group = hf.create_group('co2_%s' % (co2_chn))

            co2_data = gadata('pcden%s' % (co2_chn), shotn, connection=atlconn)

            co2_group['co2_%s_xdata' % (co2_chn)] = co2_data.xdata
            co2_group['co2_%s_ydata' % (co2_chn)] = co2_data.ydata
            co2_group['co2_%s_zdata' % (co2_chn)] = co2_data.zdata

            co2_group['co2_%s_xunits' % (co2_chn)] = str(co2_data.xunits)
            co2_group['co2_%s_yunits' % (co2_chn)] = str(co2_data.yunits)
            co2_group['co2_%s_zunits' % (co2_chn)] = str(co2_data.zunits )       
        hf.close()

####


    if co2:
        hf = h5py.File(os.path.join(shot_path, f'{shotn}_co2.h5'),'w')
        for co2_chn in ['r0','v1','v2','v3']:
            co2_group = hf.create_group('co2_%s' % (co2_chn))
            if shotn <= 195740: # old pointnames: 
                for co2_idx in range(10):

                    co2_data = gadata('den%s_uf_%i' % (co2_chn,co2_idx), shotn, connection=atlconn)

                    co2_group['co2_%s_%i_xdata' % (co2_chn,co2_idx)] = co2_data.xdata
                    co2_group['co2_%s_%i_ydata' % (co2_chn,co2_idx)] = co2_data.ydata
                    co2_group['co2_%s_%i_zdata' % (co2_chn,co2_idx)] = co2_data.zdata

                    co2_group['co2_%s_%i_xunits' % (co2_chn,co2_idx)] = str(co2_data.xunits)
                    co2_group['co2_%s_%i_yunits' % (co2_chn,co2_idx)] = str(co2_data.yunits)
                    co2_group['co2_%s_%i_zunits' % (co2_chn,co2_idx)] = str(co2_data.zunits)
            else:
                co2_data = gadata('den%suf' % (co2_chn), shotn, connection=atlconn)

                co2_group['co2_%s_xdata' % (co2_chn)] = co2_data.xdata
                co2_group['co2_%s_ydata' % (co2_chn)] = co2_data.ydata
                co2_group['co2_%s_zdata' % (co2_chn)] = co2_data.zdata

                co2_group['co2_%s_xunits' % (co2_chn)] = str(co2_data.xunits)
                co2_group['co2_%s_yunits' % (co2_chn)] = str(co2_data.yunits)
                co2_group['co2_%s_zunits' % (co2_chn)] = str(co2_data.zunits )       
        hf.close()


    if cer:
        hf = h5py.File(os.path.join(shot_path, f'{shotn}_cer.h5'),'w')
        for prof in ['ti','rot']:
            for pos,chnum in zip(['t', 'v'],[48,32]):
                for chn in np.arange(1,chnum+1):
                    cer_group = hf.create_group('cera%s%s%d' % (prof,pos,chn))
                    try:
                        cer_data = gadata('cera%s%s%d' % (prof,pos,chn), shotn, connection=atlconn)
                        cer_group['cera%s%s%d_xdata' % (prof,pos,chn)] = cer_data.xdata
                        cer_group['cera%s%s%d_ydata' % (prof,pos,chn)] = cer_data.ydata
                        cer_group['cera%s%s%d_zdata' % (prof,pos,chn)] = cer_data.zdata

                        cer_group['cera%s%s%d_xunits' % (prof,pos,chn)] = str(cer_data.xunits)
                        cer_group['cera%s%s%d_yunits' % (prof,pos,chn)] = str(cer_data.yunits)
                        cer_group['cera%s%s%d_zunits' % (prof,pos,chn)] = str(cer_data.zunits)
                    except Exception as e:
                        cer_group['cera%s%s%d_xdata' % (prof,pos,chn)] = -1
                        cer_group['cera%s%s%d_ydata' % (prof,pos,chn)] = -1
                        cer_group['cera%s%s%d_zdata' % (prof,pos,chn)] = -1

                        cer_group['cera%s%s%d_xunits' % (prof,pos,chn)] = -1
                        cer_group['cera%s%s%d_yunits' % (prof,pos,chn)] = -1
                        cer_group['cera%s%s%d_zunits' % (prof,pos,chn)] = -1
                        print('Bad shot %d - cera%s%s%d\n%s' % (shotn,prof,pos,chn,e))
                        pass
        hf.close()


    if mse:
        hf = h5py.File(os.path.join(shot_path, f'{shotn}_mse.h5'),'w')

        for k in range(69):
            mse_group = hf.create_group('msep%d' % (k+1))
            try:
                mse_data = gadata('msep%d' % (k+1), shotn, connection=atlconn)
                mse_group['xdata'] = mse_data.xdata
                mse_group['ydata'] = mse_data.ydata
                mse_group['zdata'] = mse_data.zdata
                mse_group['xunits'] = str(mse_data.xunits)
                mse_group['yunits'] = str(mse_data.yunits)
                mse_group['zunits'] = str(mse_data.zunits)
            except Exception as e:
                mse_group['xdata'] = -1
                mse_group['ydata'] = -1
                mse_group['zdata'] = -1
                mse_group['xunits'] = -1
                mse_group['yunits'] = -1
                mse_group['zunits'] = -1
                print('Bad shot %d - msep%d\n%s' % (shotn,k+1,e))
                pass
        hf.close()

    if fs:
        hf = h5py.File(os.path.join(shot_path, f'{shotn}_fs.h5'),'w')

        for signame in fs_ptnames:
            fs_group = hf.create_group(signame)
            fs_data = gadata(signame, shotn, connection=atlconn)
            fs_group['xdata'] = fs_data.xdata
            fs_group['ydata'] = fs_data.ydata
            fs_group['zdata'] = fs_data.zdata
            fs_group['xunits'] = str(fs_data.xunits)
            fs_group['yunits'] = str(fs_data.yunits)
            fs_group['zunits'] = str(fs_data.zunits)

        hf.close()

    if bes:
        hf = h5py.File(os.path.join(shot_path, f'{shotn}_bes.h5'),'w')
        bess_group = hf.create_group('besslow')	
        besf_group = hf.create_group('besfast')

        for k in range(64):
            bess_data = gadata('bessu%02d' % (k+1), shotn, connection=atlconn)
            bess_group['bessu%02d' % (k+1)] = bess_data.zdata
            besf_data = gadata('besfu%02d' % (k+1), shotn, connection=atlconn)
            besf_group['besfu%02d' % (k+1)] = besf_data.zdata

        bess_group['xdata'] = bess_data.xdata
        bess_group['ydata'] = bess_data.ydata
        bess_group['xunits'] = str(bess_data.xunits)
        bess_group['yunits'] = str(bess_data.yunits)
        bess_group['bessunits'] = str(bess_data.zunits)

        besf_group['xdata'] = besf_data.xdata
        besf_group['ydata'] = besf_data.ydata
        besf_group['xunits'] = str(besf_data.xunits)
        besf_group['yunits'] = str(besf_data.yunits)
        besf_group['besfunits'] = str(besf_data.zunits)

        hf.close()

    if magnetics:
        with open('magnetics_list.txt','r') as fid:
            mag_list=fid.read().splitlines()
            hf = h5py.File(os.path.join(shot_path, f'{shotn}_magnetics.h5'),'w')

            for mag_name in mag_list:
                mag_group = hf.create_group(mag_name)
                mag_data = gadata(mag_name, shotn, connection=atlconn)

                mag_group['xdata'] = mag_data.xdata
                mag_group['ydata'] = mag_data.ydata
                mag_group['zdata'] = mag_data.zdata
                mag_group['xunits'] = str(mag_data.xunits)
                mag_group['yunits'] = str(mag_data.yunits)
                mag_group['zunits'] = str(mag_data.zunits)

            hf.close()

    if fida:
        hf = h5py.File(os.path.join(shot_path, f'{shotn}_fida.h5'),'w')

        for i in range(10):
            for j in range(4):
                fida_group = hf.create_group('fida%d_%d' % (i,j))
                fida_data = gadata('fida%d_%d' % (i+1,j+1), shotn, connection=atlconn)

                fida_group['xdata'] = fida_data.xdata
                fida_group['ydata'] = fida_data.ydata
                fida_group['zdata'] = fida_data.zdata
                fida_group['xunits'] = fida_data.xunits
                fida_group['yunits'] = fida_data.yunits
                fida_group['zunits'] = fida_data.zunits

        hf.close()

    if gfile:
        tempfname = 'tmp'+generate_random_string(10)
        data={x:[] for x in gfile_dict_keys}
        data['AuxQuantities']={x:[] for x in gfile_aux_keys}
        data['fluxSurfaces']={x:[] for x in gfile_flux_keys}
        data['Time']=[]
        for tm in gfile_time:
            try:
                efit_data = OMFITgeqdsk(tempfname).from_mdsplus(device='DIII-D', shot=shotn, time=tm, SNAPfile=efit_type)
                data['Time'].append(tm)
                for dkey in gfile_dict_keys:
                    data[dkey].append(np.asarray(efit_data[dkey]))
                for akey in gfile_aux_keys:
                    data['AuxQuantities'][akey].append(np.asarray(efit_data['AuxQuantities'][akey]))
                for fkey in gfile_flux_keys:
                    data['fluxSurfaces'][fkey].append(np.asarray(efit_data['fluxSurfaces'][fkey]))
                    
            except:
                print('error in %d - %d' % (shotn,tm))
        with h5py.File(os.path.join(shot_path, f'{shotn}_gfile.h5'),'w') as hf:
            dict_group = hf.create_group('general')
            for dkey in gfile_dict_keys:
                dict_group[dkey]=np.squeeze(np.dstack(data[dkey]))
            dict_group['Time']=np.squeeze(np.dstack(data['Time']))
            aux_group = hf.create_group('AuxQuantities')
            for akey in gfile_aux_keys:
                aux_group[akey]=np.squeeze(np.dstack(data['AuxQuantities'][akey]))
            flux_group = hf.create_group('fluxSurfaces')
            for fkey in gfile_flux_keys:
                flux_group[fkey]=np.squeeze(np.dstack(data['fluxSurfaces'][fkey])) 
        temp_files = glob(f'g{shotn}.*')
        for tmpfl in temp_files:
            os.remove(tmpfl)
        os.remove(tempfname)
#	print('time per shot:%ds' % (time.time()-t1))
