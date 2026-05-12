import MDSplus
from gadata import gadata
import numpy as np

# signals = ['bmspinj30l', 'bmspinj30r', 'bmspinj33l', 'bmspinj33r', 'bmspinj15l', 'bmspinj15r', 'bmspinj21l', 'bmspinj21r','gasA','echpwrc']
# on_axis_beams = ['bmspinj30l', 'bmspinj30r','bmspinj33l', 'bmspinj33r']
# off_axis_beams = ['bmspinj15l', 'bmspinj15r','bmspinj21l', 'bmspinj21r']

signals = ['tinj', 'pinj', 'ip', 'q95', 'EC.J_ECCD', 'EC.QRFE','prmtan_newid']

conn2 = MDSplus.Connection('atlas.gat.com')

####
# tau_act = 100.0


####
shot = 206593
act_data = {}
on_axis_pwr = None
off_axis_pwr = None
for signal in signals:
    data = gadata(signal, shot, connection=conn2)
    x = np.asarray(data.xdata)
    y = np.asarray(data.zdata)

    ## Sample and Smooth properly

    act_data[signal] = y
    act_data[signal+'_time'] = x

#save file
import pickle
with open('bo_data.pkl', 'wb') as f:
    pickle.dump(act_data, f)