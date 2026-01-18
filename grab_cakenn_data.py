import numpy as np
import matplotlib.pyplot as plt
import MDSplus as mds
import os


shot_list = np.arange(203015, 203028)

connection = mds.Connection('atlas.gat.com')

all_shot_data = {}
for shotn in shot_list:
    print(f'Grabbing shot {shotn}')
    try:
        connection.openTree('d3d', shotn)

        cakenn = connection.get(f'PTDATA(\'CKXOUT\', {shotn})')
        
        NYMODEL = 2
        NYOUT_PTS = 101
        NYPROF = 7

        prof_data = cakenn.reshape((-1, NYMODEL, NYOUT_PTS, NYPROF))

        dens_profiles = prof_data[:, 1, :, 3]  # 3-- for density || 6 for rotation
        rot_profiles = prof_data[:, 1, :, 6]
        q_profiles = prof_data[:, 1, :, 2]
        pres_profiles = prof_data[:, 1, :, 0]
        etemp_profiles = prof_data[:, 1, :, 4]

        all_shot_data[shotn] = {
            'density': dens_profiles,
            'rotation': rot_profiles,
            'q': q_profiles,
            'pressure': pres_profiles,
            'etemp': etemp_profiles
        }
        print(f'  Successfully retrieved data for shot {shotn}')
    except Exception as e:
        print(f'  Error with shot {shotn}: {type(e).__name__}: {str(e)}')
        print(f'  Skipping shot {shotn}')
        continue

output_dir = 'cakenn_data'
os.makedirs(output_dir, exist_ok=True)

#save pickle
import pickle
with open(os.path.join(output_dir, 'cakenn_profiles_exp25.pkl'), 'wb') as f:
    pickle.dump(all_shot_data, f)