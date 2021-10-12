import tensorflow as tf
# import tensorflow_addons as tfa
import numpy as np
import os
import ast
# import logging
import yaml
# from datetime import datetime

from dimenet.model.dimenet import DimeNet
from dimenet.model.dimenet_pp import DimeNetPP
from dimenet.model.activations import swish
# from dimenet.training.trainer import Trainer
# from dimenet.training.metrics import Metrics
from dimenet.training.data_container import DataContainer
from dimenet.training.data_provider import DataProvider
# import shutil

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
with open('config_pp_test.yaml', 'r') as c:
    config = yaml.safe_load(c)
    

# For strings that yaml doesn't parse (e.g. None)
for key, val in config.items():
    if type(val) is str:
        try:
            config[key] = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            pass
        
model_name = config['model_name']

if model_name == "dimenet":
    num_bilinear = config['num_bilinear']
elif model_name == "dimenet++":
    out_emb_size = config['out_emb_size']
    int_emb_size = config['int_emb_size']
    basis_emb_size = config['basis_emb_size']
    extensive = config['extensive']
else:
    raise ValueError(f"Unknown model name: '{model_name}'")
    
emb_size = config['emb_size']
num_blocks = config['num_blocks']

num_spherical = config['num_spherical']
num_radial = config['num_radial']
output_init = config['output_init']

cutoff = config['cutoff']
envelope_exponent = config['envelope_exponent']

num_before_skip = config['num_before_skip']
num_after_skip = config['num_after_skip']
num_dense_output = config['num_dense_output']

dataset = config['dataset']

batch_size = config['batch_size']
comment = config['comment']
target = config['target']

data_aug = config['data_aug']

#####################################2Dmatpedia################################
model_dir = os.path.join(target,'best','ckpt')

if dataset.split('/')[1].split('.')[0] in ['Thermodynamic_stability_level','Phonon_dynamic_stability',
              'Stiffness_dynamic_stability','Dynamical_stable','Stable',
              'Meta_stable','Vmin_class','Vmax_class']:
    task='Classify'
else:
    task='Regress' 
    
data_container = DataContainer(dataset, cutoff, dataset.split('/')[1].split('.')[0], task, data_aug, None)
data_provider = DataProvider(data_container, 0, 0, batch_size, seed=None,
                             randomized=False)

model = DimeNetPP(
        emb_size=emb_size, out_emb_size=out_emb_size,
        int_emb_size=int_emb_size, basis_emb_size=basis_emb_size,
        num_blocks=num_blocks, num_spherical=num_spherical, num_radial=num_radial,
        cutoff=cutoff, envelope_exponent=envelope_exponent,
        num_before_skip=num_before_skip, num_after_skip=num_after_skip,
        num_dense_output=num_dense_output, num_targets=data_container.target.shape[1],
        activation=swish, extensive=extensive, output_init=output_init)

model.load_weights(model_dir) 

test_dataset = data_provider.get_dataset('test').prefetch(tf.data.experimental.AUTOTUNE)
test_dataset_iter = iter(test_dataset)

preds_list = []
targets_list = []
for i in range(np.ceil(len(data_container)/batch_size).astype(int)):
    inputs, targets = next(test_dataset_iter)
    preds = model(inputs, training=False)
    preds_list.append(preds.numpy())
    targets_list.append(targets.numpy())
preds = np.concatenate(preds_list, axis=0)
targets = np.squeeze(np.concatenate(targets_list, axis=0))

Stable = np.squeeze(preds)

from matplotlib import pyplot as plt

data=np.load('data/Decomposition_energy_nan.npz')

Vmin_model_dir='Vmin/best/ckpt'
model.load_weights(Vmin_model_dir) 
preds_list = []
targets_list = []
for i in range(np.ceil(len(data_container)/batch_size).astype(int)):
    inputs, targets = next(test_dataset_iter)
    preds = model(inputs, training=False)
    preds_list.append(preds.numpy())
    targets_list.append(targets.numpy())
preds = np.concatenate(preds_list, axis=0)
targets = np.squeeze(np.concatenate(targets_list, axis=0))
Vmin=np.squeeze(preds) 
plt.hist(Vmin,bins=20,range=[-1,1])
Vmin[np.logical_and(Vmin<0, Stable>0, ~(targets>0.2))]
data['id'][np.logical_and(Vmin<0, Stable>0, ~(targets>0.2))]

Vmax_model_dir='Vmax/best/ckpt'
model.load_weights(Vmax_model_dir) 
preds_list = []
targets_list = []
for i in range(np.ceil(len(data_container)/batch_size).astype(int)):
    inputs, targets = next(test_dataset_iter)
    preds = model(inputs, training=False)
    preds_list.append(preds.numpy())
    targets_list.append(targets.numpy())
preds = np.concatenate(preds_list, axis=0)
targets = np.squeeze(np.concatenate(targets_list, axis=0))
Vmax=np.squeeze(preds)
plt.hist(Vmax,bins=20,range=[-1,1])
Vmax[np.logical_and(Vmax<0, Stable>0, ~(targets>0.2))]
data['id'][np.logical_and(Vmax<0, Stable>0, ~(targets>0.2))]

import json

data_list=[]
for line in  open('D:/OneDrive/OneDrive - zju.edu.cn/桌面/2Dmaterial/2Dmatpedia数据集/db.json','r'):
    data_list.append(json.loads(line))
    
sg_symbol=[]
sg_number=[]
formula_pretty=[]
formula_anonymous=[]
data_dict_ids={}
for i in data_list:
    sg_symbol.append(i['sg_symbol'])
    sg_number.append(i['sg_number'])
    formula_pretty.append(i['formula_pretty'])
    formula_anonymous.append(i['formula_anonymous'])
    data_dict_ids[i['material_id']] = i['structure']
sg_symbol=np.array(sg_symbol)
sg_number=np.array(sg_number)
formula_pretty=np.array(formula_pretty)
formula_anonymous=np.array(formula_anonymous)

data_ids = np.array(list(data_dict_ids.keys()))

import pandas as pd
Vmin_bool=np.logical_and(np.logical_and(Vmin<0, Stable>0), ~(targets>0.2))
Vmin_data_dict =  {'id':data_ids[Vmin_bool],
                   'Vmin':Vmin[Vmin_bool],
                   'sg_symbol':sg_symbol[Vmin_bool],
                   'sg_number':sg_number[Vmin_bool],
                   'formula_pretty':formula_pretty[Vmin_bool],
                   'formula_anonymous':formula_anonymous[Vmin_bool]}
Vmin_Dataframe = pd.DataFrame(Vmin_data_dict)
Vmin_Dataframe.to_csv('D:/OneDrive/OneDrive - zju.edu.cn/桌面/C2DB文章/negative poisson ratio/Result/2DMatPedia/MinPR/Vmin_2Dmatpedia.csv')

from pymatgen.core.structure import IStructure
for i in data_ids[Vmin_bool]:
    poscar=IStructure.from_dict(data_dict_ids[i])
    poscar.to('poscar','D:/OneDrive/OneDrive - zju.edu.cn/桌面/C2DB文章/negative poisson ratio/Result/2DMatPedia/MinPR/POSCAR/POSCAR_'+i)
    
Vmax_bool=np.logical_and(np.logical_and(Vmax<0, Stable>0), ~(targets>0.2))
Vmax_data_dict =  {'id':data_ids[Vmax_bool],
                   'Vmax':Vmax[Vmax_bool],
                   'sg_symbol':sg_symbol[Vmax_bool],
                   'sg_number':sg_number[Vmax_bool],
                   'formula_pretty':formula_pretty[Vmax_bool],
                   'formula_anonymous':formula_anonymous[Vmax_bool]}
Vmax_Dataframe = pd.DataFrame(Vmax_data_dict)
Vmax_Dataframe.to_csv('D:/OneDrive/OneDrive - zju.edu.cn/桌面/C2DB文章/negative poisson ratio/Result/2DMatPedia/MaxPR/Vmax_2Dmatpedia.csv')
    
for i in data_ids[Vmax_bool]:
    poscar=IStructure.from_dict(data_dict_ids[i])
    poscar.to('poscar','D:/OneDrive/OneDrive - zju.edu.cn/桌面/C2DB文章/negative poisson ratio/Result/2DMatPedia/MaxPR/POSCAR/POSCAR_'+i)
    
###################################jarvis##################################
model_dir = os.path.join('Stable','best','ckpt')
dataset = 'data/Formation_energy_peratom_nan.npz'

if dataset.split('/')[1].split('.')[0] in ['Thermodynamic_stability_level','Phonon_dynamic_stability',
              'Stiffness_dynamic_stability','Dynamical_stable','Stable',
              'Meta_stable','Vmin_class','Vmax_class']:
    task='Classify'
else:
    task='Regress' 
    
data_container = DataContainer(dataset, cutoff, dataset.split('/')[1].split('.')[0],
                               task, data_aug, None)
data_provider = DataProvider(data_container, 0, 0, batch_size, seed=None,
                             randomized=False)
    
model.load_weights(model_dir) 

test_dataset = data_provider.get_dataset('test').prefetch(tf.data.experimental.AUTOTUNE)
test_dataset_iter = iter(test_dataset)

preds_list = []
targets_list = []
for i in range(np.ceil(len(data_container)/batch_size).astype(int)):
    inputs, targets = next(test_dataset_iter)
    preds = model(inputs, training=False)
    preds_list.append(preds.numpy())
    targets_list.append(targets.numpy())
preds = np.concatenate(preds_list, axis=0)
targets = np.squeeze(np.concatenate(targets_list, axis=0))

Stable = np.squeeze(preds)

data=np.load(dataset)

Vmin_model_dir='Vmin/best/ckpt'
model.load_weights(Vmin_model_dir) 
preds_list = []
targets_list = []
for i in range(np.ceil(len(data_container)/batch_size).astype(int)):
    inputs, targets = next(test_dataset_iter)
    preds = model(inputs, training=False)
    preds_list.append(preds.numpy())
    targets_list.append(targets.numpy())
preds = np.concatenate(preds_list, axis=0)
targets = np.squeeze(np.concatenate(targets_list, axis=0))
Vmin=np.squeeze(preds) 
plt.hist(Vmin,bins=20,range=[-1,1])
Vmin[np.logical_and(Vmin<0, Stable>0, ~(targets>0.2))]
data['id'][np.logical_and(Vmin<0, Stable>0, ~(targets>0.2))]

Vmax_model_dir='Vmax/best/ckpt'
model.load_weights(Vmax_model_dir) 
preds_list = []
targets_list = []
for i in range(np.ceil(len(data_container)/batch_size).astype(int)):
    inputs, targets = next(test_dataset_iter)
    preds = model(inputs, training=False)
    preds_list.append(preds.numpy())
    targets_list.append(targets.numpy())
preds = np.concatenate(preds_list, axis=0)
targets = np.squeeze(np.concatenate(targets_list, axis=0))
Vmax=np.squeeze(preds)
plt.hist(Vmax,bins=20,range=[-1,1])
Vmax[np.logical_and(Vmax<0, Stable>0, ~(targets>0.2))]
data['id'][np.logical_and(Vmax<0, Stable>0, ~(targets>0.2))]

from jarvis.core.atoms import Atoms
from jarvis.db.figshare import data
import numpy as np

dft_2d = data(dataset='dft_2d')
    
spg_symbol=[]
spg_number=[]
formula=[]
data_dict_ids={}
for i in dft_2d:
    spg_symbol.append(i['spg_symbol'])
    spg_number.append(i['spg_number'])
    formula.append(i['formula'])
    data_dict_ids[i['jid']] = i['atoms']
spg_symbol=np.array(spg_symbol)
spg_number=np.array(spg_number)
formula=np.array(formula)

Vmin_bool = np.logical_and(np.logical_and(Vmin<0, Stable>0), ~(targets>0.2))
Vmin_data_dict =  {'id':data['id'][Vmin_bool],
                   'Vmin':Vmin[Vmin_bool],
                   'spg_symbol':spg_symbol[Vmin_bool],
                   'spg_number':spg_number[Vmin_bool],
                   'formula':formula[Vmin_bool]}
Vmin_Dataframe = pd.DataFrame(Vmin_data_dict)
Vmin_Dataframe.to_csv('Result/jarvis/Vmin/Vmin_jarvis.csv')

for i in data['id'][Vmin_bool]:
    poscar=Atoms.from_dict(data_dict_ids[i])
    poscar.write_poscar('Result/jarvis/Vmin/POSCAR/POSCAR_'+i)

Vmax_bool = np.logical_and(np.logical_and(Vmax<0, Stable>0), ~(targets>0.2))    
Vmax_data_dict =  {'id':data['id'][Vmax_bool],
                   'Vmax':Vmax[Vmax_bool],
                   'spg_symbol':spg_symbol[Vmax_bool],
                   'spg_number':spg_number[Vmax_bool],
                   'formula':formula[Vmax_bool]}
Vmax_Dataframe = pd.DataFrame(Vmax_data_dict)
Vmax_Dataframe.to_csv('Result/jarvis/Vmax/Vmax_jarvis.csv')
    
for i in data['id'][Vmax_bool]:
    poscar=Atoms.from_dict(data_dict_ids[i])
    poscar.write_poscar('Result//jarvis/Vmax/POSCAR/POSCAR_'+i)
    
##########################Energy evaluation figure#########################
model_dir = os.path.join('Heat_of_formation','best','ckpt')
dataset = 'data/Formation_energy_peratom_nan.npz'
data_container = DataContainer(dataset, cutoff, dataset.split('/')[1].split('.')[0],
                               'Regress', data_aug, None)
data_provider = DataProvider(data_container, 0, 0, batch_size, seed=None,
                               randomized=False)
model.load_weights(model_dir) 
test_dataset = data_provider.get_dataset('test').prefetch(tf.data.experimental.AUTOTUNE)
test_dataset_iter = iter(test_dataset)
preds_list = []
targets_list = []
for i in range(np.ceil(len(data_container)/batch_size).astype(int)):
    inputs, targets = next(test_dataset_iter)
    preds = model(inputs, training=False)
    preds_list.append(preds.numpy())
    targets_list.append(targets.numpy())
preds_jarvis = np.squeeze(np.concatenate(preds_list, axis=0))
targets_jarvis = np.squeeze(np.concatenate(targets_list, axis=0))

data=np.load('Heat_of_formation/best/best_loss.npz')
preds_c2db = data['preds']
targets_c2db = data['targets']

from matplotlib import pyplot as plt
import matplotlib as mpl

plt.figure(figsize=(8,8))
plt.scatter(targets_c2db, preds_c2db, s=10, c='blue', alpha=0.5, label='C2DB test set', edgecolors='none')
plt.scatter(targets_jarvis, preds_jarvis, s=10, c='orange', alpha=0.5, label='jarvis', edgecolors='none')
plt.legend(loc='upper left', bbox_to_anchor=(0.01, 0.99), prop=mpl.rc(
    'font',family='Times New Roman'), fontsize=24, markerscale=5)
plt.axis([-3.5,2.2,-3.5,2.2])
plt.xticks(list(range(-3,3,1)))
plt.yticks(list(range(-3,3,1)))
plt.tick_params(labelsize=28)
plt.xlabel('True heat of formation (eV/atom)' , fontproperties='Times New Roman',fontsize=28)
plt.ylabel('Predicted heat of formation (eV/atom)' , fontproperties='Times New Roman',fontsize=28)
plt.savefig('Energy_evaluation.png', dpi=300)

###########################stability figure#############################
model_dir = os.path.join('Stable','best','ckpt')
model.load_weights(model_dir) 
dataset = 'data/Decomposition_energy_nan.npz'
data_container = DataContainer(dataset, cutoff, dataset.split('/')[1].split('.')[0],
                               'Regress', data_aug, None)
data_provider = DataProvider(data_container, 0, 0, batch_size, seed=None,
                               randomized=False)
test_dataset = data_provider.get_dataset('test').prefetch(tf.data.experimental.AUTOTUNE)
test_dataset_iter = iter(test_dataset)
preds_list = []
targets_list = []
for i in range(np.ceil(len(data_container)/batch_size).astype(int)):
    inputs, targets = next(test_dataset_iter)
    preds = model(inputs, training=False)
    preds_list.append(preds.numpy())
    targets_list.append(targets.numpy())
preds_2dmatpedia_stable = np.squeeze(np.concatenate(preds_list, axis=0))
targets_2dmatpedia_energy = np.squeeze(np.concatenate(targets_list, axis=0))

plt.figure(figsize=(8,8))
energies = [targets_2dmatpedia_energy[preds_2dmatpedia_stable>0], 
            targets_2dmatpedia_energy[preds_2dmatpedia_stable<=0]]
colors = ['orange', 'dodgerblue']
names = ['Stable', 'Unstable']
bins = np.arange(0.0, 3.6, 0.2)
plt.hist(energies, bins = bins, stacked = True, density = False, color = colors,
         label=names, rwidth=0.95, alpha=0.8)
plt.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95), prop=mpl.rc(
    'font',family='Times New Roman'), fontsize=28, markerscale=5)
plt.tick_params(labelsize=28)
plt.xlabel('Decomposition energy (eV/atom)' , fontproperties='Times New Roman',fontsize=28)
plt.ylabel('Counts' , fontproperties='Times New Roman',fontsize=28)
plt.tight_layout()
plt.savefig('Decomposition_energy_vs_stability.png', dpi=300)

###
dataset = 'data/Formation_energy_peratom_nan.npz'
data_container = DataContainer(dataset, cutoff, dataset.split('/')[1].split('.')[0],
                               'Regress', data_aug, None)
data_provider = DataProvider(data_container, 0, 0, batch_size, seed=None,
                               randomized=False)
test_dataset = data_provider.get_dataset('test').prefetch(tf.data.experimental.AUTOTUNE)
test_dataset_iter = iter(test_dataset)
preds_list = []
targets_list = []
for i in range(np.ceil(len(data_container)/batch_size).astype(int)):
    inputs, targets = next(test_dataset_iter)
    preds = model(inputs, training=False)
    preds_list.append(preds.numpy())
    targets_list.append(targets.numpy())
preds_jarvis_stable = np.squeeze(np.concatenate(preds_list, axis=0))
targets_jarvis_energy = np.squeeze(np.concatenate(targets_list, axis=0))

plt.figure(figsize=(8,8))
energies = [targets_jarvis_energy[preds_jarvis_stable>0], 
            targets_jarvis_energy[preds_jarvis_stable<=0]]
colors = ['orange', 'dodgerblue']
names = ['Stable', 'Unstable']
bins = np.arange(-3.4, 2.0, 0.2)
plt.hist(energies, bins = bins, stacked = True, density = False, color = colors,
         label=names, rwidth=0.95, alpha=0.8)
plt.legend(loc='upper left', bbox_to_anchor=(0.05, 0.95), prop=mpl.rc(
    'font',family='Times New Roman'), fontsize=28, markerscale=5)
plt.tick_params(labelsize=28)
plt.xlabel('Formation energy per atom (eV/atom)' , fontproperties='Times New Roman',fontsize=28)
plt.ylabel('Counts' , fontproperties='Times New Roman',fontsize=28)
plt.tight_layout()
plt.savefig('Formation_energy_peratom_vs_stability.png', dpi=300)

###
model_dir = os.path.join('Vmax','best','ckpt')
model.load_weights(model_dir) 
dataset = 'data/Decomposition_energy_nan.npz'
data_container = DataContainer(dataset, cutoff, dataset.split('/')[1].split('.')[0],
                               'Regress', data_aug, None)
data_provider = DataProvider(data_container, 0, 0, batch_size, seed=None,
                               randomized=False)
test_dataset = data_provider.get_dataset('test').prefetch(tf.data.experimental.AUTOTUNE)
test_dataset_iter = iter(test_dataset)
preds_list = []
targets_list = []
for i in range(np.ceil(len(data_container)/batch_size).astype(int)):
    inputs, targets = next(test_dataset_iter)
    preds = model(inputs, training=False)
    preds_list.append(preds.numpy())
    targets_list.append(targets.numpy())
preds_2dmatpedia_Vmax = np.squeeze(np.concatenate(preds_list, axis=0))
targets_2dmatpedia_energy = np.squeeze(np.concatenate(targets_list, axis=0))

dataset = 'data/Formation_energy_peratom_nan.npz'
data_container = DataContainer(dataset, cutoff, dataset.split('/')[1].split('.')[0],
                               'Regress', data_aug, None)
data_provider = DataProvider(data_container, 0, 0, batch_size, seed=None,
                               randomized=False)
test_dataset = data_provider.get_dataset('test').prefetch(tf.data.experimental.AUTOTUNE)
test_dataset_iter = iter(test_dataset)
preds_list = []
targets_list = []
for i in range(np.ceil(len(data_container)/batch_size).astype(int)):
    inputs, targets = next(test_dataset_iter)
    preds = model(inputs, training=False)
    preds_list.append(preds.numpy())
    targets_list.append(targets.numpy())
preds_jarvis_Vmax = np.squeeze(np.concatenate(preds_list, axis=0))
targets_jarvis_energy = np.squeeze(np.concatenate(targets_list, axis=0))

###
model_dir = os.path.join('Vmin','best','ckpt')
model.load_weights(model_dir) 
dataset = 'data/Decomposition_energy_nan.npz'
data_container = DataContainer(dataset, cutoff, dataset.split('/')[1].split('.')[0],
                               'Regress', data_aug, None)
data_provider = DataProvider(data_container, 0, 0, batch_size, seed=None,
                               randomized=False)
test_dataset = data_provider.get_dataset('test').prefetch(tf.data.experimental.AUTOTUNE)
test_dataset_iter = iter(test_dataset)
preds_list = []
targets_list = []
for i in range(np.ceil(len(data_container)/batch_size).astype(int)):
    inputs, targets = next(test_dataset_iter)
    preds = model(inputs, training=False)
    preds_list.append(preds.numpy())
    targets_list.append(targets.numpy())
preds_2dmatpedia_Vmin = np.squeeze(np.concatenate(preds_list, axis=0))
targets_2dmatpedia_energy = np.squeeze(np.concatenate(targets_list, axis=0))

dataset = 'data/Formation_energy_peratom_nan.npz'
data_container = DataContainer(dataset, cutoff, dataset.split('/')[1].split('.')[0],
                               'Regress', data_aug, None)
data_provider = DataProvider(data_container, 0, 0, batch_size, seed=None,
                               randomized=False)
test_dataset = data_provider.get_dataset('test').prefetch(tf.data.experimental.AUTOTUNE)
test_dataset_iter = iter(test_dataset)
preds_list = []
targets_list = []
for i in range(np.ceil(len(data_container)/batch_size).astype(int)):
    inputs, targets = next(test_dataset_iter)
    preds = model(inputs, training=False)
    preds_list.append(preds.numpy())
    targets_list.append(targets.numpy())
preds_jarvis_Vmin = np.squeeze(np.concatenate(preds_list, axis=0))
targets_jarvis_energy = np.squeeze(np.concatenate(targets_list, axis=0))

np.savez('2dmatpedia.npz', Decomposition_energy_nan=targets_2dmatpedia_energy,
         Stable=preds_2dmatpedia_stable, Vmax=preds_2dmatpedia_Vmax,
         Vmin=preds_2dmatpedia_Vmin)
np.savez('jarvis.npz', Formation_energy_peratom_nan=targets_jarvis_energy,
         Stable=preds_jarvis_stable, Vmax=preds_jarvis_Vmax, Vmin=preds_jarvis_Vmin)

Vmax_2dmatpedia=preds_2dmatpedia_Vmax[np.logical_and(preds_2dmatpedia_stable>0,
                                                     ~(targets_2dmatpedia_energy>0.2))]
Vmin_2dmatpedia = preds_2dmatpedia_Vmin[np.logical_and(preds_2dmatpedia_stable>0,
                                                       ~(targets_2dmatpedia_energy>0.2))]
Vmax_jarvis = preds_jarvis_Vmax[np.logical_and(preds_jarvis_stable>0,
                                               ~(targets_jarvis_energy>0))]
Vmin_jarvis= preds_jarvis_Vmin[np.logical_and(preds_jarvis_stable>0,
                                               ~(targets_jarvis_energy>0))]

plt.figure(figsize=(8,8))
plt.plot([-1,2.5],[-1,2.5], alpha=1, color='orange', linestyle='--')
plt.axhline(0, color='green', linestyle='--', alpha=1)
plt.axvline(0, color='green', linestyle='--', alpha=1)
plt.scatter(Vmin_2dmatpedia, Vmax_2dmatpedia, s=15, c='dodgerblue', alpha=1)
plt.axis([-0.5,1.5,-0.5,1.5])
plt.xticks(np.arange(-0.5,1.6,0.5))
plt.yticks(np.arange(-0.5,1.6,0.5))
plt.tick_params(labelsize=28)
plt.xlabel('Vmin' , fontproperties='Times New Roman',fontsize=28)
plt.ylabel('Vmax' , fontproperties='Times New Roman',fontsize=28)
plt.tight_layout()
plt.savefig('2dmatpedia_vmax_vmin.png', dpi=300)

plt.figure(figsize=(8,8))
plt.plot([-1,2.5],[-1,2.5], alpha=1, color='orange', linestyle='--')
plt.axhline(0, color='green', linestyle='--', alpha=1)
plt.axvline(0, color='green', linestyle='--', alpha=1)
plt.scatter(Vmin_jarvis, Vmax_jarvis, s=15, c='dodgerblue', alpha=1)
plt.axis([-0.5,1.5,-0.5,1.5])
plt.xticks(np.arange(-0.5,1.6,0.5))
plt.yticks(np.arange(-0.5,1.6,0.5))
plt.tick_params(labelsize=28)
plt.xlabel('Vmin' , fontproperties='Times New Roman',fontsize=28)
plt.ylabel('Vmax' , fontproperties='Times New Roman',fontsize=28)
plt.tight_layout()
plt.savefig('jarvis_vmax_vmin.png', dpi=300)
###########################stability confusion matrix######################
data_stable = np.load('Stable/best/best_loss.npz')
targets_stable = data_stable['targets'].astype(int)
preds_stable = data_stable['preds']
binary_stable = (preds_stable>=0.5).astype(int)

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(targets_stable, binary_stable)

fig, ax = plt.subplots(figsize=(8, 8)) 
ax.matshow(confusion_matrix, cmap=plt.cm.Blues, alpha=0.5) 
for i in range(confusion_matrix.shape[0]):
    for j in range(confusion_matrix.shape[1]):
        plt.text(x=j, y=i,s=confusion_matrix[i, j], va='center', ha='center', fontsize=30, fontproperties='Times New Roman') 
plt.xlabel('Predicted stability', fontproperties='Times New Roman', fontsize=30) 
plt.ylabel('True stability', fontproperties='Times New Roman', fontsize=30) 
plt.title('Confusion Matrix', fontproperties='Times New Roman', fontsize=36)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.tick_params(labelsize=30)
plt.savefig('Stability_confusion_matrix.png', dpi=800)

########################Vmax and Vmin test################################
data_Vmax = np.load('Vmax/best/best_loss.npz')
preds_Vmax = data_Vmax['preds']
targets_Vmax = data_Vmax['targets']
data_Vmin = np.load('Vmin/best/best_loss.npz')
preds_Vmin = data_Vmin['preds']
targets_Vmin = data_Vmin['targets']

plt.figure(figsize=(8,8))
plt.plot([-1,2],[-1,2], alpha=0.8, color='orange', linestyle='--')
plt.scatter(targets_Vmax, preds_Vmax, s=20, c='dodgerblue', alpha=0.8, edgecolors='none')
plt.axis([0,1.2,0,1.2])
plt.xticks(np.arange(0.0,1.2,0.5))
plt.yticks(np.arange(0.0,1.2,0.5))
plt.tick_params(labelsize=32)
plt.xlabel("True maximun Poisson's ratio", fontproperties='Times New Roman',fontsize=32)
plt.ylabel("Predicted maximun Poisson's ratio", fontproperties='Times New Roman',fontsize=32)
plt.text(0.2, 1.0, s='MAE = 0.080', fontproperties='Times New Roman',fontsize=32)
plt.tight_layout()
plt.savefig('Vmax_evaluation.png', dpi=300)

plt.figure(figsize=(8,8))
plt.plot([-1,2],[-1,2], alpha=0.8, color='orange', linestyle='--')
plt.axhline(0, color='green', linestyle='--', alpha=0.8)
plt.axvline(0, color='green', linestyle='--', alpha=0.8)
plt.scatter(targets_Vmin, preds_Vmin, s=20, c='dodgerblue', alpha=0.8, edgecolors='none')
plt.axis([-0.5,1.0,-0.5,1.0])
plt.xticks(np.arange(-0.5,1.1,0.5))
plt.yticks(np.arange(-0.5,1.1,0.5))
plt.tick_params(labelsize=25)
plt.xlabel("True minimun Poisson's ratio", fontproperties='Times New Roman',fontsize=32)
plt.ylabel("Predicted minimun Poisson's ratio", fontproperties='Times New Roman',fontsize=32)
plt.text(-0.45, 0.75, s='MAE = 0.074', fontproperties='Times New Roman',fontsize=32)
plt.tight_layout()
plt.savefig('Vmin_evaluation.png', dpi=300)

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix((targets_Vmin<0).astype(int),
                                    ((preds_Vmin)<0).astype(int))
fig, ax = plt.subplots(figsize=(8, 8)) 
ax.matshow(confusion_matrix, cmap=plt.cm.Blues, alpha=0.5) 
for i in range(confusion_matrix.shape[0]):
    for j in range(confusion_matrix.shape[1]):
        plt.text(x=j, y=i,s=confusion_matrix[i, j], va='center', ha='center', fontsize=56, fontproperties='Times New Roman') 
plt.xlabel('Predicted Vmin<0', fontproperties='Times New Roman', fontsize=56) 
plt.ylabel('True Vmin<0', fontproperties='Times New Roman', fontsize=56) 
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.tick_params(labelsize=56)
plt.tight_layout()
plt.savefig('Vmin_confusion_matrix.png', dpi=300, transparent=True)

##########################ROC curve################################
import numpy as np

data=np.load('Stable/best/best_loss.npz')

from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(data['targets'], data['preds'])
roc_auc = auc(fpr, tpr)

from matplotlib import pyplot as plt

plt.figure(figsize=(8,8))
lw = 4
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman',fontsize=32)
plt.ylabel('True Positive Rate', fontproperties='Times New Roman',fontsize=32)
plt.legend(loc="lower right", prop={'family':'Times New Roman', 'size':28})
plt.tick_params(labelsize=25)
plt.tight_layout()
plt.savefig('D:/OneDrive/OneDrive - zju.edu.cn/桌面/C2DB文章/negative poisson ratio/figure/ROC.png', dpi=300)
plt.show()



########################new structure analyse########################
import pickle

with open('D:/OneDrive/OneDrive - zju.edu.cn/桌面/2Dmaterial/C2DB数据集/data_dicts_total.pickle', 'rb') as f:
    data_dict_list=pickle.load(f)
    
prototype_dict={}
for i in data_dict_list.values():
    prototype='-'.join(i['Crystal type'].split('-')[:2])
    if prototype in prototype_dict.keys():
        prototype_dict[prototype].append(dict(sorted(i['species_dict'].items(), key=lambda t: (t[1], t[0]))))
    else:
        prototype_dict[prototype]=[dict(sorted(i['species_dict'].items(), key=lambda t: (t[1], t[0])))]

prototye_2dmatpedia=['-'.join([i['formula_anonymous'], str(i['sg_number'])]) for i in data_list]

from itertools import groupby
def retrieve_species(formula_reduced_abc):
    s_d = formula_reduced_abc.split()
    species_dict={}
    for s in s_d:
        ss = [''.join(list(g)) for k, g in groupby(s, key=lambda x: x.isdigit())]
        species_dict[ss[0]]=int(ss[1])
    species_dict = dict(sorted(species_dict.items(), key=lambda t: (t[1], t[0])))
    return species_dict
species_dict_2dmatpedia=[retrieve_species(i['formula_reduced_abc']) for i in data_list]

matpedia_in_C2DB=[]
for i, j in zip(prototye_2dmatpedia, species_dict_2dmatpedia):
    if i in prototype_dict.keys():
        if list(j.keys()) in [list(k.keys()) for k in prototype_dict[i]]:
            matpedia_in_C2DB.append(True)
        else:
            matpedia_in_C2DB.append(False)     
    else:
        matpedia_in_C2DB.append(False)
matpedia_in_C2DB=np.array(matpedia_in_C2DB)

matpedia_result=np.load('2dmatpedia.npz')
targets = matpedia_result['Decomposition_energy_nan']
Stable = matpedia_result['Stable']
Vmax = matpedia_result['Vmax']
Vmin = matpedia_result['Vmin']

Vmin_bool=np.logical_and(np.logical_and(Vmin<0, Stable>0), ~(targets>0.2))##118 crystals
Vmin_new_bool=np.logical_and(np.logical_and(np.logical_and(Vmin<0, Stable>0), ~(targets>0.2)), ~matpedia_in_C2DB)##88 crystals
    
Vmax_bool=np.logical_and(np.logical_and(Vmax<0, Stable>0), ~(targets>0.2))#11 crystals
Vmax_new_bool=np.logical_and(np.logical_and(np.logical_and(Vmax<0, Stable>0), ~(targets>0.2)), ~matpedia_in_C2DB)##10 crystals

        
from pymatgen.core.structure import IStructure
from jarvis.core.atoms import Atoms
from jarvis.db.figshare import data
from fractions import gcd
from functools import reduce

dft_2d = data(dataset='dft_2d')

def find_gcd(list):
    x = reduce(gcd, list)
    return x

prototype_jarvis=[]
species_dict_jarvis=[]
for i in dft_2d:
    spg_num=dft_2d[0]['spg_number']
    poscar=Atoms.from_dict(i['atoms'])
    prototype_jarvis.append('-'.join([poscar.composition.prototype, str(i['spg_number'])]))
    species_dict = poscar.composition.to_dict()
    gcd_n = find_gcd(list(species_dict.values()))
    values = [int(v/gcd_n) for v in species_dict.values()]
    species_dict = dict(zip(species_dict.keys(), values))
    species_dict = dict(sorted(species_dict.items(), key=lambda t: (t[1], t[0])))
    species_dict_jarvis.append(species_dict)

jarvis_in_C2DB=[]
for i, j in zip(prototype_jarvis, species_dict_jarvis):
    if i in prototype_dict.keys():
        if list(j.keys()) in [list(k.keys()) for k in prototype_dict[i]]:
            jarvis_in_C2DB.append(True)
        else:
            jarvis_in_C2DB.append(False)     
    else:
        jarvis_in_C2DB.append(False)
jarvis_in_C2DB=np.array(jarvis_in_C2DB)  

jarvis_result=np.load('jarvis.npz')
targets = jarvis_result['Formation_energy_peratom_nan']
Stable = jarvis_result['Stable']
Vmax = jarvis_result['Vmax']
Vmin = jarvis_result['Vmin']

Vmin_bool=np.logical_and(np.logical_and(Vmin<0, Stable>0), ~(targets>0.2))##23 crystals
Vmin_new_bool=np.logical_and(np.logical_and(np.logical_and(Vmin<0, Stable>0), ~(targets>0.2)), ~jarvis_in_C2DB)##11 crystals
    
Vmax_bool=np.logical_and(np.logical_and(Vmax<0, Stable>0), ~(targets>0.2))#1 crystals
Vmax_new_bool=np.logical_and(np.logical_and(np.logical_and(Vmax<0, Stable>0), ~(targets>0.2)), ~jarvis_in_C2DB)##1 crystals


prototype_dict_2dmatpedia={}
for i, j in zip(prototye_2dmatpedia, species_dict_2dmatpedia):
    if i in prototype_dict_2dmatpedia.keys():
        prototype_dict_2dmatpedia[i].append(dict(sorted(j.items(), key=lambda t: (t[1], t[0]))))
    else:
        prototype_dict_2dmatpedia[i]=[dict(sorted(j.items(), key=lambda t: (t[1], t[0])))]

jarvis_in_2dmatpedia=[]
for i, j in zip(prototype_jarvis, species_dict_jarvis):
    if i in prototype_dict_2dmatpedia.keys():
        if list(j.keys()) in [list(k.keys()) for k in prototype_dict_2dmatpedia[i]]:
            jarvis_in_2dmatpedia.append(True)
        else:
            jarvis_in_2dmatpedia.append(False)     
    else:
        jarvis_in_2dmatpedia.append(False)
jarvis_in_2dmatpedia=np.array(jarvis_in_2dmatpedia) 


Vmin_new_bool=np.logical_and(np.logical_and(np.logical_and(np.logical_and(Vmin<0, Stable>0), ~(targets>0.2)), ~jarvis_in_C2DB), ~jarvis_in_2dmatpedia)##3 crystals
Vmax_new_bool=np.logical_and(np.logical_and(np.logical_and(np.logical_and(Vmax<0, Stable>0), ~(targets>0.2)), ~jarvis_in_C2DB), ~jarvis_in_2dmatpedia)##0 crystals
