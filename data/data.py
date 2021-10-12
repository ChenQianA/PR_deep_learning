from pymatgen.core.structure import IStructure
from pymatgen.optimization.neighbors import find_points_in_spheres
import numpy as np
import pickle
import math

structure = IStructure.from_file('C:/Users/Administrator/Desktop/2Dmaterial/POSCAR/POSCAR_1')


class neighbors:
    def __init__(self,r=5,pbc=np.array([1,1,0],dtype=int)):
        self.r=float(r)
        self.pbc=pbc
    def neighbors_list(self, coords, lattice, tol=1e-8):
        center_indices, points_indices, images, distances = find_points_in_spheres(
            coords, coords, r=self.r, pbc=self.pbc, lattice=lattice, tol=tol)
        self_pair = (center_indices == points_indices) & (distances <= tol)
        cond = ~self_pair
        center_indices=center_indices[cond]
        points_indices=points_indices[cond]
        images=images[cond]
        distances=distances[cond]
        vectors=coords[center_indices]-coords[points_indices]-np.matmul(images,lattice.T)
        return center_indices, points_indices, distances, vectors
        
        
with open('D:/OneDrive/OneDrive - zju.edu.cn/桌面/2Dmaterial/C2DB数据集/data_dicts_total.pickle', 'rb') as f:
    data_dict_list=pickle.load(f)
  
#####################generate EGv######################################## 
def tensor_sum(S, index_d, omega):
    S = (S + S.T)/2
    S4 = np.array([[S[0,0], S[0,1], S[0,2]/2], [S[1,0], S[1,1], S[1,2]/2], [S[2,0]/2, S[2,1]/2, S[2,2]/4]])
    t_s = 0
    for index_ijkt in [[i, j, k, t] for i in range(1, 3) for j in range(1, 3) for k in range(1, 3) for t in range(1, 3)]:
        if index_ijkt[:2] == [1, 1]:
            index_0 = 0
        elif index_ijkt[:2] == [2, 2]:
            index_0 = 1
        else:
            index_0 = 2
        if index_ijkt[2:] == [1, 1]:
            index_1 = 0
        elif index_ijkt[2:] == [2, 2]:
            index_1 = 1
        else:
            index_1 = 2
        t_s += S4[index_0, index_1]*omega[index_d[0]-1,index_ijkt[0]-1]*omega[index_d[1]-1,index_ijkt[1]-1]*\
               omega[index_d[2]-1,index_ijkt[2]-1]*omega[index_d[3]-1,index_ijkt[3]-1]
    return t_s

def getEGv(X):
    theta = np.linspace(-math.pi, math.pi, 1000)
    l1 = np.expand_dims(np.cos(theta), axis=-1)
    l2 = np.expand_dims(np.sin(theta), axis=-1)
    m1 = np.expand_dims(-np.sin(theta), axis=-1)
    m2 = np.expand_dims(np.cos(theta), axis=-1)
    omega = np.array([[l1, l2], [m1, m2]])
    S = X.I
    S1111_ = tensor_sum(S, [1, 1, 1, 1], omega)
    S1212_ = tensor_sum(S, [1, 2, 1, 2], omega)
    S1122_ = tensor_sum(S, [1, 1, 2, 2], omega)
    E = 1 / S1111_
    G = 1 / (4 * S1212_)
    v = -S1122_ / S1111_
    return theta, E, G, v



################################Dataset################################   
R_list=[]
N_list=[]
Z_list=[]
lattice_list=[]
id_list=[]
Thermodynamic_stability_level_list=[]
for i in data_dict_list.values():
    if 'Thermodynamic stability level' in i.keys():
        urid=i['Uniqe row ID']
        pos_path='C:/Users/Administrator/Desktop/2Dmaterial/POSCAR/POSCAR_'+urid
        poscar=IStructure.from_file(pos_path)
        R_list.append(poscar.frac_coords)
        N_list.append(poscar.num_sites)
        Z_list.extend([s.number for s in poscar.species])
        lattice_list.append(np.array(poscar.lattice.matrix))
        id_list.append(i['Unique identifier'])
        Thermodynamic_stability_level_list.append(i['Thermodynamic stability level'])
    else:
        continue
R=np.concatenate(R_list,axis=0)
N=np.array(N_list, dtype=int)
Z=np.array(Z_list, dtype=int)
lattice=np.concatenate(lattice_list,axis=0) 
ids=np.array(id_list)
Thermodynamic_stability_level=np.expand_dims(np.array(Thermodynamic_stability_level_list, dtype=int)-1,axis=1)
np.savez('C:/Users/Administrator/Desktop/C2DB文章/ML/data/Thermodynamic_stability_level.npz',
         R=R, N=N, Z=Z, lattice=lattice, id=ids, Thermodynamic_stability_level=Thermodynamic_stability_level)

   
R_list=[]
N_list=[]
Z_list=[]
lattice_list=[]
id_list=[]
Phonon_dynamic_stability_list=[]
for i in data_dict_list.values():
    if 'Phonon dynamic stability (low/high)' in i.keys():
        urid=i['Uniqe row ID']
        pos_path='C:/Users/Administrator/Desktop/2Dmaterial/POSCAR/POSCAR_'+urid
        poscar=IStructure.from_file(pos_path)
        R_list.append(poscar.frac_coords)
        N_list.append(poscar.num_sites)
        Z_list.extend([s.number for s in poscar.species])
        lattice_list.append(np.array(poscar.lattice.matrix))
        id_list.append(i['Unique identifier'])
        Phonon_dynamic_stability_list.append(i['Phonon dynamic stability (low/high)'])
    else:
        continue
R=np.concatenate(R_list,axis=0)
N=np.array(N_list, dtype=int)
Z=np.array(Z_list, dtype=int)
lattice=np.concatenate(lattice_list,axis=0) 
ids=np.array(id_list)
Phonon_dynamic_stability=np.array(Phonon_dynamic_stability_list)
Phonon_dynamic_stability[Phonon_dynamic_stability=='low']=0
Phonon_dynamic_stability[Phonon_dynamic_stability=='high']=1
Phonon_dynamic_stability=np.expand_dims(Phonon_dynamic_stability.astype(int),axis=1)
np.savez('C:/Users/Administrator/Desktop/C2DB文章/ML/data/Phonon_dynamic_stability.npz',
         R=R, N=N, Z=Z, lattice=lattice, id=ids, Phonon_dynamic_stability=Phonon_dynamic_stability)


R_list=[]
N_list=[]
Z_list=[]
lattice_list=[]
id_list=[]
Stiffness_dynamic_stability_list=[]
for i in data_dict_list.values():
    if 'Stiffness dynamic stability (low/high)' in i.keys():
        urid=i['Uniqe row ID']
        pos_path='C:/Users/Administrator/Desktop/2Dmaterial/POSCAR/POSCAR_'+urid
        poscar=IStructure.from_file(pos_path)
        R_list.append(poscar.frac_coords)
        N_list.append(poscar.num_sites)
        Z_list.extend([s.number for s in poscar.species])
        lattice_list.append(np.array(poscar.lattice.matrix))
        id_list.append(i['Unique identifier'])
        Stiffness_dynamic_stability_list.append(i['Stiffness dynamic stability (low/high)'])
    else:
        continue
R=np.concatenate(R_list,axis=0)
N=np.array(N_list, dtype=int)
Z=np.array(Z_list, dtype=int)
lattice=np.concatenate(lattice_list,axis=0) 
ids=np.array(id_list)
Stiffness_dynamic_stability=np.array(Stiffness_dynamic_stability_list)
Stiffness_dynamic_stability[Stiffness_dynamic_stability=='low']=0
Stiffness_dynamic_stability[Stiffness_dynamic_stability=='high']=1
Stiffness_dynamic_stability=np.expand_dims(Stiffness_dynamic_stability.astype(int),axis=1)
np.savez('C:/Users/Administrator/Desktop/C2DB文章/ML/data/Stiffness_dynamic_stability.npz',
         R=R, N=N, Z=Z, lattice=lattice, id=ids, Stiffness_dynamic_stability=Stiffness_dynamic_stability)


R_list=[]
N_list=[]
Z_list=[]
lattice_list=[]
id_list=[]
Work_function_list=[]
for i in data_dict_list.values():
    if 'Work function (avg. if finite dipole)' in i.keys():
        urid=i['Uniqe row ID']
        pos_path='C:/Users/Administrator/Desktop/2Dmaterial/POSCAR/POSCAR_'+urid
        poscar=IStructure.from_file(pos_path)
        R_list.append(poscar.frac_coords)
        N_list.append(poscar.num_sites)
        Z_list.extend([s.number for s in poscar.species])
        lattice_list.append(np.array(poscar.lattice.matrix))
        id_list.append(i['Unique identifier'])
        Work_function_list.append(i['Work function (avg. if finite dipole)'].split()[0])
    else:
        continue
R=np.concatenate(R_list,axis=0)
N=np.array(N_list, dtype=int)
Z=np.array(Z_list, dtype=int)
lattice=np.concatenate(lattice_list,axis=0) 
ids=np.array(id_list)
Work_function=np.expand_dims(np.array(Work_function_list, dtype=float),axis=1)
np.savez('C:/Users/Administrator/Desktop/C2DB文章/ML/data/Work_function.npz',
         R=R, N=N, Z=Z, lattice=lattice, id=ids, Work_function=Work_function)


R_list=[]
N_list=[]
Z_list=[]
lattice_list=[]
id_list=[]
Emax_list=[]
Emin_list=[]
Gmax_list=[]
Gmin_list=[]
Vmax_list=[]
Vmin_list=[]
for j in data_dict_list.values():
    if 'Stiffness tensor, 11-component' in j.keys() and \
        'Stiffness dynamic stability (low/high)' in j.keys() and\
            j['Stiffness dynamic stability (low/high)']=='high' and\
                'Phonon dynamic stability (low/high)' in j.keys() and\
                j['Phonon dynamic stability (low/high)']=='high':
        C11 = float(j['Stiffness tensor, 11-component'].split()[0])
        C12 = float(j['Stiffness tensor, 12-component'].split()[0])
        C13 = float(j['Stiffness tensor, 13-component'].split()[0])
        C21 = float(j['Stiffness tensor, 21-component'].split()[0])
        C22 = float(j['Stiffness tensor, 22-component'].split()[0])
        C23 = float(j['Stiffness tensor, 23-component'].split()[0])
        C31 = float(j['Stiffness tensor, 31-component'].split()[0])
        C32 = float(j['Stiffness tensor, 32-component'].split()[0])
        C33 = float(j['Stiffness tensor, 33-component'].split()[0])
        X = np.matrix([[C11, C12, C13],[C21, C22, C23],[C31, C32, C33]])
        eigenvalue, featurevector = np.linalg.eig(X)
        if eigenvalue.min()>0:
            X = np.matrix([[C11, C12, C13/math.sqrt(2)],[C21, C22, C23/math.sqrt(2)],[C31/math.sqrt(2), C32/math.sqrt(2), C33/2]])
            _,E,G,v=getEGv(X)
            Emax_list.append(E.max())
            Emin_list.append(E.min())
            Gmax_list.append(G.max())
            Gmin_list.append(G.min())            
            Vmax_list.append(v.max())
            Vmin_list.append(v.min())        
            urid=j['Uniqe row ID']
            pos_path='C:/Users/Administrator/Desktop/2Dmaterial/POSCAR/POSCAR_'+urid
            poscar=IStructure.from_file(pos_path)
            R_list.append(poscar.frac_coords)
            N_list.append(poscar.num_sites)
            Z_list.extend([s.number for s in poscar.species])
            lattice_list.append(np.array(poscar.lattice.matrix))
            id_list.append(j['Unique identifier'])
        else:
            continue
    else:
        continue           
R=np.concatenate(R_list,axis=0)
N=np.array(N_list, dtype=int)
Z=np.array(Z_list, dtype=int)
lattice=np.concatenate(lattice_list,axis=0) 
ids=np.array(id_list)
Emax=np.log(np.expand_dims(np.array(Emax_list, dtype=float),axis=1))
Emin=np.log(np.expand_dims(np.array(Emin_list, dtype=float),axis=1))
Gmax=np.log(np.expand_dims(np.array(Gmax_list, dtype=float),axis=1))
Gmin=np.log(np.expand_dims(np.array(Gmin_list, dtype=float),axis=1))
Vmax=np.expand_dims(np.array(Vmax_list, dtype=float),axis=1)
Vmin=np.expand_dims(np.array(Vmin_list, dtype=float),axis=1)
Elastic = np.concatenate([Emax, Emin, Gmax, Gmin, Vmax, Vmin], axis=-1)
np.savez('C:/Users/Administrator/Desktop/C2DB文章/ML/data/Vmax.npz',
         R=R, N=N, Z=Z, lattice=lattice, id=ids, Vmax=Vmax)
np.savez('C:/Users/Administrator/Desktop/C2DB文章/ML/data/Vmin.npz',
         R=R, N=N, Z=Z, lattice=lattice, id=ids, Vmin=Vmin)
np.savez('C:/Users/Administrator/Desktop/C2DB文章/ML/data/Elastic.npz',
         R=R, N=N, Z=Z, lattice=lattice, id=ids, Elastic=Elastic)
   

R_list=[]
N_list=[]
Z_list=[]
lattice_list=[]
id_list=[]
Vmax_all_list=[]
Vmin_all_list=[]
for j in data_dict_list.values():
    if 'Stiffness tensor, 11-component' in j.keys() and \
        'Stiffness dynamic stability (low/high)' in j.keys() and\
            j['Stiffness dynamic stability (low/high)']=='high' and\
                'Phonon dynamic stability (low/high)' in j.keys() and\
                j['Phonon dynamic stability (low/high)']=='high':
        C11 = float(j['Stiffness tensor, 11-component'].split()[0])
        C12 = float(j['Stiffness tensor, 12-component'].split()[0])
        C13 = float(j['Stiffness tensor, 13-component'].split()[0])
        C21 = float(j['Stiffness tensor, 21-component'].split()[0])
        C22 = float(j['Stiffness tensor, 22-component'].split()[0])
        C23 = float(j['Stiffness tensor, 23-component'].split()[0])
        C31 = float(j['Stiffness tensor, 31-component'].split()[0])
        C32 = float(j['Stiffness tensor, 32-component'].split()[0])
        C33 = float(j['Stiffness tensor, 33-component'].split()[0])
        X = np.matrix([[C11, C12, C13],[C21, C22, C23],[C31, C32, C33]])
        eigenvalue, featurevector = np.linalg.eig(X)
        if eigenvalue.min()>0:
            X = np.matrix([[C11, C12, C13/math.sqrt(2)],[C21, C22, C23/math.sqrt(2)],[C31/math.sqrt(2), C32/math.sqrt(2), C33/2]])
            _,E,G,v=getEGv(X)        
            Vmax_all_list.append(v.max())
            Vmin_all_list.append(v.min())        
            urid=j['Uniqe row ID']
            pos_path='D:/OneDrive/OneDrive - zju.edu.cn/桌面/2Dmaterial/POSCAR/POSCAR_'+urid
            poscar=IStructure.from_file(pos_path)
            R_list.append(poscar.frac_coords)
            N_list.append(poscar.num_sites)
            Z_list.extend([s.number for s in poscar.species])
            lattice_list.append(np.array(poscar.lattice.matrix))
            id_list.append(j['Unique identifier'])
        else:
            continue
    elif 'Stiffness tensor, 11-component' in j.keys() and \
        'Stiffness dynamic stability (low/high)' in j.keys() and\
            j['Stiffness dynamic stability (low/high)']=='high' and\
                'Phonon dynamic stability (low/high)' in j.keys() and\
                    j['Phonon dynamic stability (low/high)']=='low':
        C11 = float(j['Stiffness tensor, 11-component'].split()[0])
        C12 = float(j['Stiffness tensor, 12-component'].split()[0])
        C13 = float(j['Stiffness tensor, 13-component'].split()[0])
        C21 = float(j['Stiffness tensor, 21-component'].split()[0])
        C22 = float(j['Stiffness tensor, 22-component'].split()[0])
        C23 = float(j['Stiffness tensor, 23-component'].split()[0])
        C31 = float(j['Stiffness tensor, 31-component'].split()[0])
        C32 = float(j['Stiffness tensor, 32-component'].split()[0])
        C33 = float(j['Stiffness tensor, 33-component'].split()[0])
        X = np.matrix([[C11, C12, C13],[C21, C22, C23],[C31, C32, C33]])
        eigenvalue, featurevector = np.linalg.eig(X)
        X = np.matrix([[C11, C12, C13/math.sqrt(2)],[C21, C22, C23/math.sqrt(2)],[C31/math.sqrt(2), C32/math.sqrt(2), C33/2]])
        _,E,G,v=getEGv(X)
        if eigenvalue.min()>0 and v.min()>-1 and v.max()<1:
            Vmax_all_list.append(v.max())
            Vmin_all_list.append(v.min())        
            urid=j['Uniqe row ID']
            pos_path='D:/OneDrive/OneDrive - zju.edu.cn/桌面/2Dmaterial/POSCAR/POSCAR_'+urid
            poscar=IStructure.from_file(pos_path)
            R_list.append(poscar.frac_coords)
            N_list.append(poscar.num_sites)
            Z_list.extend([s.number for s in poscar.species])
            lattice_list.append(np.array(poscar.lattice.matrix))
            id_list.append(j['Unique identifier'])
        else:
            continue
    else:
        continue           
R=np.concatenate(R_list,axis=0)
N=np.array(N_list, dtype=int)
Z=np.array(Z_list, dtype=int)
lattice=np.concatenate(lattice_list,axis=0) 
ids=np.array(id_list)
Vmax_all=np.expand_dims(np.array(Vmax_all_list, dtype=float),axis=1)
Vmin_all=np.expand_dims(np.array(Vmin_all_list, dtype=float),axis=1)
np.savez('D:/OneDrive/OneDrive - zju.edu.cn/桌面/C2DB文章/ML/data/Vmax_all.npz',
         R=R, N=N, Z=Z, lattice=lattice, id=ids, Vmax_all=Vmax_all)
np.savez('D:/OneDrive/OneDrive - zju.edu.cn/桌面/C2DB文章/ML/data/Vmin_all.npz',
         R=R, N=N, Z=Z, lattice=lattice, id=ids, Vmin_all=Vmin_all)

    
R_list=[]
N_list=[]
Z_list=[]
lattice_list=[]
id_list=[]
Meta_stable=[]
for i in data_dict_list.values():
    if 'Thermodynamic stability level' in i.keys() and\
        'Stiffness dynamic stability (low/high)' in i.keys() and\
            'Phonon dynamic stability (low/high)' in i.keys():
                if i['Stiffness dynamic stability (low/high)']=='high' and\
                    i['Phonon dynamic stability (low/high)']=='high' and\
                        int(i['Thermodynamic stability level'])>=2:
                    urid=i['Uniqe row ID']
                    pos_path='C:/Users/Administrator/Desktop/2Dmaterial/POSCAR/POSCAR_'+urid
                    poscar=IStructure.from_file(pos_path)
                    R_list.append(poscar.frac_coords) 
                    N_list.append(poscar.num_sites) 
                    Z_list.extend([s.number for s in poscar.species]) 
                    lattice_list.append(np.array(poscar.lattice.matrix)) 
                    id_list.append(i['Unique identifier']) 
                    Meta_stable.append(1)
                else:
                    urid=i['Uniqe row ID']
                    pos_path='C:/Users/Administrator/Desktop/2Dmaterial/POSCAR/POSCAR_'+urid
                    poscar=IStructure.from_file(pos_path)
                    R_list.append(poscar.frac_coords) 
                    N_list.append(poscar.num_sites) 
                    Z_list.extend([s.number for s in poscar.species]) 
                    lattice_list.append(np.array(poscar.lattice.matrix)) 
                    id_list.append(i['Unique identifier']) 
                    Meta_stable.append(0)
    else:
        continue
R=np.concatenate(R_list,axis=0)
N=np.array(N_list, dtype=int)
Z=np.array(Z_list, dtype=int)
lattice=np.concatenate(lattice_list,axis=0) 
ids=np.array(id_list)
Meta_stable=np.expand_dims(np.array(Meta_stable, dtype=int),axis=1)
np.savez('C:/Users/Administrator/Desktop/C2DB文章/ML/data/Meta_stable.npz',
         R=R, N=N, Z=Z, lattice=lattice, id=ids, Meta_stable=Meta_stable)


R_list=[]
N_list=[]
Z_list=[]
lattice_list=[]
id_list=[]
Stable=[]
for i in data_dict_list.values():
    if 'Thermodynamic stability level' in i.keys() and\
        'Stiffness dynamic stability (low/high)' in i.keys() and\
            'Phonon dynamic stability (low/high)' in i.keys():
                if i['Stiffness dynamic stability (low/high)']=='high' and\
                    i['Phonon dynamic stability (low/high)']=='high' and\
                        int(i['Thermodynamic stability level'])==3:
                    urid=i['Uniqe row ID']
                    pos_path='C:/Users/Administrator/Desktop/2Dmaterial/POSCAR/POSCAR_'+urid
                    poscar=IStructure.from_file(pos_path)
                    R_list.append(poscar.frac_coords) 
                    N_list.append(poscar.num_sites) 
                    Z_list.extend([s.number for s in poscar.species]) 
                    lattice_list.append(np.array(poscar.lattice.matrix)) 
                    id_list.append(i['Unique identifier']) 
                    Stable.append(1)
                else:
                    urid=i['Uniqe row ID']
                    pos_path='C:/Users/Administrator/Desktop/2Dmaterial/POSCAR/POSCAR_'+urid
                    poscar=IStructure.from_file(pos_path)
                    R_list.append(poscar.frac_coords) 
                    N_list.append(poscar.num_sites) 
                    Z_list.extend([s.number for s in poscar.species]) 
                    lattice_list.append(np.array(poscar.lattice.matrix)) 
                    id_list.append(i['Unique identifier']) 
                    Stable.append(0)
    else:
        continue
R=np.concatenate(R_list,axis=0)
N=np.array(N_list, dtype=int)
Z=np.array(Z_list, dtype=int)
lattice=np.concatenate(lattice_list,axis=0) 
ids=np.array(id_list)
Stable=np.expand_dims(np.array(Stable, dtype=int),axis=1)
np.savez('C:/Users/Administrator/Desktop/C2DB文章/ML/data/Stable.npz',
         R=R, N=N, Z=Z, lattice=lattice, id=ids, Stable=Stable)


R_list=[]
N_list=[]
Z_list=[]
lattice_list=[]
id_list=[]
Dynamical_stable=[]
for i in data_dict_list.values():
    if 'Stiffness dynamic stability (low/high)' in i.keys() and\
        'Phonon dynamic stability (low/high)' in i.keys():
                if i['Stiffness dynamic stability (low/high)']=='high' and\
                    i['Phonon dynamic stability (low/high)']=='high':
                    urid=i['Uniqe row ID']
                    pos_path='C:/Users/Administrator/Desktop/2Dmaterial/POSCAR/POSCAR_'+urid
                    poscar=IStructure.from_file(pos_path)
                    R_list.append(poscar.frac_coords) 
                    N_list.append(poscar.num_sites) 
                    Z_list.extend([s.number for s in poscar.species]) 
                    lattice_list.append(np.array(poscar.lattice.matrix)) 
                    id_list.append(i['Unique identifier']) 
                    Dynamical_stable.append(1)
                else:
                    urid=i['Uniqe row ID']
                    pos_path='C:/Users/Administrator/Desktop/2Dmaterial/POSCAR/POSCAR_'+urid
                    poscar=IStructure.from_file(pos_path)
                    R_list.append(poscar.frac_coords) 
                    N_list.append(poscar.num_sites) 
                    Z_list.extend([s.number for s in poscar.species]) 
                    lattice_list.append(np.array(poscar.lattice.matrix)) 
                    id_list.append(i['Unique identifier']) 
                    Dynamical_stable.append(0)
    else:
        continue
R=np.concatenate(R_list,axis=0)
N=np.array(N_list, dtype=int)
Z=np.array(Z_list, dtype=int)
lattice=np.concatenate(lattice_list,axis=0) 
ids=np.array(id_list)
Dynamical_stable=np.expand_dims(np.array(Dynamical_stable, dtype=int),axis=1)
np.savez('C:/Users/Administrator/Desktop/C2DB文章/ML/data/Dynamical_stable.npz',
         R=R, N=N, Z=Z, lattice=lattice, id=ids, Dynamical_stable=Dynamical_stable)


R_list=[]
N_list=[]
Z_list=[]
lattice_list=[]
id_list=[]
Heat_of_formation_list=[]
for i in data_dict_list.values():
    if 'Heat of formation' in i.keys():
        urid=i['Uniqe row ID']
        pos_path='C:/Users/Administrator/Desktop/2Dmaterial/POSCAR/POSCAR_'+urid
        poscar=IStructure.from_file(pos_path)
        R_list.append(poscar.frac_coords)
        N_list.append(poscar.num_sites)
        Z_list.extend([s.number for s in poscar.species])
        lattice_list.append(np.array(poscar.lattice.matrix))
        id_list.append(i['Unique identifier'])
        Heat_of_formation_list.append(i['Heat of formation'].split()[0])
    else:
        continue
R=np.concatenate(R_list,axis=0)
N=np.array(N_list, dtype=int)
Z=np.array(Z_list, dtype=int)
lattice=np.concatenate(lattice_list,axis=0) 
ids=np.array(id_list)
Heat_of_formation=np.expand_dims(np.array(Heat_of_formation_list, dtype=float),axis=1)
np.savez('C:/Users/Administrator/Desktop/C2DB文章/ML/data/Heat_of_formation.npz',
         R=R, N=N, Z=Z, lattice=lattice, id=ids, Heat_of_formation=Heat_of_formation)
#########################2Dmatpedia##################################
import json

data_list=[]
for line in  open('D:/OneDrive/OneDrive - zju.edu.cn/桌面/2Dmaterial/2Dmatpedia数据集/db.json','r'):
    data_list.append(json.loads(line))   
    
R_list=[]
N_list=[]
Z_list=[]
lattice_list=[]
id_list=[]
Exfoliation_energy_list=[]
for i in data_list:
    if 'exfoliation_energy_per_atom' in i.keys():
        poscar=IStructure.from_dict(i['structure'])
        R_list.append(poscar.frac_coords)
        N_list.append(poscar.num_sites)
        Z_list.extend([s.number for s in poscar.species])
        lattice_list.append(np.array(poscar.lattice.matrix))
        id_list.append(i['material_id'])
        Exfoliation_energy_list.append(i['exfoliation_energy_per_atom'])
    else:
        continue
R=np.concatenate(R_list,axis=0)
N=np.array(N_list, dtype=int)
Z=np.array(Z_list, dtype=int)
lattice=np.concatenate(lattice_list,axis=0) 
ids=np.array(id_list)
Exfoliation_energy=np.expand_dims(np.array(Exfoliation_energy_list, dtype=float),axis=1)
np.savez('D:/OneDrive/OneDrive - zju.edu.cn/桌面/C2DB文章/model/data/Exfoliation_energy.npz',
         R=R, N=N, Z=Z, lattice=lattice, id=ids, Exfoliation_energy=Exfoliation_energy)

R_list=[]
N_list=[]
Z_list=[]
lattice_list=[]
id_list=[]
Decomposition_energy_list=[]
for i in data_list:
    if 'decomposition_energy' in i.keys():
        poscar=IStructure.from_dict(i['structure'])
        R_list.append(poscar.frac_coords)
        N_list.append(poscar.num_sites)
        Z_list.extend([s.number for s in poscar.species])
        lattice_list.append(np.array(poscar.lattice.matrix))
        id_list.append(i['material_id'])
        Decomposition_energy_list.append(i['decomposition_energy'])
    else:
        continue
R=np.concatenate(R_list,axis=0)
N=np.array(N_list, dtype=int)
Z=np.array(Z_list, dtype=int)
lattice=np.concatenate(lattice_list,axis=0) 
ids=np.array(id_list)
Decomposition_energy=np.expand_dims(np.array(Decomposition_energy_list, dtype=float),axis=1)
np.savez('D:/OneDrive/OneDrive - zju.edu.cn/桌面/C2DB文章/model/data/Decomposition_energy.npz',
         R=R, N=N, Z=Z, lattice=lattice, id=ids, Decomposition_energy=Decomposition_energy)

R_list=[]
N_list=[]
Z_list=[]
lattice_list=[]
id_list=[]
Decomposition_energy_list=[]
for i in data_list:
    if 'decomposition_energy' in i.keys():
        poscar=IStructure.from_dict(i['structure'])
        R_list.append(poscar.frac_coords)
        N_list.append(poscar.num_sites)
        Z_list.extend([s.number for s in poscar.species])
        lattice_list.append(np.array(poscar.lattice.matrix))
        id_list.append(i['material_id'])
        Decomposition_energy_list.append(i['decomposition_energy'])
    else:
        poscar=IStructure.from_dict(i['structure'])
        R_list.append(poscar.frac_coords)
        N_list.append(poscar.num_sites)
        Z_list.extend([s.number for s in poscar.species])
        lattice_list.append(np.array(poscar.lattice.matrix))
        id_list.append(i['material_id'])
        Decomposition_energy_list.append(np.nan)     
R=np.concatenate(R_list,axis=0)
N=np.array(N_list, dtype=int)
Z=np.array(Z_list, dtype=int)
lattice=np.concatenate(lattice_list,axis=0) 
ids=np.array(id_list)
Decomposition_energy=np.expand_dims(np.array(Decomposition_energy_list, dtype=float),axis=1)
np.savez('D:/OneDrive/OneDrive - zju.edu.cn/桌面/C2DB文章/model/data/Decomposition_energy_nan.npz',
         R=R, N=N, Z=Z, lattice=lattice, id=ids, Decomposition_energy_nan=Decomposition_energy)

R_list=[]
N_list=[]
Z_list=[]
lattice_list=[]
id_list=[]
Exfoliation_energy_list=[]
for i in data_list:
    if 'exfoliation_energy_per_atom' in i.keys():
        poscar=IStructure.from_dict(i['structure'])
        R_list.append(poscar.frac_coords)
        N_list.append(poscar.num_sites)
        Z_list.extend([s.number for s in poscar.species])
        lattice_list.append(np.array(poscar.lattice.matrix))
        id_list.append(i['material_id'])
        Exfoliation_energy_list.append(i['exfoliation_energy_per_atom'])
    else:
        poscar=IStructure.from_dict(i['structure'])
        R_list.append(poscar.frac_coords)
        N_list.append(poscar.num_sites)
        Z_list.extend([s.number for s in poscar.species])
        lattice_list.append(np.array(poscar.lattice.matrix))
        id_list.append(i['material_id'])
        Exfoliation_energy_list.append(np.nan)     
R=np.concatenate(R_list,axis=0)
N=np.array(N_list, dtype=int)
Z=np.array(Z_list, dtype=int)
lattice=np.concatenate(lattice_list,axis=0) 
ids=np.array(id_list)
Exfoliation_energy_list=np.expand_dims(np.array(Exfoliation_energy_list, dtype=float),axis=1)
np.savez('D:/OneDrive/OneDrive - zju.edu.cn/桌面/C2DB文章/model/data/Exfoliation_energy_nan.npz',
         R=R, N=N, Z=Z, lattice=lattice, id=ids, Exfoliation_energy_nan=Exfoliation_energy_list)

R_list=[]
N_list=[]
Z_list=[]
lattice_list=[]
id_list=[]
Exfoliation_energy_list=[]
for i in data_list:
    if 'exfoliation_energy_per_atom' in i.keys():
        poscar=IStructure.from_dict(i['structure'])
        R_list.append(poscar.frac_coords)
        N_list.append(poscar.num_sites)
        Z_list.extend([s.number for s in poscar.species])
        lattice_list.append(np.array(poscar.lattice.matrix))
        id_list.append(i['material_id'])
        num_sites = poscar.num_sites
        area = poscar.lattice.a * poscar.lattice.b * math.sin(poscar.lattice.gamma/180*math.pi)
        Exfoliation_energy_list.append(i['exfoliation_energy_per_atom']*num_sites/area)
    else:
        continue
R=np.concatenate(R_list,axis=0)
N=np.array(N_list, dtype=int)
Z=np.array(Z_list, dtype=int)
lattice=np.concatenate(lattice_list,axis=0) 
ids=np.array(id_list)
Exfoliation_energy=np.expand_dims(np.array(Exfoliation_energy_list, dtype=float),axis=1)
np.savez('D:/OneDrive/OneDrive - zju.edu.cn/桌面/C2DB文章/model/data/Exfoliation_energy_perarea.npz',
         R=R, N=N, Z=Z, lattice=lattice, id=ids, Exfoliation_energy_perarea=Exfoliation_energy)

##############################Jarvis##################################
from jarvis.core.atoms import Atoms
from jarvis.db.figshare import data
import numpy as np

dft_2d = data(dataset='dft_2d')

R_list=[]
N_list=[]
Z_list=[]
lattice_list=[]
id_list=[]
Exfoliation_energy_list=[]
for i in dft_2d:
    if i['exfoliation_energy'] != 'na':
        poscar=Atoms.from_dict(i['atoms'])
        R_list.append(poscar.frac_coords)
        N_list.append(poscar.num_atoms)
        Z_list.extend(poscar.atomic_numbers)
        lattice_list.append(np.array(poscar.lattice_mat))
        id_list.append(i['jid'])
        Exfoliation_energy_list.append(i['exfoliation_energy'])
    else:
        continue
R=np.concatenate(R_list,axis=0)
N=np.array(N_list, dtype=int)
Z=np.array(Z_list, dtype=int)
lattice=np.concatenate(lattice_list,axis=0) 
ids=np.array(id_list)
Exfoliation_energy=np.expand_dims(np.array(Exfoliation_energy_list, dtype=float),axis=1)
np.savez('C:/Users/Administrator/Desktop/C2DB文章/ML/data/Exfoliation_energy_jarvis.npz',
         R=R, N=N, Z=Z, lattice=lattice, id=ids, Exfoliation_energy_jarvis=Exfoliation_energy)

R_list=[]
N_list=[]
Z_list=[]
lattice_list=[]
id_list=[]
Formation_energy_peratom_list=[]
for i in dft_2d:
    if i['formation_energy_peratom'] != 'na':
        poscar=Atoms.from_dict(i['atoms'])
        R_list.append(poscar.frac_coords)
        N_list.append(poscar.num_atoms)
        Z_list.extend(poscar.atomic_numbers)
        lattice_list.append(np.array(poscar.lattice_mat))
        id_list.append(i['jid'])
        Formation_energy_peratom_list.append(i['formation_energy_peratom'])
    else:
        continue
R=np.concatenate(R_list,axis=0)
N=np.array(N_list, dtype=int)
Z=np.array(Z_list, dtype=int)
lattice=np.concatenate(lattice_list,axis=0) 
ids=np.array(id_list)
Formation_energy_peratom=np.expand_dims(np.array(Formation_energy_peratom_list, dtype=float),axis=1)
np.savez('C:/Users/Administrator/Desktop/C2DB文章/ML/data/Formation_energy_peratom.npz',
         R=R, N=N, Z=Z, lattice=lattice, id=ids, Formation_energy_peratom=Formation_energy_peratom)

R_list=[]
N_list=[]
Z_list=[]
lattice_list=[]
id_list=[]
Formation_energy_peratom_list=[]
for i in dft_2d:
    if i['formation_energy_peratom'] != 'na':
        poscar=Atoms.from_dict(i['atoms'])
        R_list.append(poscar.frac_coords)
        N_list.append(poscar.num_atoms)
        Z_list.extend(poscar.atomic_numbers)
        lattice_list.append(np.array(poscar.lattice_mat))
        id_list.append(i['jid'])
        Formation_energy_peratom_list.append(i['formation_energy_peratom'])
    else:
        poscar=Atoms.from_dict(i['atoms'])
        R_list.append(poscar.frac_coords)
        N_list.append(poscar.num_atoms)
        Z_list.extend(poscar.atomic_numbers)
        lattice_list.append(np.array(poscar.lattice_mat))
        id_list.append(i['jid'])
        Formation_energy_peratom_list.append(np.nan)
R=np.concatenate(R_list,axis=0)
N=np.array(N_list, dtype=int)
Z=np.array(Z_list, dtype=int)
lattice=np.concatenate(lattice_list,axis=0) 
ids=np.array(id_list)
Formation_energy_peratom=np.expand_dims(np.array(Formation_energy_peratom_list, dtype=float),axis=1)
np.savez('C:/Users/Administrator/Desktop/C2DB文章/ML/data/Formation_energy_peratom_nan.npz',
         R=R, N=N, Z=Z, lattice=lattice, id=ids, Formation_energy_peratom_nan=Formation_energy_peratom)

#############################poisson ratio category#######################
R_list=[]
N_list=[]
Z_list=[]
lattice_list=[]
id_list=[]
Vmax_list=[]
Vmin_list=[]
for j in data_dict_list.values():
    if 'Stiffness tensor, 11-component' in j.keys() and \
        'Stiffness dynamic stability (low/high)' in j.keys() and\
            j['Stiffness dynamic stability (low/high)']=='high' and\
                'Phonon dynamic stability (low/high)' in j.keys() and\
                j['Phonon dynamic stability (low/high)']=='high':
        C11 = float(j['Stiffness tensor, 11-component'].split()[0])
        C12 = float(j['Stiffness tensor, 12-component'].split()[0])
        C13 = float(j['Stiffness tensor, 13-component'].split()[0])
        C21 = float(j['Stiffness tensor, 21-component'].split()[0])
        C22 = float(j['Stiffness tensor, 22-component'].split()[0])
        C23 = float(j['Stiffness tensor, 23-component'].split()[0])
        C31 = float(j['Stiffness tensor, 31-component'].split()[0])
        C32 = float(j['Stiffness tensor, 32-component'].split()[0])
        C33 = float(j['Stiffness tensor, 33-component'].split()[0])
        X = np.matrix([[C11, C12, C13],[C21, C22, C23],[C31, C32, C33]])
        eigenvalue, featurevector = np.linalg.eig(X)
        if eigenvalue.min()>0:
            X = np.matrix([[C11, C12, C13/math.sqrt(2)],[C21, C22, C23/math.sqrt(2)],[C31/math.sqrt(2), C32/math.sqrt(2), C33/2]])
            _,E,G,v=getEGv(X)
            if v.max()<0:
                Vmax_list.append(1)
            else:
                Vmax_list.append(0)
            if v.min()<0:
                Vmin_list.append(1)
            else:
                Vmin_list.append(0)        
            urid=j['Uniqe row ID']
            pos_path='C:/Users/Administrator/Desktop/2Dmaterial/POSCAR/POSCAR_'+urid
            poscar=IStructure.from_file(pos_path)
            R_list.append(poscar.frac_coords)
            N_list.append(poscar.num_sites)
            Z_list.extend([s.number for s in poscar.species])
            lattice_list.append(np.array(poscar.lattice.matrix))
            id_list.append(j['Unique identifier'])
        else:
            continue
    else:
        continue           
R=np.concatenate(R_list,axis=0)
N=np.array(N_list, dtype=int)
Z=np.array(Z_list, dtype=int)
lattice=np.concatenate(lattice_list,axis=0) 
ids=np.array(id_list)
Vmax=np.expand_dims(np.array(Vmax_list, dtype=int),axis=1)
Vmin=np.expand_dims(np.array(Vmin_list, dtype=int),axis=1)
np.savez('C:/Users/Administrator/Desktop/C2DB文章/ML/data/Vmax_class.npz',
         R=R, N=N, Z=Z, lattice=lattice, id=ids, Vmax_class=Vmax)
np.savez('C:/Users/Administrator/Desktop/C2DB文章/ML/data/Vmin_class.npz',
         R=R, N=N, Z=Z, lattice=lattice, id=ids, Vmin_class=Vmin)


####################Thermodynamic_stability_level_binary##################
data=np.load('data/Thermodynamic_stability_level.npz')

Thermodynamic_stability_level_biary = data['Thermodynamic_stability_level'].squeeze().copy()
Thermodynamic_stability_level_biary[Thermodynamic_stability_level_biary==1]=0
Thermodynamic_stability_level_biary[Thermodynamic_stability_level_biary==2]=1
Thermodynamic_stability_level_biary=Thermodynamic_stability_level_biary.reshape(-1,1)
np.savez('D:/OneDrive/OneDrive - zju.edu.cn/桌面/C2DB文章/model/data/Thermodynamic_stability_level_biary.npz',
         R=data['R'], N=data['N'], Z=data['Z'], lattice=data['lattice'], 
         id=data['id'], Thermodynamic_stability_level_biary=Thermodynamic_stability_level_biary)

############################atom feature############################
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.molecule_structure_comparator import CovalentRadius

atom_feature=[]
for i in range(1,90):
    ele=Element.from_Z(i)
    atom_feature.append([ele.mendeleev_no, ele.X, ele.electron_affinity,
                         ele.ionization_energy, ele.group, ele.row, 
                         ele.atomic_radius_calculated, ele.atomic_radius,
                         ele.van_der_waals_radius, ele.average_anionic_radius,
                         ele.average_cationic_radius, ele.average_ionic_radius,
                         CovalentRadius().radius[ele.symbol]])
atom_feature=np.array(atom_feature,dtype=float)  
atom_feature=np.where(np.isfinite(atom_feature), atom_feature, np.zeros_like(atom_feature))
atom_feature=(atom_feature-atom_feature.min(axis=0))/(atom_feature.max(axis=0)-atom_feature.min(axis=0))
atom_feature=np.concatenate([[[0]*13],atom_feature],axis=0)
np.save('data/atom_feature_new.npy',atom_feature) #doesn't work
