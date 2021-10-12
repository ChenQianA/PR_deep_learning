import numpy as np
from pymatgen.optimization.neighbors import find_points_in_spheres


index_keys = ["batch_seg", "idnb_i", "idnb_j", "id_expand_kj",
              "id_reduce_ji"]


class DataContainer:
    def __init__(self, filename, cutoff, target_key, task='Classify', 
                 data_aug=True, seed=None):
        data_dict = np.load(filename, allow_pickle=True)
        self.cutoff = float(cutoff)
        self.target_key = target_key
        self.data_aug = data_aug
        self.task = task
        self._random_state = np.random.RandomState(seed=seed)
        for key in ['id', 'N', 'Z', 'R', 'lattice']:
            if key in data_dict:
                setattr(self, key, data_dict[key])
            else:
                setattr(self, key, None)
        
        # if task=='Classify':
        #     self.target = np.eye(len(np.unique(data_dict[target_key])))[np.squeeze(data_dict[target_key])]
        # else:
        #     self.target = data_dict[target_key]
        
        self.target = data_dict[target_key]

        if self.N is None:
            self.N = np.zeros(len(self.target), dtype=np.int32)
        self.N_cumsum = np.concatenate([[0], np.cumsum(self.N)])

        assert self.R is not None
        
        self.pbc = np.array([1,1,0],dtype=int)

    def _neighbors_list(self, coords, lattice, tol=1e-8):
        center_indices, points_indices, images, distances = find_points_in_spheres(
            coords, coords, r=self.cutoff, pbc=self.pbc, lattice=lattice, tol=tol)
        self_pair = (center_indices == points_indices) & (distances <= tol)
        cond = ~self_pair
        center_indices=center_indices[cond]
        points_indices=points_indices[cond]
        images=images[cond]
        distances=distances[cond]
        vectors=coords[center_indices]-coords[points_indices]-np.matmul(images,lattice.T)
        return center_indices, points_indices, distances, vectors

    def __len__(self):
        return self.target.shape[0]

    def __getitem__(self, idx):
        if type(idx) is int or type(idx) is np.int64:
            idx = [idx]

        data = {}
        data['target'] = self.target[idx]
        data['id'] = self.id[idx]
        data['N'] = self.N[idx]
        data['batch_seg'] = np.repeat(np.arange(len(idx), dtype=np.int32), data['N'])
        adj_matrices = []

        data['Z'] = np.zeros(np.sum(data['N']), dtype=np.int32)
        data['R'] = np.zeros([np.sum(data['N']), 3], dtype=np.float32)
        data['lattice'] = np.zeros([len(idx)*3, 3], dtype=np.float32)

        nend = 0
        enend = 0
        for k, i in enumerate(idx):
            n = data['N'][k]  # number of atoms
            nstart = nend
            nend = nstart + n

            if self.Z is not None:
                data['Z'][nstart:nend] = self.Z[self.N_cumsum[i]:self.N_cumsum[i + 1]]

            R = self.R[self.N_cumsum[i]:self.N_cumsum[i + 1]]
            lattice = self.lattice[3*i:3*(i+1)]
            data['lattice'][3*k:3*(k+1)] = lattice
            
            if self.data_aug:
                lattice = lattice*self._random_state.normal(1, 0.01, size=(3,1))+\
                    self._random_state.normal(0, 0.02, size=(3,3))
                R = np.matmul(R,lattice)+self._random_state.normal(0, 0.02, size=R.shape)
            else:
                R = np.matmul(R,lattice)
                
            data['R'][nstart:nend] = R
            
            r0, r1, r2, r3 = self._neighbors_list(R, lattice)
            r0 = r0 + nstart
            r1 = r1 + nstart
            
            enstart = enend
            enend = enstart + r0.shape[0]
                        
            mesh_kj, mesh_ji = np.meshgrid(r0, r1)
            id_kj, id_ji = (mesh_kj==mesh_ji).nonzero()
            filter_bool = ~np.isclose(r3[id_kj], -r3[id_ji]).all(axis=-1)
            id_kj = id_kj[filter_bool] + enstart
            id_ji = id_ji[filter_bool] + enstart

            adj_matrices.append([r0, r1, r2, r3, id_kj, id_ji])                    
            
        # Target (i) and source (j) nodes of edges
        data['idnb_i'] = np.concatenate([i[0] for i in adj_matrices], axis=0).astype(np.int32)
        data['idnb_j'] = np.concatenate([i[1] for i in adj_matrices], axis=0).astype(np.int32)
        data['distance'] = np.concatenate([i[2] for i in adj_matrices], axis=0).astype(np.float32)
        data['vector'] = np.concatenate([i[3] for i in adj_matrices], axis=0).astype(np.float32)
        data['id_expand_kj'] = np.concatenate([i[4] for i in adj_matrices], axis=0).astype(np.int32)
        data['id_reduce_ji'] = np.concatenate([i[5] for i in adj_matrices], axis=0).astype(np.int32)
        
        return data
