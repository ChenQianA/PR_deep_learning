from collections import OrderedDict
import numpy as np
import tensorflow as tf
from .data_container import index_keys


class DataProvider:
    def __init__(self, data_container, ntrain, nvalid, batch_size=1,
                 seed=None, randomized=False):
        self.data_container = data_container
        self._ndata = len(data_container)
        self.nsamples = {'train': ntrain, 'val': nvalid, 'test': len(data_container) - ntrain - nvalid}
        self.batch_size = batch_size

        # Random state parameter, such that random operations are reproducible if wanted
        self._random_state = np.random.RandomState(seed=seed)

        all_idx = np.arange(len(self.data_container))
        if randomized:
            # Shuffle indices
            all_idx = self._random_state.permutation(all_idx)

        # Store indices of training, validation and test data
        self.idx = {'train': all_idx[0:ntrain],
                    'val': all_idx[ntrain:ntrain+nvalid],
                    'test': all_idx[ntrain+nvalid:]}

        if data_container.task == 'Classify':
            self.idx_train = all_idx[0:ntrain]
            target = np.squeeze(data_container.target)
            self.unique_value = np.unique(target)            
            target = target[self.idx_train]
            self.idx_train_list = []
            for i in self.unique_value:
                self.idx_train_list.append(self.idx_train[(target==i).nonzero()[0]])
            self.train_parms = [ntrain//len(self.unique_value), ntrain%len(self.unique_value)] 
    
        # Index for retrieving batches
        self.idx_in_epoch = {'train': 0, 'val': 0, 'test': 0}

        # dtypes of dataset values
        self.dtypes_input = OrderedDict()
        self.dtypes_input['Z'] = tf.int32
        self.dtypes_input['distance'] = tf.float32       
        self.dtypes_input['vector'] = tf.float32        
        for key in index_keys:
            self.dtypes_input[key] = tf.int32
        self.dtype_target = tf.float32

        # Shapes of dataset values
        self.shapes_input = {}
        self.shapes_input['Z'] = [None]
        self.shapes_input['distance'] = [None]
        self.shapes_input['vector'] = [None, 3]       
        for key in index_keys:
            self.shapes_input[key] = [None]
        self.shape_target = [None, data_container.target.shape[1]]

    def shuffle_train(self):
        """Shuffle the training data"""
        if self.data_container.task == 'Classify':
            id_train_list=[]
            for i in self.idx_train_list:
                id_train_list.append(self._random_state.choice(i,self.train_parms[0]))
            for j in self._random_state.choice(self.unique_value, self.train_parms[1]):
                id_train_list.append(self._random_state.choice(self.idx_train_list[j],1))
            self.idx['train'] = np.concatenate(id_train_list, axis=0)
            
        self.idx['train'] = self._random_state.permutation(self.idx['train'])

    def get_batch_idx(self, split):
        """Return the indices for a batch of samples from the specified set"""
        start = self.idx_in_epoch[split]

        # Is epoch finished?
        if self.idx_in_epoch[split] == self.nsamples[split]:
            start = 0
            self.idx_in_epoch[split] = 0

        # shuffle training set at start of epoch
        if start == 0 and split == 'train':
            self.shuffle_train()

        # Set end of batch
        self.idx_in_epoch[split] += self.batch_size
        if self.idx_in_epoch[split] > self.nsamples[split]:
            self.idx_in_epoch[split] = self.nsamples[split]
        end = self.idx_in_epoch[split]

        return self.idx[split][start:end]

    def idx_to_data(self, idx, return_flattened=False):
        """Convert a batch of indices to a batch of data"""
        batch = self.data_container[idx]

        if return_flattened:
            inputs_targets = []
            for key, dtype in self.dtypes_input.items():
                inputs_targets.append(tf.constant(batch[key], dtype=dtype))
            inputs_targets.append(tf.constant(batch['target'], dtype=tf.float32))
            return inputs_targets
        else:
            inputs = {}
            for key, dtype in self.dtypes_input.items():
                inputs[key] = tf.constant(batch[key], dtype=dtype)
            target = tf.constant(batch['target'], dtype=tf.float32)
            return (inputs, target)

    def get_dataset(self, split):
        """Get a generator-based tf.dataset"""
        def generator():
            while True:
                idx = self.get_batch_idx(split)
                yield self.idx_to_data(idx)
        return tf.data.Dataset.from_generator(
                generator,
                output_types=(dict(self.dtypes_input), self.dtype_target),
                output_shapes=(self.shapes_input, self.shape_target))

    def get_idx_dataset(self, split):
        """Get a generator-based tf.dataset returning just the indices"""
        def generator():
            while True:
                batch_idx = self.get_batch_idx(split)
                yield tf.constant(batch_idx, dtype=tf.int32)
        return tf.data.Dataset.from_generator(
                generator,
                output_types=tf.int32,
                output_shapes=[None])

    def idx_to_data_tf(self, idx):
        """Convert a batch of indices to a batch of data from TensorFlow"""
        dtypes_flattened = list(self.dtypes_input.values())
        dtypes_flattened.append(self.dtype_target)

        inputs_targets = tf.py_function(lambda idx: self.idx_to_data(idx.numpy(), return_flattened=True),
                                        inp=[idx], Tout=dtypes_flattened)

        inputs = {}
        for i, key in enumerate(self.dtypes_input.keys()):
            inputs[key] = inputs_targets[i]
            inputs[key].set_shape(self.shapes_input[key])
        targets = inputs_targets[-1]
        targets.set_shape(self.shape_target)
        return (inputs, targets)
