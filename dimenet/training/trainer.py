import tensorflow as tf
import tensorflow_addons as tfa
from .schedules import LinearWarmupExponentialDecay
import numpy as np


class Trainer:
    def __init__(self, model, learning_rate=1e-3, warmup_steps=None,
                 decay_steps=100000, decay_rate=0.96,
                 ema_decay=0.999, max_grad_norm=10.0, task='Classify',
                 data_container=None):
        self.model = model
        self.ema_decay = ema_decay
        self.max_grad_norm = max_grad_norm
        self.task = task
        # if task == 'Classify':
        #     segment_ids = tf.constant(np.squeeze(data_container.target))
        #     self.counts = tf.math.unsorted_segment_sum(tf.ones(
        #         segment_ids.shape),segment_ids,2)

        if warmup_steps is not None:
            self.learning_rate = LinearWarmupExponentialDecay(
                learning_rate, warmup_steps, decay_steps, decay_rate)
        else:
            self.learning_rate = tf.optimizers.schedules.ExponentialDecay(
                learning_rate, decay_steps, decay_rate)

        opt = tf.optimizers.Adam(learning_rate=self.learning_rate, amsgrad=True)
        self.optimizer = tfa.optimizers.MovingAverage(opt, average_decay=self.ema_decay)

        # Initialize backup variables
        if model.built:
            self.backup_vars = [tf.Variable(var, dtype=var.dtype, trainable=False)
                                for var in self.model.trainable_weights]
        else:
            self.backup_vars = None

    def update_weights(self, loss, gradient_tape):
        grads = gradient_tape.gradient(loss, self.model.trainable_weights)

        global_norm = tf.linalg.global_norm(grads)
        if self.max_grad_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm, use_norm=global_norm)

        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

    def load_averaged_variables(self):
        self.optimizer.assign_average_vars(self.model.trainable_weights)

    def save_variable_backups(self):
        if self.backup_vars is None:
            self.backup_vars = [tf.Variable(var, dtype=var.dtype, trainable=False)
                                for var in self.model.trainable_weights]
        else:
            for var, bck in zip(self.model.trainable_weights, self.backup_vars):
                bck.assign(var)

    def restore_variable_backups(self):
        for var, bck in zip(self.model.trainable_weights, self.backup_vars):
            var.assign(bck)

    @tf.function
    def train_on_batch(self, dataset_iter, metrics):
        inputs, targets = next(dataset_iter)
        # if self.task == 'Classify':
        #     types = tf.squeeze(targets)
        #     weight = tf.gather(1/self.counts, tf.cast(types, dtype=tf.int32))
        #     weight = weight/tf.reduce_sum(weight)
        
        with tf.GradientTape() as tape:
            preds = self.model(inputs, training=True)
            if self.task == 'Classify':
                preds = tf.math.sigmoid(preds)
                bc = tf.keras.losses.binary_crossentropy(targets, preds)
                loss = tf.reduce_mean(bc)
                # loss = tf.reduce_sum(bc * weight)                                               
            else:                         
                loss = tf.reduce_mean(tf.keras.losses.mae(targets, preds))

        self.update_weights(loss, tape)

        nsamples = tf.shape(preds)[0]
        
        if self.task == 'Classify':
            metrics[0].update_state(loss, nsamples)
            metrics[1].update_state(targets, preds)
        else:
            metrics[0].update_state(loss, nsamples)
            metrics[1].update_state(tf.reduce_mean(tf.keras.losses.mae(targets, preds)), nsamples)
            
        return loss

    @tf.function
    def test_on_batch(self, dataset_iter, metrics):
        inputs, targets = next(dataset_iter)
        preds = self.model(inputs, training=False)
        
        nsamples = tf.shape(preds)[0]

        if self.task == 'Classify':
            preds = tf.math.sigmoid(preds)
            types = tf.squeeze(targets)
            # weight = tf.gather(1/self.counts, tf.cast(types, dtype=tf.int32))
            # weight = weight/tf.reduce_sum(weight)
            bc = tf.keras.losses.binary_crossentropy(targets, preds)
            # loss = tf.reduce_sum(bc * weight)      
            loss = tf.reduce_mean(bc)
            metrics[0].update_state(loss, nsamples)
            metrics[1].update_state(targets, preds)
            return types, tf.squeeze(preds)
        else:
            metrics[0].update_state(tf.reduce_mean(tf.keras.losses.mae(targets, preds)), nsamples)
            metrics[1].update_state(tf.reduce_mean(tf.keras.losses.mae(targets, preds)), nsamples)
            return targets, preds

    @tf.function
    def predict_on_batch(self, inputs):
        
        preds = self.model(inputs, training=False)
                
        if self.task == 'Classify':
            preds = tf.math.sigmoid(preds)
            
        return preds
