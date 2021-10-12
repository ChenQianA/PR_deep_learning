import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import os
import ast
import logging
import yaml
from datetime import datetime

from dimenet.model.dimenet import DimeNet
from dimenet.model.dimenet_pp import DimeNetPP
from dimenet.model.activations import swish
from dimenet.training.trainer import Trainer
# from dimenet.training.metrics import Metrics
from dimenet.training.data_container import DataContainer
from dimenet.training.data_provider import DataProvider
import shutil

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Set up logger
logger = logging.getLogger()
logger.handlers = []
ch = logging.StreamHandler()
formatter = logging.Formatter(
        fmt='%(asctime)s (%(levelname)s): %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel('INFO')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
tf.get_logger().setLevel('WARN')
tf.autograph.set_verbosity(2)

# config.yaml for DimeNet, config_pp.yaml for DimeNet++
with open('config_pp.yaml', 'r') as c:
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

ratio_train = config['ratio_train']
ratio_valid = config['ratio_valid']
data_seed = config['data_seed']
dataset = config['dataset']
logdir = config['logdir']

num_steps = config['num_steps']
ema_decay = config['ema_decay']

learning_rate = config['learning_rate']
warmup_steps = config['warmup_steps']
decay_rate = config['decay_rate']
decay_steps = config['decay_steps']

batch_size = config['batch_size']
evaluation_interval = config['evaluation_interval']
save_interval = config['save_interval']
restart = config['restart']
comment = config['comment']
target = config['target']

data_aug = config['data_aug']

# Create directories
if restart is None:
    directory = (logdir + "/" + target + "_" + 
                 datetime.now().strftime("%Y%m%d_%H%M%S"))
else:
    directory = restart

logging.info(f"Directory: {directory}")

if not os.path.exists(directory):
    os.makedirs(directory)
best_dir = os.path.join(directory, 'best')
if not os.path.exists(best_dir):
    os.makedirs(best_dir)
log_dir = os.path.join(directory, 'logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
best_loss_file = os.path.join(best_dir, 'best_loss.npz')
best_ckpt_file = os.path.join(best_dir, 'ckpt')
step_ckpt_folder = log_dir

fh = logging.FileHandler(directory + "/log.txt", mode='w')
fh.setFormatter(formatter)
logger.addHandler(fh)

shutil.copyfile('config_pp.yaml', directory+'/config_pp.yaml')

summary_writer = tf.summary.create_file_writer(log_dir)

if target in ['Thermodynamic_stability_level','Phonon_dynamic_stability',
              'Stiffness_dynamic_stability','Stable','Meta_stable','Dynamical_stable','Vmin_class','Vmax_class']:
    task='Classify'
else:
    task='Regress' 
   
data_container = DataContainer(dataset, cutoff, target, task, data_aug, data_seed)

# Initialize DataProvider (splits dataset into 3 sets based on data_seed and provides tf.datasets)
num_train = round(len(data_container)*ratio_train)
num_valid = round(len(data_container)*ratio_valid)
data_provider = DataProvider(data_container, num_train, num_valid, batch_size,
                             seed=data_seed, randomized=True)

train = {}
validation = {}
test={}

if task == 'Classify':
    train['metrics'] = [tf.metrics.Mean(),tf.keras.metrics.AUC(curve='PR')]
    validation['metrics'] = [tf.metrics.Mean(),tf.keras.metrics.AUC(curve='PR')]
    test['metrics'] = [tf.metrics.Mean(),tf.keras.metrics.AUC(curve='PR')]  
else:    
    train['metrics'] = [tf.metrics.Mean(), tf.metrics.Mean()]
    validation['metrics'] = [tf.metrics.Mean(), tf.metrics.Mean()]
    test['metrics'] = [tf.metrics.Mean(), tf.metrics.Mean()]

# Initialize datasets
train['dataset'] = data_provider.get_dataset('train').prefetch(tf.data.experimental.AUTOTUNE)
train['dataset_iter'] = iter(train['dataset'])
validation['dataset'] = data_provider.get_dataset('val').prefetch(tf.data.experimental.AUTOTUNE)
validation['dataset_iter'] = iter(validation['dataset'])

if model_name == "dimenet":
    model = DimeNet(
            emb_size=emb_size, num_blocks=num_blocks, num_bilinear=num_bilinear,
            num_spherical=num_spherical, num_radial=num_radial,
            cutoff=cutoff, envelope_exponent=envelope_exponent,
            num_before_skip=num_before_skip, num_after_skip=num_after_skip,
            num_dense_output=num_dense_output, num_targets=len(targets),
            activation=swish, output_init=output_init)
elif model_name == "dimenet++":
    model = DimeNetPP(
            emb_size=emb_size, out_emb_size=out_emb_size,
            int_emb_size=int_emb_size, basis_emb_size=basis_emb_size,
            num_blocks=num_blocks, num_spherical=num_spherical, num_radial=num_radial,
            cutoff=cutoff, envelope_exponent=envelope_exponent,
            num_before_skip=num_before_skip, num_after_skip=num_after_skip,
            num_dense_output=num_dense_output, num_targets=data_container.target.shape[1],
            activation=swish, extensive=extensive, output_init=output_init)
else:
    raise ValueError(f"Unknown model name: '{model_name}'")
    
if os.path.isfile(best_loss_file):
    loss_file = np.load(best_loss_file)
    metrics_best = {k: v.item() for k, v in loss_file.items()}
else:
    metrics_best = {}
    if task == 'Classify':
        metrics_best['binary_crossentropy'] = 1
        metrics_best['AUC'] = 0 
    else:
        metrics_best['MAE_val'] = np.inf 
    metrics_best['step'] = 0
    np.savez(best_loss_file, **metrics_best)
    
trainer = Trainer(model, learning_rate, warmup_steps, decay_steps, decay_rate,
                  ema_decay=ema_decay, max_grad_norm=1000, task=task,
                    data_container=data_container)

# Set up checkpointing
ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=trainer.optimizer, model=model)
manager = tf.train.CheckpointManager(ckpt, step_ckpt_folder, max_to_keep=3)

# Restore latest checkpoint
ckpt_restored = tf.train.latest_checkpoint(log_dir)
if ckpt_restored is not None:
    ckpt.restore(ckpt_restored)
   
with summary_writer.as_default():
    steps_per_epoch = int(np.ceil(num_train / batch_size))

    if ckpt_restored is not None:
        step_init = ckpt.step.numpy()
    else:
        step_init = 1
    for step in range(step_init, num_steps + 1):
        # Update step number
        ckpt.step.assign(step)
        tf.summary.experimental.set_step(step)

        # Perform training step
        trainer.train_on_batch(train['dataset_iter'], train['metrics'])

        # Save progress
        if (step % save_interval == 0):
            manager.save()

        # Evaluate model and log results
        if (step % evaluation_interval == 0):

            # Save backup variables and load averaged variables
            trainer.save_variable_backups()
            trainer.load_averaged_variables()

            # Compute results on the validation set
            for i in range(int(np.ceil(num_valid / batch_size))):
                trainer.test_on_batch(validation['dataset_iter'], validation['metrics'])

            # Update and save best result
            if task == 'Classify':
                if validation['metrics'][-1].result().numpy() > metrics_best['AUC']:
                    metrics_best['step'] = step
                    metrics_best.update({'binary_crossentropy_val': validation['metrics'][0].result().numpy(),
                                         'AUC_val': validation['metrics'][1].result().numpy()})
                    np.savez(best_loss_file, **metrics_best)
                    model.save_weights(best_ckpt_file)
                
                tf.summary.scalar('binary_crossentropy_train', train['metrics'][0].result())
                tf.summary.scalar('AUC_train', train['metrics'][1].result())
                tf.summary.scalar('binary_crossentropy_validation', validation['metrics'][0].result())
                tf.summary.scalar('AUC_validation', validation['metrics'][1].result())
                epoch = step // steps_per_epoch
                logging.info(f"{step}/{num_steps} (epoch {epoch+1}): " 
                             f"binary_crossentropy: train={train['metrics'][0].result().numpy():.6f}, val={validation['metrics'][0].result().numpy():.6f} "
                             f"AUC: train={train['metrics'][1].result().numpy():.6f}, val={validation['metrics'][1].result().numpy():.6f}")
            else:
                if validation['metrics'][-1].result().numpy() < metrics_best['MAE_val']:
                    metrics_best['step'] = step
                    metrics_best.update({'MAE_val': validation['metrics'][-1].result().numpy()})
                    np.savez(best_loss_file, **metrics_best)
                    model.save_weights(best_ckpt_file)
                    
                tf.summary.scalar('huber_train', train['metrics'][0].result())
                tf.summary.scalar('MAE_train', train['metrics'][1].result())
                tf.summary.scalar('huber_validation', validation['metrics'][0].result())
                tf.summary.scalar('MAE_validation', validation['metrics'][1].result())
                epoch = step // steps_per_epoch
                logging.info(f"{step}/{num_steps} (epoch {epoch+1}): " 
                             f"MAE: train={train['metrics'][1].result().numpy():.6f}, val={validation['metrics'][1].result().numpy():.6f}")

            for metric in train['metrics']:
                metric.reset_states()
            
            for metric in validation['metrics']:
                metric.reset_states()

            # Restore backup variables
            trainer.restore_variable_backups()
            
model.load_weights(best_ckpt_file)
test['dataset'] = data_provider.get_dataset('test').prefetch(tf.data.experimental.AUTOTUNE)
test['dataset_iter'] = iter(test['dataset'])


if task == 'Classify':
    targets_list=[]
    preds_list=[]
    for i in range(int(np.ceil(num_valid/ batch_size))):
        targets, preds = trainer.test_on_batch(validation['dataset_iter'], validation['metrics'])
        targets_list.append(targets.numpy())
        preds_list.append(preds.numpy())   
    metrics_best.update({'val_targets':np.squeeze(np.concatenate(targets_list, axis=0)), 
                         'val_preds':np.squeeze(np.concatenate(preds_list, axis=0))})


targets_list=[]
preds_list=[]
for i in range(int(np.ceil((len(data_container)-num_train-num_valid)/ batch_size))):
    targets, preds = trainer.test_on_batch(test['dataset_iter'], test['metrics'])
    targets_list.append(targets.numpy())
    preds_list.append(preds.numpy())
    
metrics_best.update({'targets':np.squeeze(np.concatenate(targets_list, axis=0)),
                     'preds':np.squeeze(np.concatenate(preds_list, axis=0))})
    
if task == 'Classify':
    metrics_best.update({'binary_crossentropy_test': test['metrics'][0].result().numpy(),
                         'AUC_test': test['metrics'][1].result().numpy()})
    logging.info(f"test: binary_crossentropy={test['metrics'][0].result().numpy():.6f}, AUC={test['metrics'][1].result().numpy():.6f}")
    with summary_writer.as_default():
        tf.summary.scalar('binary_crossentropy_test', test['metrics'][0].result())
        tf.summary.scalar('AUC_test', test['metrics'][1].result())
else:
    metrics_best.update({'MAE_test': test['metrics'][-1].result().numpy()})
    logging.info(f"test: MAE={test['metrics'][1].result().numpy():.6f}")
    with summary_writer.as_default():
        tf.summary.scalar('MAE_test', test['metrics'][-1].result())

np.savez(best_loss_file, **metrics_best)


