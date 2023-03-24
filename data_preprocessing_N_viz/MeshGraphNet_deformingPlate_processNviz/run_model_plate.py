# Lint as: python3
# pylint: disable=g-bad-file-header
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Runs the learner/evaluator."""
# import os
import pickle
from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf
# import tensorflow.compat.v1 as tfv1
from meshgraphnets import cfd_eval
from meshgraphnets import cfd_model
from meshgraphnets import cloth_eval
from meshgraphnets import cloth_model
from meshgraphnets import core_model
from meshgraphnets import dataset
from tqdm import tqdm
import os
import json
import functools
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress all messages/errors

FLAGS = flags.FLAGS
flags.DEFINE_enum('mode', 'train', ['train', 'eval'],
                  'Train model, or run evaluation.')
flags.DEFINE_enum('model', None, ['cfd', 'cloth', 'plate'],
                  'Select model to run.')
flags.DEFINE_string('checkpoint_dir', None, 'Directory to save checkpoint')
flags.DEFINE_string('dataset_dir', None, 'Directory to load dataset from.')
flags.DEFINE_string('rollout_path', None,
                    'Pickle file to save eval trajectories')
flags.DEFINE_enum('rollout_split', 'valid', ['train', 'test', 'valid'],
                  'Dataset split to use for rollouts.')
flags.DEFINE_integer('num_rollouts', 10, 'No. of rollout trajectories')
flags.DEFINE_integer('num_training_steps', int(10e6), 'No. of training steps')

PARAMETERS = {
    'cfd': dict(noise=0.02, gamma=1.0, field='velocity', history=False,
                size=2, batch=2, model=cfd_model, evaluator=cfd_eval),
    'cloth': dict(noise=0.003, gamma=0.1, field='world_pos', history=True,
                  size=3, batch=1, model=cloth_model, evaluator=cloth_eval),
    'plate': dict(noise=0.003, gamma=0.1, field='world_pos', history=True,
                  size=3, batch=1, model=cloth_model, evaluator=cloth_eval)
}

@tf.function
def _parse(proto, meta):
  """ directly taken from meshgraphnet code"""
  feature_lists = {k: tf.io.VarLenFeature(tf.string) for k in meta['field_names']}
  features = tf.io.parse_single_example(proto, feature_lists)
  out = {}
  for key, field in meta['features'].items():
      data = tf.io.decode_raw(features[key].values, getattr(tf, field['dtype']))
      data = tf.reshape(data, field['shape'])
      if field['type'] == 'static':
          data = tf.tile(data, [meta['trajectory_length'], 1, 1])
      elif field['type'] == 'dynamic_varlen':
          length = tf.io.decode_raw(features['length_' + key].values, tf.int32)
          length = tf.reshape(length, [-1])
          data = tf.RaggedTensor.from_row_lengths(data, row_lengths=length)
      elif field['type'] != 'dynamic':
          raise ValueError('invalid data format')
      out[key] = data
  return out


def load_dataset(split):
    """Load dataset (directly taken from meshgraphnet code)"""
    with open(os.path.join('meshgraphnets/', FLAGS.dataset_dir, 'meta.json'), 'r') as fp:
        meta = json.loads(fp.read())
        # print('meta: ', meta)
        # raise
    ds = tf.data.TFRecordDataset('meshgraphnets/data/deforming_plate/' + split + '.tfrecord')
    # print('ds is: ', ds)
    # raise
    ds = ds.map(functools.partial(_parse, meta=meta), num_parallel_calls=8)
    ds = ds.prefetch(1)
    # print('ds is: ', ds)
    # raise
    return ds

@tf.function
def read_tf_records_to_dictionary(tag, keys=None):
    data = load_dataset(tag)
    # print('data is: ', tf.data.get_output_classes(data).keys())
    # raise
    keys = tf.data.get_output_classes(data).keys() if keys is None else keys
    dict_data = {k: [] for k in keys}
    # print('data is: ', data)
    # raise
    for key in keys:
        for d in data:
            # print('key is : ', key)
            # print('d is: ', d)
            # raise
            print('d[key].numpy() is: ', d[key])
            raise
            dict_data[key].append(d[key].numpy())

    return dict_data


def learner(model, params):
  """Run a learner job."""
  # print('FLAGS.dataset_dir is: ', FLAGS.dataset_dir)
  # raise
  # print('cur wd: ', os.getcwd())
  ds = dataset.load_dataset('meshgraphnets/'+FLAGS.dataset_dir, 'train')
  # print('Original ds is: ', ds)
  # # raise
  ds = dataset.add_targets(ds, [params['field']], add_history=params['history'])
  ds = dataset.split_and_preprocess(ds, noise_field=params['field'],
                                    noise_scale=params['noise'],
                                    noise_gamma=params['gamma'])
  test=True
  tag = "test" if test else "train"
  all_data = read_tf_records_to_dictionary(tag)
  print('all_data shape is: ', all_data.shape)
  raise
  cells = []
  mesh_pos = []
  node_type = []
  world_pos = []
  prev_world_pos = []
  target_world_pos = []
  stress = []
  all_inputs_data_iter = tf.data.make_one_shot_iterator(ds).get_next()
  print('all_inputs_data_iter is: ', all_inputs_data_iter['cells'].eval(session=tf.Session()).shape)
  raise
  with tf.device("/gpu:0"):
    for _ in tqdm(range(400)):
      cell = all_inputs_data_iter['cells'].eval(session=tf.Session())
      mesh_p = all_inputs_data_iter['mesh_pos'].eval(session=tf.Session())
      node_t = all_inputs_data_iter['node_type'].eval(session=tf.Session())
      world_p = all_inputs_data_iter['world_pos'].eval(session=tf.Session())
      prev_world_p = all_inputs_data_iter['prev|world_pos'].eval(session=tf.Session())
      target_world_p = all_inputs_data_iter['target|world_pos'].eval(session=tf.Session())
      stre = all_inputs_data_iter['stress'].eval(session=tf.Session())
      print('mesh_p shape is: ', mesh_p.shape)
      print('node_t shape is: ', node_t.shape)
      print('world_p shape is: ', world_p.shape)
      print('prev_world_p shape is: ', prev_world_p.shape)
      print('target_world_p shape is: ', target_world_p.shape)
      print('stre shape is: ', stre.shape)
      raise
      cells.append(cell)
      mesh_pos.append(mesh_p)
      node_type.append(node_t)
      world_pos.append(world_p)
      prev_world_pos.append(prev_world_p)
      target_world_pos.append(target_world_p)
      stress.append(stre)
  print('cells: ', cells)
  np.savez('./meshgraphnets/cells', cells)
  np.savez('./meshgraphnets/mesh_pos', mesh_pos)
  np.savez('./meshgraphnets/node_type', node_type)
  np.savez('./meshgraphnets/world_pos', world_pos)
  np.savez('./meshgraphnets/prev_world_pos', prev_world_pos)
  np.savez('./meshgraphnets/target_world_pos', target_world_pos)
  np.savez('./meshgraphnets/stress', stress)
  # print('cells_np shape is: ', cells_np.shape)
  raise
  inputs = tf.data.make_one_shot_iterator(ds).get_next()


  print('ds:', ds)
  print('cells: ', inputs['cells'].eval(session=tf.Session())[:10,],\
    'Min: ', inputs['cells'].eval(session=tf.Session()).min(),\
    'Max: ', inputs['cells'].eval(session=tf.Session()).max(),\
    len(inputs['cells'].eval(session=tf.Session())))
  print('mesh_pos: ', inputs['mesh_pos'].eval(session=tf.Session())[:10,],\
    'Min: ', inputs['mesh_pos'].eval(session=tf.Session()).min(),\
    'Max: ', inputs['mesh_pos'].eval(session=tf.Session()).max())
  print('node_type: ', inputs['node_type'].eval(session=tf.Session())[:10,],\
    'Min: ', inputs['node_type'].eval(session=tf.Session()).min(),\
    'Max: ', inputs['node_type'].eval(session=tf.Session()).max())
  print('world_pos: ', inputs['world_pos'].eval(session=tf.Session())[:10,],\
    'Min: ', inputs['world_pos'].eval(session=tf.Session()).min(),\
    'Max: ', inputs['world_pos'].eval(session=tf.Session()).max())
  print('inputs.keys(): ', inputs.keys())
  print('prev|world_pos: ', inputs['prev|world_pos'].eval(session=tf.Session())[:10,],\
    'Min: ', inputs['prev|world_pos'].eval(session=tf.Session()).min(),\
    'Max: ', inputs['prev|world_pos'].eval(session=tf.Session()).max())
  print('target|world_pos: ', inputs['target|world_pos'].eval(session=tf.Session())[:10,],\
    'Min: ', inputs['target|world_pos'].eval(session=tf.Session()).min(),\
    'Max: ', inputs['target|world_pos'].eval(session=tf.Session()).max())
  print('stress: ', inputs['stress'].eval(session=tf.Session())[:10,],\
    'Min: ', inputs['stress'].eval(session=tf.Session()).min(),\
    'Max: ', inputs['stress'].eval(session=tf.Session()).max() )
  print('inputs: ', inputs)
  raise
  loss_op = model.loss(inputs)

  global_step = tf.train.create_global_step()
  lr = tf.train.exponential_decay(learning_rate=1e-4,
                                  global_step=global_step,
                                  decay_steps=int(5e6),
                                  decay_rate=0.1) + 1e-6
  optimizer = tf.train.AdamOptimizer(learning_rate=lr)
  train_op = optimizer.minimize(loss_op, global_step=global_step)
  # Don't train for the first few steps, just accumulate normalization stats
  train_op = tf.cond(tf.less(global_step, 1000),
                     lambda: tf.group(tf.assign_add(global_step, 1)),
                     lambda: tf.group(train_op))

  with tf.train.MonitoredTrainingSession(
      hooks=[tf.train.StopAtStepHook(last_step=FLAGS.num_training_steps)],
      checkpoint_dir=FLAGS.checkpoint_dir,
      save_checkpoint_secs=600) as sess:

    while not sess.should_stop():
      _, step, loss = sess.run([train_op, global_step, loss_op])
      if step % 1000 == 0:
        logging.info('Step %d: Loss %g', step, loss)
    logging.info('Training complete.')


def evaluator(model, params):
  """Run a model rollout trajectory."""
  ds = dataset.load_dataset(FLAGS.dataset_dir, FLAGS.rollout_split)
  ds = dataset.add_targets(ds, [params['field']], add_history=params['history'])
  inputs = tf.data.make_one_shot_iterator(ds).get_next()
  scalar_op, traj_ops = params['evaluator'].evaluate(model, inputs)
  tf.train.create_global_step()

  with tf.train.MonitoredTrainingSession(
      checkpoint_dir=FLAGS.checkpoint_dir,
      save_checkpoint_secs=None,
      save_checkpoint_steps=None) as sess:
    trajectories = []
    scalars = []
    for traj_idx in range(FLAGS.num_rollouts):
      logging.info('Rollout trajectory %d', traj_idx)
      scalar_data, traj_data = sess.run([scalar_op, traj_ops])
      trajectories.append(traj_data)
      scalars.append(scalar_data)
    for key in scalars[0]:
      logging.info('%s: %g', key, np.mean([x[key] for x in scalars]))
    with open(FLAGS.rollout_path, 'wb') as fp:
      pickle.dump(trajectories, fp)


def main(argv):
  print('argv is: ', argv)
  del argv
  # print('argv is: ', argv)

  tf.enable_resource_variables()
  tf.disable_eager_execution()
  params = PARAMETERS[FLAGS.model]
  learned_model = core_model.EncodeProcessDecode(
      output_size=params['size'],
      latent_size=128,
      num_layers=2,
      message_passing_steps=15)
  # print('params[model]: ', params['model'])
  # raise
  model = params['model'].Model(learned_model)
  if FLAGS.mode == 'train':
    learner(model, params)
  elif FLAGS.mode == 'eval':
    evaluator(model, params)

if __name__ == '__main__':
  app.run(main)
