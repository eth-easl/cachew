# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import abc
import functools
import multiprocessing
import sys
import threading
import warnings

import numpy as np
import six
from six.moves import queue as Queue  # pylint: disable=redefined-builtin



from tensorflow.core.framework import dataset_metadata_pb2
from tensorflow.core.framework import dataset_options_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.python import tf2
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.data.ops import structured_function
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import random_seed
from tensorflow.python.data.util import structure
from tensorflow.python.data.util import traverse
from tensorflow.python.eager import context
from tensorflow.python.framework import auto_control_deps
from tensorflow.python.framework import auto_control_deps_utils as acd_utils
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed as core_random_seed
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.ops import gen_io_ops
from tensorflow.python.ops import gen_stateless_random_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.training.tracking import base as tracking_base
from tensorflow.python.training.tracking import tracking
from tensorflow.python.types import trace
from tensorflow.python.util import deprecation
from tensorflow.python.util import lazy_loader
from tensorflow.python.util import nest as tf_nest
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export

# TODO: Find place for tf.variant
'''dtypes_by_bytes = [tf2.int8,
                   tf2.qint8,
                   tf2.quint8,
                   tf2.uint8,
                   tf2.bfloat16,
                   tf2.float16,
                   tf2.half,
                   tf2.int16,
                   tf2.qint16,
                   tf2.quint16,
                   tf2.uint16,
                   tf2.float32,
                   tf2.int32,
                   tf2.qint32,
                   tf2.uint32,
                   tf2.float64,
                   tf2.int64,
                   tf2.uint64
                   ]'''

dtypes_by_bytes = ["<dtype: \'int8\'>",
                   "<dtype: \'uint8\'>",
                   "<dtype: \'bfloat16\'>",
                   "<dtype: \'float16\'>",
                   "<dtype: \'int16\'>",
                   "<dtype: \'float32\'>",
                   "<dtype: \'int32\'>",
                   "<dtype: \'float64\'>",
                   "<dtype: \'int64\'>"
                   ]

dtypes_by_bytes = ["int8",
                   "uint8",
                   "bfloat16",
                   "float16",
                   "int16",
                   "float32",
                   "int32",
                   "float64",
                   "int64"
                   ]

def get_ds_dtypes_shapes(dataset):
  types = [] # First type will be the outer type (dict, tuple, single elem_spec)
  shapes = []

  elem_spec = dataset.element_spec
  print(elem_spec)
  if isinstance(elem_spec, tuple):
    #print("Elem spec is a tuple!")
    types.append('tuple')

    num_elems = len(elem_spec)
    for i in range(num_elems):
      #print(elem_spec[i])

      if isinstance(elem_spec[i], dict):
        print("Nested dicts are currently not supported!")
        types.append('dict')
        shapes += []
      elif 'NoneTensorSpec' in str(type(elem_spec[i])):
        types.append(None)
        shapes += [None]
      else:
        types.append(str(elem_spec[i].dtype).split('\'')[1])
        #print(str(elem_spec[i].dtype).split('\'')[1])
        shapes += list(elem_spec[i].shape)
        #print(elem_spec[i].shape)
  elif isinstance(elem_spec, dict):
    #print("Elem spec is a dict!")
    types.append('dict')

    num_elems = len(elem_spec)
    for i in sorted(elem_spec.items()):
      #print("Key is: " + i[0])
      #print(i)

      val = i[1]
      if str(type(val)) == "<class 'tensorflow.python.framework.tensor_spec.TensorSpec'>":
        types.append(str(val.dtype).split('\'')[1])
        #print(types[-1])
        shapes += list(val.shape)
        #print(val.shape)
      

  elif str(type(elem_spec)) == "<class 'tensorflow.python.framework.tensor_spec.TensorSpec'>":
    types.append('Elem_spec')

    types.append(str(elem_spec.dtype).split('\'')[1])
    #print(elem_spec.dtype)
    cur_s = list(elem_spec.shape)
    shapes += cur_s
    #print(str(elem_spec.dtype).split('\'')[1])
  else:
    print("Unsupported spec type")
  print(types)
  print(shapes)
  return types, shapes

def should_reorder(org_types, org_shapes, new_types, new_shapes):
  print("Inside should_reorder()")
  if new_types[0] != org_types[0]:
    print("Not matching outer types")
    print(org_types[0], new_types[0])
    return False, False
  org_types = org_types[1:]
  new_types = new_types[1:]

  if org_shapes != new_shapes:
    # This op changes the shape => not a casting op
    print("different shape")
    return False, False
  else:
    for t in org_types:
      if t == 'dict':
        print("Dict type")
        pass
      elif t not in dtypes_by_bytes:
        print("not num type")
        print(t)
        print("Num types are: ")
        print(dtypes_by_bytes)
        return False, False
    for t in new_types:
      if t == 'dict':
        print("Dict type")
        pass
      elif t not in dtypes_by_bytes:
        print("not num type")
        print(t)
        print("Num types are: ")
        print(dtypes_by_bytes)
        return False, False
    
    org_types = [i for i in org_types if i != 'dict']
    new_types = [i for i in new_types if i != 'dict']
    
    if org_types != new_types:
      for i in range(len(org_types)):
        print(dtypes_by_bytes.index(str(new_types[i])))
        print(dtypes_by_bytes.index(str(org_types[i])))
        # At least some component changed to a 'cheaper' dtype
        if (dtypes_by_bytes.index(str(new_types[i])) < dtypes_by_bytes.index(str(org_types[i]))):
          return True, False
      for i in range(len(org_types)):
        # At least some component changed to a more expensive dtype
        if (dtypes_by_bytes.index(str(new_types[i])) > dtypes_by_bytes.index(str(org_types[i]))):
          return False, True
      return False, False
    else:
      print("input output types were identical")
      return False, False

def op_preserves_shape(dataset):
  
  cur_types, cur_shapes = get_ds_dtypes_shapes(dataset)
  org_types, org_shapes = get_ds_dtypes_shapes(dataset._input_dataset)

  if cur_types[0] != org_types[0]:
    print("Outer elem types don't match!")
    return False

  if (len(cur_shapes) == len(org_shapes)):
    return True
  else:
    return False

# get position of the op w.r.t the user's order. This position corresponds to the position after the dtype reordering
def get_op_position(dataset):

  pos = 0

  if hasattr(dataset, '_position'):
    return dataset._position
  if hasattr(dataset, '_input_dataset'):
    return get_op_position(dataset._input_dataset)

  return pos

def node_does_unknown_resize(dataset):

  in_types, in_shapes = get_ds_dtypes_shapes(dataset)
  out_types, out_shapes = get_ds_dtypes_shapes(dataset._input_dataset)

  if 'dict' in in_types[1:] or 'dict' in out_types[1:]:
    print('Nested dicts are currently not supported')
    return False
  
  if (len(in_shapes) != len(out_shapes)):
    print("Dimensions changed, do not reorder")
    return False

  if (in_shapes == out_shapes):
    print("In/out shapes were identical")
    return False

  for i in range(len(in_shapes)):
    if (in_shapes[i] != out_shapes[i] and (out_shapes[i] == None or in_shapes[i] == None)):
      return True

  return False

def node_does_known_resize(dataset):
  in_types, in_shapes = get_ds_dtypes_shapes(dataset)
  out_types, out_shapes = get_ds_dtypes_shapes(dataset._input_dataset)

  if 'dict' in in_types[1:] or 'dict' in out_types[1:]:
    print('Nested dicts are currently not supported')
    return False
  
  if (len(in_shapes) != len(out_shapes)):
    print("Dimensions changed, do not reorder")
    return False

  if (in_shapes == out_shapes):
    print("In/out shapes were identical")
    return False

  for i in range(len(in_shapes)):
    if (in_shapes[i] != out_shapes[i] and out_shapes[i] != None and in_shapes[i] != None):
      return True

  return False

def node_increased_size(dataset):
  _, in_shapes = get_ds_dtypes_shapes(dataset._input_dataset)
  _, out_shapes = get_ds_dtypes_shapes(dataset)

  for i in range(len(in_shapes)):
    if in_shapes[i] < out_shapes[i]:
      return True
  return False

def may_reorder(dataset):
  # For now just check whether the user didn't set the keep_position flag to True
  if hasattr(dataset, '_keep_position'):
    return not dataset._keep_position
  else:
    return True

def get_source_ds(dataset):
  if hasattr(dataset, '_input_dataset'):
    return get_source_ds(dataset)
  return dataset