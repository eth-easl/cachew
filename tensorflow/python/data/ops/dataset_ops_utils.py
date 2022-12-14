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

dtypes_by_bytes = ["<dtype: 'tf2.int8'>",
                   "<dtype: 'tf2.bfloat16'>",
                   "<dtype: 'tf2.float16'>",
                   "<dtype: 'tf2.int16'>",
                   "<dtype: 'float32'>",
                   "<dtype: 'tf2.int32'>",
                   "<dtype: 'tf2.float64'>",
                   "<dtype: 'tf2.int64'>"
                   ]

def get_ds_dtypes_shapes(dataset):
  types = []
  shapes = []

  elem_spec = dataset.element_spec
  if isinstance(elem_spec, tuple):
    num_elems = len(dataset.element_spec)
    for i in range(num_elems):
      types.append(elem_spec[i].dtype)
      shapes += list(elem_spec[i].shape)
  else:
    types.append(elem_spec.dtype)
    cur_s = list(elem_spec.shapes)
    shapes += cur_s
  return types, shapes

def should_reorder(org_types, org_shapes, new_types, new_shapes):
  if org_shapes != new_shapes:
    # This op changes the shape => not a casting op
    return False
  else:
    for t in org_types:
      if t not in dtypes_by_bytes:
        return False
    for t in new_types:
      if t not in dtypes_by_bytes:
        return False
    if org_types != new_types:
      for i in range(len(org_types)):
        # At least some component changed to a 'cheaper' dtype
        if dtypes_by_bytes.index(str(new_types[i])) < dtypes_by_bytes.index(str(org_types[i])):
          return True
      return False
    else:
      return False
