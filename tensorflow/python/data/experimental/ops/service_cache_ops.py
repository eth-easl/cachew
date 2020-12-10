from __future__ import absolute_import


import multiprocessing

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export

class _ServiceCachePutDataset(dataset_ops.UnaryUnchangedStructureDataset):
    """A dataset that allows transparent store to disk."""

    def __init__(self, input_dataset, path):

        self._input_dataset = input_dataset
        self._path = path

        variant_tensor = ged_ops.service_cache_put_dataset(
            input_dataset._variant_tensor,  # pylint: disable=protected-access
            path,
            **self._flat_structure)
        super(_ServiceCachePutDataset, self).__init__(input_dataset, variant_tensor)

    def _functions(self):
        return [self._reader_func, self._shard_func]

    def _transformation_name(self):
        return "Dataset.serviceCachePutDataset"


#@tf_export("data.experimental.service_cache_put")
def service_cache_put(path):

    def _apply_fn(dataset):
        """Actual dataset transformation."""
        project_func = None
        dataset = _ServiceCachePutDataset( input_dataset=dataset, path=path)

        return dataset

    return _apply_fn