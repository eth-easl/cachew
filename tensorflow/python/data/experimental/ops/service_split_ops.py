from __future__ import absolute_import

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.util.tf_export import tf_export


class _SplitMarkerDataset(dataset_ops.UnaryUnchangedStructureDataset):
  """
  A dataset that allows to add a marker node in the graph representation.
  used for pipeline splitting
  """

  def __init__(self, input_dataset):

    self._input_dataset = input_dataset

    variant_tensor = ged_ops.marker_dataset(
      self._input_dataset._variant_tensor,  # pylint: disable=protected-access
      **self._flat_structure)

    super(_SplitMarkerDataset, self).__init__(input_dataset, variant_tensor)


  def _transformation_name(self):
    return "Dataset.splitMarkerDataset"


@tf_export("data.experimental.split_mark")
def service_split_mark():
  def _apply_fn(dataset):
    """Actual dataset transformation."""
    project_func = None
    dataset = _SplitMarkerDataset(
      input_dataset=dataset)
    return dataset

  return _apply_fn