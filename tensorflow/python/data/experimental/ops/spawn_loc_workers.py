from __future__ import absolute_import

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.util.tf_export import tf_export




@tf_export("data.experimental.spawn_loc_workers")
def spawn_loc_workers(workers=1
                      disptacher='localhost'):

    '''
    A function for spawning local workers under the hood.

    Args:
      workers: the number of workers you want to spawn
      disptacher: the name of the dispatcher
    '''

    loc_workers = []

print(f"Spawning {num_local_workers} Local workers to {dispatcher_target}")
for idx in range(num_local_workers):
    workers.append(
        tf.data.experimental.service.WorkerServer(
            tf.data.experimental.service.WorkerConfig(
                dispatcher_address=dispatcher_target,
                heartbeat_interval_ms=1000,
                # port=38000 + idx
            )
        )
    )

    return loc_workers