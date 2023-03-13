from __future__ import absolute_import

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.data.experimental.service import server_lib

@tf_export("data.experimental.spawn_loc_workers")
def spawn_loc_workers(workers=1,
                      disptacher='localhost'):

    '''
    A function for spawning local workers under the hood.

    Args:
      workers: the number of workers you want to spawn
      disptacher: the name of the dispatcher
    '''

    loc_workers = []

    print(f"Spawning {workers} Local workers to {disptacher}")
    for idx in range(workers):
        workers.append(
            server_lib.WorkerServer(
                server_lib.WorkerConfig(
                    dispatcher_address=disptacher,
                    heartbeat_interval_ms=1000,
                    # port=38000 + idx
                )
            )
        )

    return loc_workers