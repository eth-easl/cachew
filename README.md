![cachew-logo](docs/figures/cachew_logo.png)

# Pecan: Cost-Efficient ML Data Preprocessing with Automatic Transformation
Ordering and Hybrid Placement

Pecan is a multi-tenant service for cost-efficient input data processing in machine learning jobs. 

To minimize end-to-end ML training cost, Pecan utilizes two policies: 
1) Pecan’s AutoPlacement policy scales data preprocessing workers and places them across local and remote resources to minimize cost.
2) Pecan's AutoOrder policy transparently reorders input pipeline transformations to maximize per-worker throughput.

Pecan builds on top of the [Cachew](https://www.usenix.org/system/files/atc22-graur.pdf) data loading framework in [TensorFlow](https://github.com/tensorflow/tensorflow), extending [tf.data service](https://www.tensorflow.org/api_docs/python/tf/data/experimental/service) with the AutoPlacement and AutoOrder policies. 

**This repository is a fork of TensorFlow with the source code for Cachew.**


## Cachew System Architecture

Cachew consists of a centralized dispatcher, a dynamic number of input data workers, and a disaggregated storage cluster for data caching.

![cachew-system-architecture](docs/figures/cachew_system_arch.png?raw=true)

Users register training nodes (i.e., clients) with the Cachew dispatcher. To execute an input pipeline with Cachew, clients provide a graph representation of the input pipeline and a path to the input dataset in a cloud storage bucket. Cachew supports and extends the tf.data API for defining input data pipelines from a collection of composable and user-parametrizable operators. Users can annotate their tf.data input pipeline to mark candidate locations for caching/reusing data across executions. Cachew will automatically apply caching at the throughput-optimal location in the input pipeline among the candidate locations. 

Cachew's input data workers are stateless components responsible for producing batches of preprocessed data for clients. The dispatcher dynamically adjusts the number of input data workers for each job to minimize epoch time while keeping costs low. The dispatcher also profiles and maintains metadata about input pipeline executions across jobs to make data caching decisions. Cachew stores cached datasets in a GlusterFS remote storage cluster. 

Clients fetch data from the workers that are assigned to them by the dispatcher. Clients and workers periodically send heartbeats to the dispatcher to maintain membership in the service and provide metrics used for the autoscaling and autocaching policies.


## Deploying and Using Cachew

The [cachew_experiments](https://github.com/eth-easl/cachew_experiments) repository provides scripts and instructions to get started with a Cachew deployment and execute example ML input data pipelines. The repository also provides detailed instructions for reproducing the key results from the Cachew research paper published at USENIX ATC'22. 

## Contributing

We welcome contributions and PRs to Cachew.

 
## Referencing our work

Cachew will appear at USENIX ATC'22. If you decide to use Cachew in your work, please cite our paper: 

```
@inproceedings{cachew,
  author    = {Dan Graur and
               Damien Aymon and
               Dan Kluser and
               Tanguy Albrici and
               Chandramohan A. Thekkath and
               Ana Klimovic},
  title     = {Cachew: Machine Learning Input Data Processing as a Service},
  booktitle = {Proceedings of the USENIX Annual Technical Confernece (ATC'22)},
  publisher = {{USENIX}},
  year      = {2022},
}
```

