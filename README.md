# GraphBackdoor

This is a light-weight implementation of our **USENIX Security'21** paper **[Graph Backdoor](https://arxiv.org/abs/2006.11890)**. To be convenient for relevant projects, we simplify following functionalities with a higher running efficiency:

- **GNNs**: now we use DGL-based framework to implement our GNN, which has better memory occupation and running speed. For more information about DGL, see **Useful resources**.
- **graph encoding**: using pretrained attention network causes additional time cost. We find that directly aggregating input-space (feature/topology) matrices can also lead to a good input representation. Please see `./trojan/input.py`
- **blending function**: re-searching a subgraph to blend trigger has high cost especially on large graphs. Instead, one can always blend a generated trigger in a fixed region.
- **optimization objective**: we find the output-end optimization (based on labels) can realize similar attack efficacy comparing with imtermediate activations, but can significantly simplify the implementation. Thus we change to use label-level objective.

If you aim to compare the performance between this work and your novel attacks, or develop a defense against it, feel free to use this release on your work due to its easier accessibility and higher efficiency.

## Guide

We organize the structure of our files as follows:
```latex
.
├──  dataset/                  # keep all original dataset you may use
├──  main/
│   ├──  attack.py             # end-to-end attack codes
│   ├──  benign.py             # benign training/evaluation codes
│   └──  example.sh            # examples of running commands
├──  model/
│   ├──  gcn.py                # dgl-based GCN
│   └──  sage.py               # dgl-based GraphSAGE
├──  save/                     # temporary dir to save your trained models/perturbed data
├──  utils/
│   ├──  batch.py              # collate_batch function
│   ├──  bkdcdd.py             # codes to select victim graphs and trigger regions
│   ├──  datareader.sh         # data loader codes
│   ├──  graph.py              # simple utility function(s) related to graph processing
│   └──  mask.py               # the mask functions to scale graphs into same size or scale back
└──  config.py                 # attack configurations            

```

## Required packages
- torch   1.5.1
- dgl     0.4.2


## Useful resources:
- [TU graph kernel](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets): most of our datasets come from this source. In some cases, the we need to change the graph set such as remove some classes without too many instances, or remove graphs with small node scale.
- [DGL](https://docs.dgl.ai): we use DGL to implement our GNNs in this released version, because it has some high-efficient implementations such as [GCN](https://docs.dgl.ai/en/0.6.x/tutorials/models/1_gnn/1_gcn.html), [GAT](https://docs.dgl.ai/en/0.4.x/tutorials/models/1_gnn/9_gat.html), [GraphSAGE](https://github.com/dmlc/dgl/blob/master/examples/pytorch/graphsage/model.py).
- [TU graph datareader](https://github.com/bknyaz/graph_nn/blob/master/graph_unet.py): this repo implements a data loader to process TU graph datasets under their raw storage formats. Our `./utils/datareader.py` and `./utils/batch.py` contain the modified codes and we appreciate the authors' efforts!


## Run the code
You can directly run the attack by `python -u ./main/attack.py --use_org_node_attr --train_verbose --dataset <used dataset in ./dataset/> --target_class <set up a targeted class>`. We put some example commands in   `./main/example.sh`.


## Cite
Please cite our paper if it is helpful in your own work:
```
@inproceedings{xi2021graph,
  title={Graph backdoor},
  author={Xi, Zhaohan and Pang, Ren and Ji, Shouling and Wang, Ting},
  booktitle={30th $\{$USENIX$\}$ Security Symposium ($\{$USENIX$\}$ Security 21)},
  year={2021}
}
```


## Future release
Later we will release codes under transfer learning setting, now still working on them.