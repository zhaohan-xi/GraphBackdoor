# GraphBackdoor

This is a light-weight implementation of our **USENIX Security'21** paper **[Graph Backdoor](https://arxiv.org/abs/2006.11890)**. We simplify following functionalities for a higher running efficiency:

- **graph encoding**: instead of using a pretrained attention network to encode graphs, we find that directly using input-space aggregation can also lead to a good input representation, i.e., directly aggregate feature matrix and topology matrix by 1-hop multiplication or 2-hop. See ./trojan/input.py
- **blending function**: re-searching a subgraph to blend trigger has high cost especially on large graphs. Instead, one can directly use a fixed locality as trigger-embedded region and always blend generated trigger in this region.
- **optimization objective**: we find the output-end optimization (based on labels) can realize similar attack efficacy comparing with imtermediate activations, but can significantly simplify the implementation. Thus we use label-level objective to optimize the attack.

If you aim to compare the performance between this work and your novel attacks, or develop a defense against it, you can directly use this version on your work due to its easier accessibility and higher efficiency.

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
- [TU graph datareader](https://github.com/bknyaz/graph_nn/blob/master/graph_unet.py): this repo implements a data loader to process TU graph datasets under their raw storage formats. We use the loading codes in `./utils/datareader.py`, `./utils/batch.py` and appreciate their efforts!


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
