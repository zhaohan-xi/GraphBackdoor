python benign.py --use_org_node_attr --train_verbose

nohup python -u attack.py --use_org_node_attr --train_verbose --target_class 0 --train_epochs 20 > ../attack.log 2>&1 &