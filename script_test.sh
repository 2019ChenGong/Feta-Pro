# FETA-Pro 
python run.py setup.n_gpus_per_node=3 setup.master_port=6662 eval.mode=val -m DP-FETA-Pro -dn mnist_28 -e 1.0 -ed val_test

# FETA-Pro-mix
python run.py setup.n_gpus_per_node=3 setup.master_port=6662 eval.mode=val pretrain.mode=mix -m DP-FETA-Pro -dn mnist_28 -e 1.0 -ed val_test

# DP-MERF
python run.py setup.n_gpus_per_node=1 setup.master_port=6662 eval.mode=val -m DP-MERF -dn mnist_28 -e 1.0 -ed val_test

# DPDM
python run.py setup.n_gpus_per_node=3 setup.master_port=6662 eval.mode=val -m DPDM -dn mnist_28 -e 1.0 -ed val_test