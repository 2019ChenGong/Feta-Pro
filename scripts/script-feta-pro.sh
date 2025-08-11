# FETA-Pro 
python run.py setup.n_gpus_per_node=3 setup.master_port=6662 eval.mode=val -m DP-FETA-Pro -dn mnist_28 -e 1.0 -ed val_test

# FETA
python run.py setup.n_gpus_per_node=2 setup.master_port=6662 eval.mode=val -m DP-FETA -dn mnist_28 -e 1.0 -ed val_test

# FETA-Pro-mix
python run.py setup.n_gpus_per_node=3 setup.master_port=6662 eval.mode=val pretrain.mode=mix -m DP-FETA-Pro -dn mnist_28 -e 1.0 -ed val_test

# DP-MERF
python run.py setup.n_gpus_per_node=1 setup.master_port=6662 eval.mode=val -m DP-MERF -dn mnist_28 -e 1.0 -ed val_test

# DPDM
python run.py setup.n_gpus_per_node=3 setup.master_port=6662 eval.mode=val -m DPDM -dn mnist_28 -e 1.0 -ed val_test

# RQ3, change the privacy allocation plans
python run.py setup.n_gpus_per_node=4 --method DP-FETA-Pro --data_name mnist_28 -e 1.0 eval.mode=val train.sigma_freq=26.6 train.sigma_time=20

# Eval
CUDA_VISIBLE_DEVICES=0 python eval.py --method DP-FETA-Pro --data_name cifar10_32 --epsilon 10.0 --exp_path exp/dp-feta-pro/cifar10_32_eps10.0default_500_0.005-2025-08-09-01-47-59