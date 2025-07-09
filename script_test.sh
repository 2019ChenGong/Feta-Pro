# python inference_realesrgan.py -n RealESRGAN_x4plus -i input_cus/low_resolution_images/gen/gen.npz -n RealESRGAN_x4plus -o input_cus/low_resolution_images/gen/ --face_enhance

# python inference_realesrgan.py -n RealESRGAN_x2plus -i input_cus/privimaged_lr_64/gen/gen.npz -n RealESRGAN_x2plus -o input_cus/privimaged_hr_128/gen/ --face_enhance

# python low_to_high_dm.py -m PE -ip Real-ESRGAN/input_cus/pdp-diffusion_lr

# python low_to_high_dm.py -m PE -ip Real-ESRGAN/input_cus/privimaged_lr_64 model.api_params.num_channels=128 model.api_params.class_cond=false model.api_params.rescale_learned_sigmas=true model.api_params.rescale_timesteps=true model.api_params.model_path=/p/fzv6enresearch/PE-Refine/models/pretrained_models/imagenet64_uncond_100M_1500K.pt -vd 40

# # python eval.py --method PrivImage --data_name celeba_male_32 --epsilon 10.0 --exp_path exp/pe/celeba_male_32_eps10.0_vd90-2025-04-30-16-55-26

# python run.py -m PE -dn cifar10_32 train.initial_sample=/p/fzv6enresearch/DPImageBench/exp/privimage/cifar10_32_eps10.0val_cn1e-3-2024-12-03-00-13-54/gen/gen.npz -ed pe+privimage

# CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py -m PE setup.n_gpus_per_node=4 -dn mnist_28 -e 1.0 train.initial_sample=/p/fzv6enresearch/DPImageBench/exp/dp-feta/mnist_28_eps10.0val_central_mean-2025-03-19-07-56-07/gen/gen.npz -ed pe+privimage

# python run.py -m PE-SD -dn mnist_28 -e 1.0 train.initial_sample=/p/fzv6enresearch/DPImageBench/exp/dp-feta/mnist_28_eps10.0val_central_mean-2025-03-19-07-56-07/gen/gen.npz model.api_params.ckpt=/p/fzv6enresearch/DPImageBench/exp/pdp-diffusion/cifar10_32_eps10.0pretrain_ch1224_nf128-2024-12-16-00-22-27/pretrain/checkpoints/final_checkpoint.pth model.api_params.network.ch_mult=[1,2,2,4] model.api_params.network.attn_resolutions=[16,8,4] model.api_params.network.nf=128 model.api_params.batch_size=6000 -ed pe+feta_77M

# python run.py -m PE-SD -dn cifar10_32 -e 1.0 train.initial_sample=/p/fzv6enresearch/DPImageBench/exp/dp-feta/cifar10_32_eps10.0val_central_mean-2025-03-23-07-47-44/gen/gen.npz model.api_params.ckpt=/p/fzv6enresearch/DPImageBench/exp/pdp-diffusion/cifar10_32_eps10.0pretrain_ch1224_nf128-2024-12-16-00-22-27/pretrain/checkpoints/final_checkpoint.pth model.api_params.network.ch_mult=[1,2,2,4] model.api_params.network.attn_resolutions=[16,8,4] model.api_params.network.nf=128 model.api_params.batch_size=6000 -ed pe+feta_77M

# python run.py -m PE -dn cifar10_32 -e 1.0 train.initial_sample=/p/fzv6enresearch/DPImageBench/exp/dp-feta/cifar10_32_eps10.0val_central_mean-2025-03-23-07-47-44/gen/gen.npz -ed pe+privimage model.api_params.num_channels=128 model.api_params.class_cond=false model.api_params.rescale_learned_sigmas=true model.api_params.rescale_timesteps=true model.api_params.model_path=/p/fzv6enresearch/PE-Refine/models/pretrained_models/imagenet64_uncond_100M_1500K.pt

# # feta-pretrain_merf-pretrain_dpdm
# CUDA_VISIBLE_DEVICES=0,1,3 python run.py setup.n_gpus_per_node=3 eval.mode=val model.ckpt=/p/fzv6enresearch/DPImageBench/exp/dp-feta/celeba_male_32_eps1.0val_central_pre-2025-03-20-23-04-23/pretrain/checkpoints/final_checkpoint.pth train.dp.privacy_history=[[10,0.07,250]] public_data.name=npz public_data.train_path=/p/fzv6enresearch/DPImageBench/exp/dp-merf/celeba_male_32_eps1.0trainval-2024-10-22-03-29-00/gen/gen.npz pretrain.n_epochs=10 pretrain.batch_size=256 -m DPDM -dn celeba_male_32 -e 9.0 -ed val_eps1merf_from_feta

# python run.py setup.n_gpus_per_node=4 eval.mode=val model.ckpt=/p/fzv6enresearch/DPImageBench/exp/dp-feta/camelyon_32_eps1.0val_central_pre-2025-03-21-00-34-59/pretrain/checkpoints/final_checkpoint.pth train.dp.privacy_history=[[10,0.04,250]] public_data.name=npz public_data.train_path=/p/fzv6enresearch/DPImageBench/exp/dp-merf/camelyon_32_eps1.0trainval-2024-10-20-06-35-05/gen/gen.npz pretrain.n_epochs=10 pretrain.batch_size=256 -m DPDM -dn camelyon_32 -e 9.0 -ed val_eps1merf_from_feta

# python run.py setup.n_gpus_per_node=4 eval.mode=val model.ckpt=/p/fzv6enresearch/DPImageBench/exp/dp-feta/mnist_28_eps1.0sen_central_pre-2025-03-17-10-41-15/pretrain/checkpoints/final_checkpoint.pth train.dp.privacy_history=[[5,0.1,5]] public_data.name=npz public_data.train_path=/p/fzv6enresearch/DPImageBench/exp/dp-merf/mnist_28_eps1.0trainval-2024-10-20-06-27-04/gen/gen.npz pretrain.n_epochs=10 pretrain.batch_size=256 train.batch_size=3072 -m DPDM -dn mnist_28 -e 9.0 -ed val_eps1merf_from_feta_syn4train_100epoch

# python run.py setup.n_gpus_per_node=4 eval.mode=val model.ckpt=/p/fzv6enresearch/DPImageBench/exp/dp-feta/fmnist_28_eps1.0val_central_pre-2025-03-19-03-42-45/pretrain/checkpoints/final_checkpoint.pth train.dp.privacy_history=[[5,0.1,5]] public_data.name=npz public_data.train_path=/p/fzv6enresearch/DPImageBench/exp/dp-merf/fmnist_28_eps1.0trainval-2024-10-20-06-27-04/gen/gen.npz pretrain.n_epochs=10 pretrain.batch_size=256 train.batch_size=3072 -m DPDM -dn fmnist_28 -e 9.0 -ed val_eps1merf_from_feta_syn4train

# # dpsgd+pe
# python run.py -m PE-SD -dn fmnist_28 -e 5.0 train.initial_sample=/p/fzv6enresearch/DPImageBench/exp/dp-feta/fmnist_28_eps5.0val-2025-03-29-04-18-47/gen/gen.npz model.api_params.ckpt=/p/fzv6enresearch/DPImageBench/exp/pdp-diffusion/cifar10_32_eps10.0pretrain_ch1224_nf128-2024-12-16-00-22-27/pretrain/checkpoints/final_checkpoint.pth model.api_params.network.ch_mult=[1,2,2,4] model.api_params.network.attn_resolutions=[16,8,4] model.api_params.network.nf=128 model.api_params.batch_size=6000 -ed pe+feta_77M+finalselect

# python run.py public_data.name=null -m DP-MERF -dn mnist_28 -e 0.1
# python run.py public_data.name=null -m DP-MERF -dn fmnist_28 -e 0.1
# python run.py public_data.name=null -m DP-MERF -dn cifar10_32 -e 0.1
# python run.py public_data.name=null -m DP-MERF -dn celeba_male_32 -e 0.1
# python run.py public_data.name=null -m DP-MERF -dn camelyon_32 -e 0.1

# python run.py setup.n_gpus_per_node=4 eval.mode=val public_data.name=npz public_data.train_path=/p/fzv6enresearch/PE-Refine/exp/dp-merf/mnist_28_eps0.1-2025-05-21-01-28-05/gen/gen.npz pretrain.n_epochs=10 pretrain.batch_size=256 -m DPDM -dn mnist_28 -e 0.1 -ed val_eps0.1merf

# python run.py setup.n_gpus_per_node=4 eval.mode=val public_data.name=npz public_data.train_path=/p/fzv6enresearch/PE-Refine/exp/dp-merf/fmnist_28_eps0.1-2025-05-21-01-29-08/gen/gen.npz pretrain.n_epochs=10 pretrain.batch_size=256 -m DPDM -dn fmnist_28 -e 0.1 -ed val_eps0.1merf

# python run.py setup.n_gpus_per_node=4 eval.mode=val public_data.name=npz public_data.train_path=/p/fzv6enresearch/PE-Refine/exp/dp-merf/cifar10_32_eps0.1-2025-05-21-01-30-17/gen/gen.npz pretrain.n_epochs=10 pretrain.batch_size=256 -m DPDM -dn cifar10_32 -e 0.1 -ed val_eps0.1merf

# python run.py setup.n_gpus_per_node=4 eval.mode=val public_data.name=npz public_data.train_path=/p/fzv6enresearch/PE-Refine/exp/dp-merf/celeba_male_32_eps0.1-2025-05-21-01-31-54/gen/gen.npz pretrain.n_epochs=10 pretrain.batch_size=256 -m DPDM -dn celeba_male_32 -e 0.1 -ed val_eps0.1merf

# python run.py setup.n_gpus_per_node=4 eval.mode=val public_data.name=npz public_data.train_path=/p/fzv6enresearch/PE-Refine/exp/dp-merf/camelyon_32_eps0.1-2025-05-21-01-36-08/gen/gen.npz pretrain.n_epochs=10 pretrain.batch_size=256 -m DPDM -dn camelyon_32 -e 0.1 -ed val_eps0.1merf

# python run.py setup.n_gpus_per_node=4 eval.mode=val pretrain.n_epochs1=1000 pretrain.n_epochs2=10 pretrain.batch_size1=50 pretrain.batch_size2=256 train.merf.dp.epsilon=1.0 public_data.central.sigma=5 -m DP-FETA2 -dn mnist_28 -e 10.0 -ed val

# merf only

# python run.py setup.n_gpus_per_node=4 setup.master_port=6660 eval.mode=val pretrain.n_epochs1=1000 pretrain.n_epochs2=10 pretrain.batch_size1=50 pretrain.batch_size2=256 train.merf.dp.epsilon=0.25 public_data.central.sigma=20 pretrain.mode=merf -m DP-FETA2 -dn mnist_28 -e 1.0 -ed val_merf0.25_merfonly

# python run.py setup.n_gpus_per_node=4 setup.master_port=6660 eval.mode=val pretrain.n_epochs1=1000 pretrain.n_epochs2=10 pretrain.batch_size1=50 pretrain.batch_size2=256 train.merf.dp.epsilon=0.25 public_data.central.sigma=20 pretrain.mode=merf -m DP-FETA2 -dn fmnist_28 -e 1.0 -ed val_merf0.25_merfonly

# python run.py setup.n_gpus_per_node=4 setup.master_port=6660 eval.mode=val pretrain.n_epochs1=1000 pretrain.n_epochs2=10 pretrain.batch_size1=50 pretrain.batch_size2=256 train.merf.dp.epsilon=0.2 public_data.central.sigma=20 pretrain.mode=merf -m DP-FETA2 -dn celeba_male_32 -e 1.0 -ed val_merf0.2_merfonly

# python run.py setup.n_gpus_per_node=4 setup.master_port=6660 eval.mode=val pretrain.n_epochs1=1000 pretrain.n_epochs2=10 pretrain.batch_size1=50 pretrain.batch_size2=256 train.merf.dp.epsilon=0.15 public_data.central.sigma=20 pretrain.mode=merf -m DP-FETA2 -dn camelyon_32 -e 1.0 -ed val_merf0.15_merfonly

# mix

# python run.py setup.n_gpus_per_node=4 setup.master_port=6660 eval.mode=val pretrain.n_epochs1=1000 pretrain.n_epochs2=10 pretrain.batch_size1=50 pretrain.batch_size2=256 train.merf.dp.epsilon=0.25 public_data.central.sigma=20 pretrain.mode=feta_merf_mix -m DP-FETA2 -dn mnist_28 -e 1.0 -ed val_fetasigma20_merf0.25_mix

# python run.py setup.n_gpus_per_node=4 setup.master_port=6660 eval.mode=val pretrain.n_epochs1=1000 pretrain.n_epochs2=10 pretrain.batch_size1=50 pretrain.batch_size2=256 train.merf.dp.epsilon=0.25 public_data.central.sigma=20 pretrain.mode=feta_merf_mix -m DP-FETA2 -dn fmnist_28 -e 1.0 -ed val_fetasigma20_merf0.25_mix

# python run.py setup.n_gpus_per_node=4 setup.master_port=6660 eval.mode=val pretrain.n_epochs1=1000 pretrain.n_epochs2=10 pretrain.batch_size1=50 pretrain.batch_size2=256 train.merf.dp.epsilon=0.2 public_data.central.sigma=15 pretrain.mode=feta_merf_mix -m DP-FETA2 -dn celeba_male_32 -e 1.0 -ed val_fetasigma15_merf0.2_mix

# python run.py setup.n_gpus_per_node=4 setup.master_port=6660 eval.mode=val pretrain.n_epochs1=1000 pretrain.n_epochs2=10 pretrain.batch_size1=50 pretrain.batch_size2=256 train.merf.dp.epsilon=0.15 public_data.central.sigma=10 pretrain.mode=feta_merf_mix -m DP-FETA2 -dn camelyon_32 -e 1.0 -ed val_fetasigma10_merf0.15_mix


python run.py setup.n_gpus_per_node=4 setup.master_port=6660 eval.mode=val pretrain.n_epochs1=1000 pretrain.n_epochs2=10 pretrain.batch_size1=50 pretrain.batch_size2=256 train.merf.dp.epsilon=0.25 public_data.central.sigma=20 pretrain.mode=feta_merf train.cut_noise=true train.max_sigma=20 -m DP-FETA2 -dn mnist_28 -e 1.0 -ed val_merf0.25_merfonly_cutnoise

python run.py setup.n_gpus_per_node=4 eval.mode=val pretrain.n_epochs1=1000 pretrain.
n_epochs2=10 pretrain.batch_size1=50 pretrain.batch_size2=256 train.merf.dp.epsilon=0.05 public_data.central.sigma=20 pretrain.mode=feta_merf -m DP
-FETA2 -dn fmnist_28 -e 1.0 -ed val_merfeps1.0_fetasigma20_merf0.05