# python inference_realesrgan.py -n RealESRGAN_x4plus -i input_cus/low_resolution_images/gen/gen.npz -n RealESRGAN_x4plus -o input_cus/low_resolution_images/gen/ --face_enhance

# python inference_realesrgan.py -n RealESRGAN_x2plus -i input_cus/privimaged_lr_64/gen/gen.npz -n RealESRGAN_x2plus -o input_cus/privimaged_hr_128/gen/ --face_enhance

# python low_to_high_dm.py -m PE -ip Real-ESRGAN/input_cus/pdp-diffusion_lr

python low_to_high_dm.py -m PE -ip Real-ESRGAN/input_cus/privimaged_lr_64 model.api_params.num_channels=128 model.api_params.class_cond=false model.api_params.rescale_learned_sigmas=true model.api_params.rescale_timesteps=true model.api_params.model_path=/p/fzv6enresearch/PE-Refine/models/pretrained_models/imagenet64_uncond_100M_1500K.pt -vd 40

# python eval.py --method PrivImage --data_name celeba_male_32 --epsilon 10.0 --exp_path exp/pe/celeba_male_32_eps10.0_vd90-2025-04-30-16-55-26

python run.py -m PE -dn cifar10_32 train.initial_sample=/p/fzv6enresearch/DPImageBench/exp/privimage/cifar10_32_eps10.0val_cn1e-3-2024-12-03-00-13-54/gen/gen.npz -ed pe+privimage

CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py -m PE setup.n_gpus_per_node=4 -dn mnist_28 -e 1.0 train.initial_sample=/p/fzv6enresearch/DPImageBench/exp/dp-feta/mnist_28_eps10.0val_central_mean-2025-03-19-07-56-07/gen/gen.npz -ed pe+privimage

python run.py -m PE-SD -dn mnist_28 -e 1.0 train.initial_sample=/p/fzv6enresearch/DPImageBench/exp/dp-feta/mnist_28_eps10.0val_central_mean-2025-03-19-07-56-07/gen/gen.npz model.api_params.ckpt=/p/fzv6enresearch/DPImageBench/exp/pdp-diffusion/cifar10_32_eps10.0pretrain_ch1224_nf128-2024-12-16-00-22-27/pretrain/checkpoints/final_checkpoint.pth model.api_params.network.ch_mult=[1,2,2,4] model.api_params.network.attn_resolutions=[16,8,4] model.api_params.network.nf=128 model.api_params.batch_size=6000 -ed pe+feta_77M