# python inference_realesrgan.py -n RealESRGAN_x4plus -i input_cus/low_resolution_images/gen/gen.npz -n RealESRGAN_x4plus -o input_cus/low_resolution_images/gen/ --face_enhance

# python inference_realesrgan.py -n RealESRGAN_x2plus -i input_cus/privimaged_lr_64/gen/gen.npz -n RealESRGAN_x2plus -o input_cus/privimaged_hr_128/gen/ --face_enhance

# python low_to_high_dm.py -m PE -ip Real-ESRGAN/input_cus/pdp-diffusion_lr

python low_to_high_dm.py -m PE -ip Real-ESRGAN/input_cus/privimaged_lr_64 model.api_params.num_channels=128 model.api_params.class_cond=false model.api_params.rescale_learned_sigmas=true model.api_params.rescale_timesteps=true model.api_params.model_path=/p/fzv6enresearch/PE-Refine/models/pretrained_models/imagenet64_uncond_100M_1500K.pt -vd 40

# python eval.py --method PrivImage --data_name celeba_male_32 --epsilon 10.0 --exp_path exp/pe/celeba_male_32_eps10.0_vd90-2025-04-30-16-55-26

python run.py -m PE -dn cifar10_32 train.initial_sample=/p/fzv6enresearch/DPImageBench/exp/privimage/cifar10_32_eps10.0val_cn1e-3-2024-12-03-00-13-54/gen/gen.npz -ed pe+privimage