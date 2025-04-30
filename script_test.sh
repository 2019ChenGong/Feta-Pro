python inference_realesrgan.py -n RealESRGAN_x4plus -i input_cus/low_resolution_images/gen/gen.npz -n RealESRGAN_x4plus -o input_cus/low_resolution_images/gen/ --face_enhance

python inference_realesrgan.py -n RealESRGAN_x2plus -i input_cus/privimaged_lr_64/gen/gen.npz -n RealESRGAN_x2plus -o input_cus/privimaged_hr_128/gen/ --face_enhance

python low_to_high_dm.py --exp_path Real-ESRGAN/input_cus/pdp-diffusion_lr