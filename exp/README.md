# Results Structure

We outline the structure of the results files as follows. The training and evaluations results are recorded in the file `exp`. For example, if users leverage the DP-FETA-Pro method to generate synthetic images for the MNIST dataset under a privacy budget of `eps=1.0`, the structure of the folder is as follows:

```plaintext
exp/                                  
├── dp-feta-pro/ 
│   └── mnist_28_eps1.0-2025-07-29-13-35-18/  
│           ├── gen  
│           │   ├── gen.npz 
│           │   └── sample.png 
│           ├── gen_freq  
│           │   ├── gen.npz 
│           │   └── sample.png 
│           ├── pretrain_time  
│           │   ├── checkpoints  
│           │   │   ├── final_checkpoint.pth  
│           │   │   └── snapshot_checkpoint.pth  
│           │   └── samples 
│           │       ├── iter_2000 
│           │       └── ... 
│           ├── pretrain_freq  
│           │   ├── checkpoints  
│           │   │   ├── final_checkpoint.pth  
│           │   │   └── snapshot_checkpoint.pth  
│           │   └── samples 
│           │       ├── iter_2000 
│           │       └── ... 
│           ├── train
│           │   ├── checkooints  
│           │   │   ├── final_checkpoint.pth  
│           │   │   └── snapshot_checkpoint.pth    
│           │   └── samples 
│           │       ├── iter_2000 
│           │       └── ... 
│           ├── train_freq
│           │   ├── checkooints  
│           │   │   ├── final_checkpoint.pth  
│           │   │   └── snapshot_checkpoint.pth    
│           │   └── samples 
│           │       ├── iter_2000 
│           │       └── ... 
│           └──stdout.txt   
├── pe/ 
└── privimage/  
```

We introduce the files as follows,

- `./gen/gen.npz`: the synthetic images.
- `./gen/sample.png`: the samples of synthetic images.
- `./gen_freq/gen.npz`: the synthetic images from the auxiliary generator using frequency features.
- `./gen_freq/sample.png`: the sample of synthetic images from the auxiliary generator.
- `./pretrain_time/checkpoints/final_checkpoint.pth`: the parameters of synthsizer pretrained using the time feature at the final epochs.
- `./pretrain_time/checkpoints/snapshot_checkpoint.pth`: we store the synthesizer's parameters at the current epoch after each iteration, deleting the previous parameters to manage storage efficiently.
- `./pretrain_time/samples/iter_2000`: the synthetic images under 2000 iterations for pretraining on the time feature.
- `./pretrain_freq/checkpoints/final_checkpoint.pth`: the parameters of synthsizer pretrained using the frequency feature at the final epochs.
- `./pretrain_freq/checkpoints/snapshot_checkpoint.pth`: we store the synthesizer's parameters at the current epoch after each iteration, deleting the previous parameters to manage storage efficiently.
- `./pretrain_freq/samples/iter_2000`: the synthetic images under 2000 iterations for pretraining on the frequency feature.
- `./train/checkpoints/final_checkpoint.pth`: the parameters of synthsizer at the final epochs.
- `./train/checkpoints/snapshot_checkpoint.pth`: we store the synthesizer's parameters at the current epoch after each iteration, deleting the previous parameters to manage storage efficiently.
- `./train/samples/iter_2000`: the synthetic images under 2000 iterations for training on sensitive datasets.
- `./stdout.txt`: the file used to record the training and evaluation results.

