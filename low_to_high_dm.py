import os
import numpy as np
import sys
import argparse
import datetime
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision.utils import save_image
from models.model_loader import load_model
from data.dataset_loader import load_data
from utils.utils import initialize_environment, run, parse_config

def main(config, opt):
    initialize_environment(config)

    model, config = load_model(config)

    # Load synthetic data
    syn_path = os.path.join(opt.input_path, 'gen', 'gen.npz')

    try:
        syn = np.load(syn_path)
        syn_data, syn_labels = syn["x"], syn["y"]
        print(f"Loaded syn_data shape: {syn_data.shape}")
    except FileNotFoundError:
        print(f"Error: File not found at {syn_path}")
        return
    except Exception as e:
        print(f"Error loading .npz file: {e}")
        return

    # Convert NumPy array to PyTorch tensor
    device = torch.device("cuda" if torch.cuda.is_available() and config.setup.get("use_cuda", True) else "cpu")
    syn_data = torch.from_numpy(syn_data).float().to(device)  # Ensure float32 and move to device

    # debug: random sample 2000
    # num_samples = 2000
    # indices = torch.randperm(syn_data.shape[0], device=device)[:num_samples]  # 随机索引
    # syn_data = syn_data[indices]

    # Interpolate to 64x64
    # syn_data = F.interpolate(syn_data, size=[64, 64])

    # Define label (ensure it's a tensor if required by model)
    # label = torch.tensor(syn_labels)  # Adjust based on model requirements
    label = None

    # Generate new samples
    with torch.no_grad():
        new_samples = model.api._image_variation(syn_data, label, variation_degree=config.model.variation_degree)  # variation_degree in (0, 100)

    if isinstance(new_samples, np.ndarray):
        new_samples = torch.from_numpy(new_samples).float().to(device)

    # Normalize to [0, 1] for saving
    new_samples = new_samples.clamp(0., 1.)
    

    # Save output
    # save_image(new_samples, 'temp.png')

    # new_samples = np.array(new_samples)

    # batch_size = 100
    # output_dir = "sampled_images"
    # os.makedirs(output_dir, exist_ok=True)
    # for i in range(0, num_samples, batch_size):
    #     batch = new_samples[i:i + batch_size]
    #     output_path = os.path.join(output_dir, f"sampled_batch_{i // batch_size}.png")
    #     save_image(batch, output_path, nrow=10) 

    # Save to npz file
    os.makedirs(os.path.join(config.setup.workdir, 'gen'))
    np.savez(os.path.join(config.setup.workdir, 'gen', "gen.npz"), x=new_samples.detach().cpu().numpy(), y=syn_labels)

    print('Finished')

if __name__ == '__main__':
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir', default="configs")
    parser.add_argument('--method', '-m', default="PE")
    parser.add_argument('--epsilon', '-e', default="10.0")
    parser.add_argument('--data_name', '-dn', default="celeba_male_32")
    parser.add_argument('--exp_description', '-ed', default="")
    parser.add_argument('--resume_exp', '-re', default=None)
    parser.add_argument('--config_suffix', '-cs', default="")
    parser.add_argument('--input_path', '-ip', default="")
    parser.add_argument('--variation_degree', '-vd', type=int, default=0)
    opt, unknown = parser.parse_known_args()
    config = parse_config(opt, unknown)
    config.model['variation_degree'] = opt.variation_degree

    nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if opt.resume_exp is not None:
        config.setup.workdir = "exp/{}/{}".format(str.lower(opt.method), opt.resume_exp)
    else:
        config.setup.workdir = "exp/{}/{}_eps{}{}{}-{}".format(str.lower(opt.method), opt.data_name, opt.epsilon, opt.config_suffix, opt.exp_description+'_vd'+str(opt.variation_degree), nowTime)

    run(lambda config: main(config, opt), config)