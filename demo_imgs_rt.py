import sys
sys.path.append('core_rt')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from core_rt.rt_FFLONet import FFLONet
from core_rt.utils.utils import InputPadder, count_parameters
from PIL import Image
from matplotlib import pyplot as plt
import argparse
import logging
import time

def load_image(imfile: str) -> torch.Tensor:
    img = np.array(Image.open(imfile).convert('RGB')).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(args.device)


def demo(args):
    logging.info("Loading checkpoint...")
    model = torch.nn.DataParallel(FFLONet(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt, weights_only=True))
    logging.info(f"Done loading checkpoint.")
    model = model.module
    model.to(args.device)
    model.eval()
    logging.info("Parameter Count: %d" % count_parameters(model))

    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    with torch.no_grad():
        left_images = sorted(glob.glob(args.left_imgs, recursive=True))
        right_images = sorted(glob.glob(args.right_imgs, recursive=True))
        logging.info(f"Found {len(left_images)} images. Saving files to {output_directory}/")

        for (imfile1, imfile2) in tqdm(list(zip(left_images, right_images))):
            imfile1 = imfile1.replace('\\', '/')
            imfile2 = imfile2.replace('\\', '/')
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2) 

            start_time = time.time()
            disp = model(image1, image2, iters=args.valid_iters, test_mode=True)
            end_time = time.time()
            inference_time = end_time - start_time
            print(f"Inference time: {inference_time:.4f} seconds")
            disp = disp.cpu().numpy()
            disp = padder.unpad(disp)
            file_stem = os.path.basename(imfile1)
            file_stem = os.path.splitext(file_stem)[0]
            plt.imsave(output_directory / f"{file_stem}_rt.png", disp.squeeze(), cmap='jet')

    logging.info(f"Saving file {output_directory.absolute()}.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', default='./pretrained_models/sceneflow/RTFFLONet.pth', help="load the weights from a specific checkpoint")
    parser.add_argument('--left_imgs', default='./demo-imgs/*Left.png', help="path to all first (left) frames") 
    parser.add_argument('--right_imgs', default='./demo-imgs/*Right.png', help="path to all second (right) frames")
    parser.add_argument('--output_directory', default='./demo-output', help="directory to save output")
    parser.add_argument('--mixed_precision', action='store_true', help="use mixed precision")
    parser.add_argument('--valid_iters', type=int, default=32, help="number of flow-field updates during validation forward pass")
    parser.add_argument('--max_disp', type=int, default=192, help="max disp of geometry encoding volume")
    parser.add_argument('--device', default = 'cuda:0', help='cuda device, i.e. 0 or 0,1,2,3')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')
    Path(args.output_directory).mkdir(exist_ok=True, parents=True)
    demo(args)
