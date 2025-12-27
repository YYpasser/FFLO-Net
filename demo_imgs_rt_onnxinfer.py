import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
from core.utils.utils import InputPadder
import numpy as np
from PIL import Image
import torch
from matplotlib import pyplot as plt
import onnxruntime
import netron
import glob
from tqdm import tqdm
from pathlib import Path
import argparse
import onnxruntime as ort


def load_image(imfile):
    img = np.array(Image.open(imfile).convert('RGB')).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None]

def demo(args):
    providers = [
        ('CUDAExecutionProvider', {
            'device_id': 0,
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'gpu_mem_limit': 8 * 1024 * 1024 * 1024,
            'cudnn_conv_algo_search': 'DEFAULT',
            'do_copy_in_default_stream': True,
        })]
    sess_options = onnxruntime.SessionOptions()
    sess = onnxruntime.InferenceSession(args.onnx_model,sess_options=sess_options,providers = providers)
    # netron.start(f)
    left_images = sorted(glob.glob(args.left_imgs, recursive=True))
    right_images = sorted(glob.glob(args.right_imgs, recursive=True))
    output_directory = Path("./demo-output")
    for (imfile1, imfile2) in tqdm(list(zip(left_images, right_images))):
        imfile1 = imfile1.replace('\\', '/')
        imfile2 = imfile2.replace('\\', '/')
        image1 = load_image(imfile1)
        image2 = load_image(imfile2)

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2) 

        results_ort = sess.run(['disparity'],
                            {'leftImage' : image1.numpy(),
                            'rightImage': image2.numpy()})
        disp = np.array(results_ort[0])
        disp = padder.unpad(disp)
        file_stem = os.path.basename(imfile1)
        file_stem = os.path.splitext(file_stem)[0]
        plt.imsave(output_directory / f"{file_stem}_rt_dynamic.png", disp.squeeze(), cmap='jet')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--left_imgs', default='./demo-imgs/*Left.png', help="path to all first (left) frames") 
    parser.add_argument('--right_imgs', default='./demo-imgs/*Right.png', help="path to all second (right) frames")
    parser.add_argument('--output_directory', default='./demo-output', help='output directory')
    parser.add_argument('--onnx_model', default='./onnx_model/RTFFLONetDynamic.onnx', help='path to model')
    args = parser.parse_args()
    ort.set_default_logger_severity(3)
    demo(args)