from core_export.FFLONet import FFLONet
from core_export.utils.utils import InputPadder
import torch
from collections import OrderedDict
import argparse
import logging
import time
from pathlib import Path
import onnx

logger = logging.getLogger('export')
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def file_size(path):
    mb = 1 << 20
    path = Path(path)
    if path.is_file():
        return path.stat().st_size / mb
    elif path.is_dir():
        return sum(f.stat().st_size for f in path.glob('**/*') if f.is_file()) / mb
    else:
        return 0.0

def export_onnx(args):
    t = time.time()
    state_dict = torch.load(args.restore_ckpt)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    
    model = FFLONet(args)
    model.load_state_dict(new_state_dict, strict=True)
    model.to(args.device)
    model.eval()
    model.freeze_bn()
    model.freeze_bn3d()
    logger.info(f"starting from {args.restore_ckpt} ({file_size(args.restore_ckpt):.1f} MB)")
    # 生成随机输入数据
    image1 = torch.randn(1, 3, 720, 1280).float().to(args.device)
    image2 = torch.randn(1, 3, 720, 1280).float().to(args.device)
    padder = InputPadder(image1.shape, divis_by=32)
    image1, image2 = padder.pad(image1, image2)
    logger.info(f'ONNX: starting export with onnx {onnx.__version__}...')
    # 导出模型
    with torch.no_grad():
        torch.onnx.export(model       = model,
                        args          = (image1, image2),
                        f             = args.onnx,
                        export_params = True, 
                        input_names   = ['leftImage', 'rightImage'],    # Input names used in the ONNX model
                        output_names  = ['disparity'],  # Output names used in the ONNX model        
                        opset_version = 18, # >=16
                        # dynamo = True,
                        do_constant_folding = False,                   
                        dynamic_axes  = {'leftImage' : {2 : 'height', 3: 'width'}, # 0 : 'batch', 
                                         'rightImage': {2 : 'height', 3: 'width'}, # 0 : 'batch', 
                                         'disparity' : {2 : 'height', 3: 'width'}  # 0 : 'batch',
                                         } 
                        )

    logger.info(f'Export complete ({time.time() - t:.1f}s)')
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="load the weights from a specific checkpoint", default='./pretrained_models/sceneflow/FFLONet.pth')
    parser.add_argument('--mixed_precision', default=False, help='use mixed precision')
    parser.add_argument('--max_disp', type=int, default=192, help="max disp of geometry encoding volume")
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during validation forward pass')
    parser.add_argument('--onnx', default='./onnx_model/FFLONetDynamic.onnx', help='export onnx model name')
    parser.add_argument('--device', default='cuda:0', help='cuda device, i.e. 0 or 0,1,2,3')
    args = parser.parse_args()

    Path(args.onnx).parent.mkdir(exist_ok=True, parents=True)
    export_onnx(args)
