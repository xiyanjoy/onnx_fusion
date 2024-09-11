import os
import subprocess
import argparse
import numpy as np
import onnx
import torch
from model import Controlnet_pipeline, download_image
import controlnet_aux
import onnx_graphsurgeon as gs
from polygraphy.backend.onnx.loader import fold_constants, FoldConstants
from onnxsim import simplify
from onnx_fuse_tools import utils

def torch_snr_error(y_pred: torch.Tensor, y_real: torch.Tensor, reduction: str='mean') -> torch.Tensor:
    """
    Compute SNR between y_pred(tensor) and y_real(tensor)
    
    SNR can be calcualted as following equation:
    
        SNR(pred, real) = (pred - real) ^ 2 / (real) ^ 2
    
    if x and y are matrixs, SNR error over matrix should be the mean value of SNR error over all elements.
    
        SNR(pred, real) = mean((pred - real) ^ 2 / (real) ^ 2)

    Args:
        y_pred (torch.Tensor): _description_
        y_real (torch.Tensor): _description_
        reduction (str, optional): _description_. Defaults to 'mean'.

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        torch.Tensor: _description_
    """
    if y_pred.shape != y_real.shape:
        raise ValueError(f'Can not compute snr loss for tensors with different shape. '
            f'({y_pred.shape} and {y_real.shape})')
    reduction = str(reduction).lower()

    if y_pred.ndim == 1:
        y_pred = y_pred.unsqueeze(0)
        y_real = y_real.unsqueeze(0)

    y_pred = y_pred.flatten(start_dim=1)
    y_real = y_real.flatten(start_dim=1)

    noise_power  = torch.pow(y_pred - y_real, 2).sum(dim=-1)
    signal_power = torch.pow(y_real, 2).sum(dim=-1)
    snr = (noise_power) / (signal_power + 1e-7)

    if reduction == 'mean':
        return torch.mean(snr)
    elif reduction == 'sum':
        return torch.sum(snr)
    elif reduction == 'none':
        return snr
    else:
        raise ValueError(f'Unsupported reduction method.')

def shape2str(shape):
    assert len(shape) > 0, "shape is invalid"
    res = ""
    for i in range(len(shape)):
        res += str(shape[i])
        if i < len(shape) - 1:
            res += "_"

    return res

torch.manual_seed(777)
torch.cuda.manual_seed_all(777)

# init
parser = argparse.ArgumentParser()
parser.add_argument('--pplnn-path', help = "accept pplnn's path")
args = parser.parse_args()

height = 512
width = 512
unet_channels = 4
guidance_scale = 7.5
vae_scaling_factor = 0.18215
controlnet_scales = 1.0
device_id = 0
device = f"cuda:{device_id}"
dtype = "float16"
seed = 777

# inference
batch_size = 1
num_warmup_runs = 4
use_cuda_graph = False
denoising_steps = 20
prompt = ["a beautiful photograph of Mt. Fuji during cherry blossom"]
negative_prompt = [""]
depth_image = download_image("https://huggingface.co/lllyasviel/sd-controlnet-depth/resolve/main/images/stormtrooper.png")
print("[I] Finish downloading image ", flush=True)


demo_true = Controlnet_pipeline(
    height,
    width,
    unet_channels,
    guidance_scale,
    vae_scaling_factor,
    device_id,
    device,
    torch.float32,
    seed
)

demo = Controlnet_pipeline(
    height,
    width,
    unet_channels,
    guidance_scale,
    vae_scaling_factor,
    device_id,
    device,
    utils.string_to_torch_dtype_dict[dtype],
    seed
)

input_images = []
depth_image = controlnet_aux.LeresDetector.from_pretrained("lllyasviel/Annotators")(depth_image)
input_images.append(depth_image.resize((height, width)))
input_images = [(np.array(i.convert("RGB")).astype(np.float32) / 255.0)[..., None].transpose(3, 2, 0, 1).repeat(len(prompt), axis=0) for i in input_images]
input_images = [torch.cat( [torch.from_numpy(i).to(device).float()] * (2 if demo.do_classifier_free_guidance else 1) ) for i in input_images]
input_images = torch.cat([image[None, ...] for image in input_images], dim=0)[0]

script_path = os.path.abspath(__file__)
working_dir = os.path.dirname(script_path)
store_path = os.path.join(working_dir, "output")

# demo.infer_torch(prompt, negative_prompt, input_images, denoising_steps, torch.FloatTensor([controlnet_scales]).to(demo.device), store_path)

model_name = "unet"
pplnn_path = args.pplnn_path
print("[I] pplnn_path: {}".format(pplnn_path))
input0_path = os.path.join(store_path, "output")
model_dir = os.path.join(store_path, model_name)
input0_path = os.path.join(model_dir, "input0")
input1_path = os.path.join(model_dir, "input1")
input2_path = os.path.join(model_dir, "input2")
shape0 = (2, 4, 64, 64)
shape1 = (2,)
shape2 = (2, 77, 768)
shape0_str = shape2str(shape0)
shape1_str = shape2str(shape1)
shape2_str = shape2str(shape2)
output_path = os.path.join(model_dir, "pplnn_output-output.dat")
model_true = demo_true.basic_models['unet']         # unet
model = demo.basic_models['unet']         # unet

print(f"************{model_name}*************")
onnx_path = os.path.join(model_dir, model_name + ".onnx")
onnx_opt_path = os.path.join(model_dir, model_name + "_opt.onnx")
pmx_opt_path = os.path.join(model_dir, model_name + "_opt.pmx")
algo_path = os.path.join(model_dir, model_name + "_splitk.json")
algo_path = os.path.join(model_dir, model_name + "_cutlass_conv.json")
if not os.path.exists(store_path):
    os.makedirs(store_path)
if not os.path.exists(model_dir): 
    os.makedirs(model_dir)
input0 = torch.randn(shape0, dtype=torch.float32, device=device)
input1 = torch.randn(shape1, dtype=torch.float32, device=device)
input2 = torch.randn(shape2, dtype=torch.float32, device=device)
np.array(input0.to(utils.string_to_torch_dtype_dict[dtype]).cpu()).tofile(input0_path)
np.array(input1.to(utils.string_to_torch_dtype_dict[dtype]).cpu()).tofile(input1_path)
np.array(input2.to(utils.string_to_torch_dtype_dict[dtype]).cpu()).tofile(input2_path)
torch.onnx.export(
    model,
    (
        input0.to(utils.string_to_torch_dtype_dict[dtype]), 
        input1.to(utils.string_to_torch_dtype_dict[dtype]),
        input2.to(utils.string_to_torch_dtype_dict[dtype])
    ),
    onnx_path,
    opset_version=14,
    do_constant_folding=True,
    input_names=["input0", "input1", "input2"],
    output_names=["output"],
    dynamic_axes={
        'input0': {0: '2B', 2: 'H', 3: 'W'}, 
        'input1': {0: '2B'}, 
        'input2': {0: '2B', 1: 'L'}
    }
)
onnxmodel = onnx.load(onnx_path)
model_simp, check = simplify(onnxmodel)
assert check, "Simplified ONNX model could not be validated"
onnxmodel = FoldConstants(onnxmodel, do_shape_inference=True, size_threshold=512 << 29, allow_onnxruntime_shape_inference=True)()
graph = gs.import_onnx(onnxmodel)
utils.merge_gegelu(graph)
utils.merge_ReshapeSqueeze(graph)
utils.merge_ReshapeUnsqueeze(graph)
utils.merge_LayerNorm(graph)
utils.merge_SiLU(graph)
utils.merge_GroupNorm(graph, 6)
utils.merge_GroupNormMul(graph)
utils.merge_AddGroupNormMul(graph)
utils.merge_SelfAttention(graph, 16)
utils.merge_ContextAttention(graph, 16)

utils.erase_div_1(graph)
onnx.save(utils.onnx_pmx(gs.export_onnx(graph)), onnx_opt_path, save_as_external_data=True, all_tensors_to_one_file=True)
print(f'save model to {onnx_opt_path}', flush=True)

subprocess.run([
    pplnn_path,
    "--use-cuda",
    "--onnx-model", onnx_opt_path, 
    "--inputs", input0_path + "," + input1_path + "," + input2_path, 
    "--in-shapes", shape0_str + "," + shape1_str + "," + shape2_str,
    "--save-outputs",
    "--save-data-dir", model_dir,
    # "--export-pmx-model", pmx_opt_path,
    # "--pmx-model", pmx_opt_path,
    # "--quick-select",
    "--export-algo-file", algo_path,
    # "--import-algo-file", algo_path,
    "--enable-profiling",
    "--min-profiling-iterations", "500",
    "--kernel-type", dtype
], check=True)
output_true = model_true(input0, input1, input2)["sample"]
output_ppl = torch.tensor(np.fromfile(output_path, dtype=utils.string_to_numpy_dtype_dict[dtype])).reshape(shape0).to(device).float()
print("torch_snr_error: {}".format(torch_snr_error(output_ppl, output_true)))
# output_true.cpu().detach().numpy().tofile(os.path.join(model_dir, "output_true"))
# output_ppl.cpu().detach().numpy().tofile(os.path.join(model_dir, "output_ppl"))
# subprocess.run([
#     "/mnt/hpc/share/xiexiaotong/fp_diff/fp_diff",
#     os.path.join(model_dir, "output_true"),
#     os.path.join(model_dir, "output_ppl")
# ], check=True)