from enum import Enum
from io import BytesIO
from typing import Dict

import numpy as np

# MiDaS
import torch
from torch import device
import cv2

from torchvision.transforms import Compose

from midas.dpt_depth import DPTDepthModel
from midas.midas_net import MidasNet
from midas.midas_net_custom import MidasNet_small
from midas.transforms import Resize, NormalizeImage, PrepareForNet


class HTTPMethods(Enum):
    post = "POST"
    get = "GET"
    put = "PUT"
    delete = "DELETE"


def generate_midas_depth_map(input_buf: BytesIO, model_type, optimize=True) -> BytesIO:
    print("initialize")

    default_models: Dict[str, str] = {
        "midas_v21_small": "weights/midas_v21_small-70d6b9c8.pt",
        "midas_v21": "weights/midas_v21-f6b98070.pt",
        "dpt_large": "weights/dpt_large-midas-2f21e586.pt",
        "dpt_hybrid": "weights/dpt_hybrid-midas-501f0c75.pt",
    }

    model_path: str = default_models[model_type]

    # select device
    torch_device: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {torch_device}")

    # load network
    if model_type == "dpt_large":  # DPT-Large
        model: DPTDepthModel = DPTDepthModel(
            path=model_path,
            backbone="vitl16_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "dpt_hybrid":  # DPT-Hybrid
        model: DPTDepthModel = DPTDepthModel(
            path=model_path,
            backbone="vitb_rn50_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "midas_v21":
        model: MidasNet = MidasNet(model_path, non_negative=True)
        net_w, net_h = 384, 384
        resize_mode = "upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    elif model_type == "midas_v21_small":
        model: MidasNet_small = MidasNet_small(model_path, features=64, backbone="efficientnet_lite3", exportable=True,
                                               non_negative=True, blocks={'expand': True})
        net_w, net_h = 256, 256
        resize_mode: str = "upper_bound"
        normalization: NormalizeImage = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    else:
        print(f"model_type '{model_type}' not implemented, use: --model_type large")
        assert False

    transform: Compose = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method=resize_mode,
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    model.eval()

    if optimize:
        # rand_example = torch.rand(1, 3, net_h, net_w)
        # model(rand_example)
        # traced_script_module = torch.jit.trace(model, rand_example)
        # model = traced_script_module

        if torch_device == torch.device("cuda"):
            model = model.to(memory_format=torch.channels_last)
            model = model.half()

    model.to(torch_device)

    img = cv2.imdecode(np.frombuffer(input_buf.read(), np.uint8), 1)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    img_input = transform({"image": img})["image"]

    # compute
    with torch.no_grad():
        sample = torch.from_numpy(img_input).to(torch_device).unsqueeze(0)
        if optimize and torch_device == torch.device("cuda"):
            sample = sample.to(memory_format=torch.channels_last)
            sample = sample.half()
        prediction = model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

    return write_depth(prediction, bits=2)


def write_depth(depth, bits=1) -> BytesIO:
    depth_min = depth.min()
    depth_max = depth.max()

    max_val = (2 ** (8 * bits)) - 1

    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = np.zeros(depth.shape, dtype=depth.type)

    # newOut = np.zeros((depth.shape[0], depth.shape[1], 4))
    # for heightY in range(out.shape[0]):
    #     for widthX in range(out.shape[1]):
    #         greyPix = out[heightY][widthX]
    #         newOut[heightY][widthX] = [greyPix, greyPix, greyPix, 255]

    dtypes = {
        1: "uint8",
        2: "uint16"
    }

    is_success, buffer = cv2.imencode(".png", out.astype(dtypes[bits]))

    return BytesIO(buffer)
