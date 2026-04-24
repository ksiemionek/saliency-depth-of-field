# Eye tracking saliency

## Setup

1. Clone TranSalNet repository

```bash
   git clone https://github.com/LJOVO/TranSalNet.git
```

2. Download pretrained models [TranSalNet repository](https://github.com/LJOVO/TranSalNet) and place them in `TranSalNet/pretrained_models/`:

- `TranSalNet_Dense.pth`
- `TranSalNet_Res.pth`

3. In `TranSalNet/utils/` add

```
from torch.hub import load_state_dict_from_url
```

in `densenet.py` and `resnet.py` files.

## Result - Dense

![image](./result/saliency_dense.png)

## Result - Res

![image](./result/saliency_res.png)
