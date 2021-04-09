
# Efficient Models
| Model     |   Parameters  |   GMacs   |   GFLOPS  |   F-Pass Memory   |  Loss  |   IoU   |  Epoch |  Optimizer |  Criterion  | LR | Dataset |   Paper | Code |   Type  |
|-----------|---------------|-----------|-----------|-------------------|--------|---------|--------|------------|-------------|----|---------|---------|------|---------|
| HarDNet-DWS-39   |      2.2M         |     0.44      |     0.88      |        29.77MB           |  0 |   0 |  0  | N | N | N | 38-Cloud | [Link](https://drive.google.com/file/d/1_QFqasN4UEIzv5ku5JIzSHXH5JFrIkzF/view?usp=sharing) | [Code](https://github.com/PingoLH/Pytorch-HarDNet) | Classification |
| FCHarDNet |      453K         |     0.85      |     1.7      |        **22.04MB**           |  0.106 |   0 |  289  | Adam | DiceLoss | 0.1e-4 | 38-Cloud | [Link](https://drive.google.com/file/d/1_QFqasN4UEIzv5ku5JIzSHXH5JFrIkzF/view?usp=sharing) | [Code](https://github.com/PingoLH/FCHarDNet) | Segmentation |
| FCHarDNet |      453K         |     0.85      |     1.7      |        **22.04MB**           |  0.049 |   0 |  943  | Adam | DiceLoss | 0.1e-4 | 38-Cloud | [Link](https://drive.google.com/file/d/1_QFqasN4UEIzv5ku5JIzSHXH5JFrIkzF/view?usp=sharing) | [Code](https://github.com/PingoLH/FCHarDNet) | Segmentation |
| **ENet** |      **349K**         |     **0.4**     |     **0.8**      |        145MB           |  **0.038** |   0 |  982  | Adam | BCELoss | .1e-4 | 38-Cloud | [Link](https://drive.google.com/file/d/1v53ZwNO4281KSaDiBJIlkSCft5Zz480-/view?usp=sharing) | [Code](https://github.com/davidtvs/PyTorch-ENet) | Segmentation |
| SegNet |      24.4M         |     30.73      |     64.4     |        343.77MB           |  -0.250 |   0 |  281  | Adam | BCELoss | .1e-4 | 38-Cloud | [Link](https://drive.google.com/file/d/1jXhyW80IMargCt8IrNTsf9eAenrUcQvX/view?usp=sharing) | [Code](https://github.com/kwakuTM/SegNet) | Segmentation |
| SegNet |      24.4M         |     30.73      |     64.4     |        343.77MB           |  0.532 |   0 |  296  | SGD | BCELoss | .1e-4 | 38-Cloud | [Link](https://drive.google.com/file/d/1jXhyW80IMargCt8IrNTsf9eAenrUcQvX/view?usp=sharing) | [Code](https://github.com/kwakuTM/SegNet) | Segmentation |
| UNet |      31M         |     41.9      |     83.8     |        419.95MB           |  0.054 |   0 |  293  | Adam | DiceLoss | .1e-4 | 38-Cloud | [Link](https://drive.google.com/file/d/1vtO-1-Vnbg-kh56zGt0wPSNV0fzdyDfu/view?usp=sharing) | [Code](https://github.com/milesial/Pytorch-UNet) | Segmentation |
| FineTuned-FCHarDNet-Cloud38-RSI |      453K         |     0.85      |     1.7      |        **22.04MB**           |  **0.142** |   0 |  290  | Adam | DiceLoss | 0.1e-4 | 38-Cloud - RSI| [Link](https://drive.google.com/file/d/1_QFqasN4UEIzv5ku5JIzSHXH5JFrIkzF/view?usp=sharing) | [Code](https://github.com/PingoLH/FCHarDNet) | Segmentation |
| **ENet** |      **349K**         |     **0.4**     |     **0.8**      |        145MB           |  0.22 |   0 |  286  | Adam | BCELoss | .1e-4 | 38-Cloud | [Link](https://drive.google.com/file/d/1v53ZwNO4281KSaDiBJIlkSCft5Zz480-/view?usp=sharing) | [Code](https://github.com/davidtvs/PyTorch-ENet) | Segmentation |





# Top Models Trained in CamVid Dataset
|    Model  |    mIoU   |   Sky  |   Road   |   Building   |    Sidewalk   |    Car    |    Tree    |   Fence   |   Cyclist   |   Pedestrian   |   Pole   |  Sign  |
|-----------|-----------|--------|----------|--------------|---------------|-----------|------------|-----------|-------------|----------------|----------|--------|
| FCHarDNet |   0.506	|  51.95 |   50.99  |    40.20     |      39.32    |   37.69   |    35.33   |    19.1   |     16.18   |      13.35     |  10.02   |  8.45  |  
| ENet      |   0.566	|  52.91 |   52.59  |    45.32     |      45.02    |   40.52   |    42.20   |    23.94  |     20.34   |      19.70     |  18.58   |  0  |  