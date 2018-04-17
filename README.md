# MUNIT-Tensorflow
Simple Tensorflow implementation of ["Multimodal Unsupervised Image-to-Image Translation"](https://arxiv.org/abs/1804.04732)

## Requirements
* Tensorflow 1.4
* Python 3.6

## Usage
```bash
├── dataset
   └── YOUR_DATASET_NAME
       ├── trainA
           ├── xxx.jpg (name, format doesn't matter)
           ├── yyy.png
           └── ...
       ├── trainB
           ├── zzz.jpg
           ├── www.png
           └── ...
       ├── testA
           ├── aaa.jpg 
           ├── bbb.png
           └── ...
       └── testB
           ├── ccc.jpg 
           ├── ddd.png
           └── ...
```
### Train
* python main.py --phase train --dataset summer2winter

### Test
* python main.py --phase test --dataset summer2winter --num_style 3 --batch_size 1

### Guided Image Translation
* python main.py --phase guide --dataset summer2winter --batch_size 1 --guide_img ./guide.jpg

## Summary
![illustration](./assests/method_illustration.png)

## Architecture 
![architecture](./assests/architecture.png)

## Model Overview
![model_overview](./assests/model_overview.png)

## Results
### Edges to Shoes/handbags Translation
![edges2shoes_handbags](./assests/edges2shoes_handbags.jpg)

### Animal Image Translation
![animal](./assests/animal.jpg)

### Street Scene Translation
![street](./assests/street.jpg)

### Yosemite Summer to Winter Translation (HD)
![summer2winter_yosemite](./assests/summer2winter_yosemite.jpg)

### Example-guided Image Translation
![guide](./assests/guide.jpg)

## Related works
* [CycleGAN-Tensorflow](https://github.com/taki0112/CycleGAN-Tensorflow)
* [DiscoGAN-Tensorflow](https://github.com/taki0112/DiscoGAN-Tensorflow)
* [UNIT-Tensorflow](https://github.com/taki0112/UNIT-Tensorflow)

## Reference
* [MUNIT-Pytorch](https://github.com/NVlabs/MUNIT) (Author implementation)

## Author
Junho Kim
