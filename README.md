## Transferring Multi-Modal Domain Knowledge to Uni-Modal Domain for Urban Scene Segmentation

**[[Paper]](https://ieeexplore.ieee.org/abstract/document/10496388)**


## Setup Environment

For this project, we used python 3.8.

```shell
conda create -n mdkt python=3.8
```

In that environment, the requirements can be installed with:

```shell
conda activate mdkt
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.3.7  # requires the other packages to be installed first
```

## Setup Datasets 

### RGB data

**Cityscapes:** Download leftImg8bit_trainvaltest.zip and
gt_trainvaltest.zip from [here](https://www.cityscapes-dataset.com/downloads/)
and extract them to `data/cityscapes`.

**GTA:** Download all image and label packages from
[here](https://download.visinf.tu-darmstadt.de/data/from_games/) and extract
them to `data/gta`.

**Synthia:** Download SYNTHIA-RAND-CITYSCAPES from
[here](http://synthia-dataset.net/downloads/) and extract it to `data/synthia`.

**Data Preprocessing:** Run the following scripts to convert the label IDs to the
train IDs and to generate the class index for RCS:

```shell
python tools/convert_datasets/gta.py data/gta --nproc 8
python tools/convert_datasets/cityscapes.py data/cityscapes --nproc 8
python tools/convert_datasets/synthia.py data/synthia/ --nproc 8
```

### Depth data

The depth data of `cityscapes` and `gta` can be obtained in [CorDA](https://github.com/qinenergy/corda). Please put the depth information into the corresponding folder.
The final folder structure should look like this:

```none
mdkt
├── ...
├── data
│   ├── cityscapes
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   ├── val
│   ├── gta
│   │   ├── images
│   │   ├── labels
│   │   ├── depth
│   ├── synthia
│   │   ├── RGB
│   │   ├── GT
│   │   │   ├── LABELS
│   │   ├── Depth
├── ...
```


## Training
Download the MiT weights provided by [SegFormer](https://github.com/NVlabs/SegFormer?tab=readme-ov-file#training) and put them in the folder `pretrained/`.

A training job can be launched using:

```shell
python run_experiments.py --config configs/mdkt/mdkt.py
```

## Testing & Predictions

```shell
sh test.sh path/to/checkpoint_directory
```

## Citation

If you find this project useful in your research, please consider citing:

```

@ARTICLE{peng2024mdkt,
  author={Liu, Peng and Ge, Yanqi and Duan, Lixin and Li, Wen and Luo, Haonan and Lv, Fengmao},
  journal={IEEE Transactions on Intelligent Transportation Systems}, 
  title={Transferring Multi-Modal Domain Knowledge to Uni-Modal Domain for Urban Scene Segmentation}, 
  year={2024},
  volume={},
  number={},
  pages={1-14},
  keywords={Training;Semantic segmentation;Transformers;Task analysis;Adaptation models;Visualization;Synthetic data;Urban scene understanding;domain adaptation;semantic segmentation;multi-modal learning},
  doi={10.1109/TITS.2024.3382880}
  }
```


## Acknowledgements

This project is based on the [DAFormer](https://github.com/lhoyer/DAFormer). We thank the
authors for making the source code publicly available.
