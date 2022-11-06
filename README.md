### Install

```
pip install transformers mmcv-full
git clone https://github.com/open-mmlab/mmaction2.git
cd mmaction2
pip install -r requirements/build.txt
pip install -v -e .
```
Then change `mmaction/models/backbones/resnet3d_csn.py` line 116 `temporal_strides=(1, 2, 2, 2)` to `temporal_strides=(1, 1, 1, 1)`.
