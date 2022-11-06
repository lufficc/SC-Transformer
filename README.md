### Install

```
pip install transformers mmcv-full
git clone https://github.com/open-mmlab/mmaction2.git
cd mmaction2
pip install -r requirements/build.txt
pip install -v -e .
```
Then change `mmaction/models/backbones/resnet3d_csn.py` line 116 `temporal_strides=(1, 2, 2, 2)` to `temporal_strides=(1, 1, 1, 1)`.

### Run

```
python inference.py --video <path_to_video>
```

Sample Output:
```
Video path: 7024397291742055712_v0300fg10000c5tqm1bc77u1cnr5grog.mp4
Sampled frames: 577
Video FPS: 30
Video Duration: 57.7s
Number Boundaries: 50
[0.6, 1.5, 2.6, 3.5, 4.7, 5.7, 6.6, 7.6, 8.9, 10.1, 11.2, 12.6, 13.8, 15.3, 17.4, 20.0, 20.7, 21.9, 23.8, 24.5, 25.6, 26.4, 27.3, 27.9, 29.2, 31.0, 32.0, 33.0, 34.5, 35.6, 37.0, 38.1, 39.4, 40.1, 41.0, 41.6, 42.6, 44.1, 45.2, 46.4, 47.3, 48.0, 48.9, 50.0, 51.6, 52.4, 53.5, 54.4, 55.3, 56.7]
```
