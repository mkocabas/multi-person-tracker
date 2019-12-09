# Simple Multi Person Tracker
Simple and easy to use multi person tracker implementation. This project supports YOLOV3 & MaskRCNN as detector and
SORT as object tracker.

<p float="center">
  <img src="https://s4.gifyu.com/images/out04257fef98d27cff.gif" width="49%" />
  <img src="https://s4.gifyu.com/images/out0da89d9409694658.gif" width="49%" />
</p>

## Installation
First you need to install the requirements:

    $ pip install -r requirements.txt

Then install the package via:

    $ pip install git+https://github.com/mkocabas/multi-person-tracker.git
    
## Usage
Run [`examples/demo_video.py`](examples/demo_video.py) for a minimally working example. Here is a sample:
```python
from multi_person_tracker import MPT
from multi_person_tracker.data import video_to_images

image_folder = video_to_images('sample_video.mp4')

mpt = MPT(
    display=True,
    detector_type='yolo', # or 'maskrcnn'
    batch_size=10,
    yolo_img_size=416,
)

result = mpt(image_folder, output_file='sample_output.mp4')
``` 
## Runtime Performance

| Detector    | Tracker                | GPU      | FPS      |
| ------------|:---------:|:--------:|:--------:|
| MaskRCNN    | Sort | RTX2080Ti | 13        |
| YOLOv3-256  | Sort | RTX2080Ti | 120       |
| YOLOv3-416  | Sort | RTX2080Ti | 80        |
| YOLOv3-608  | Sort | RTX2080Ti | 46        |

## Important Note
- Install [torchvision](https://github.com/pytorch/vision) from the source as done in 
[`requirements.txt`](requirements.txt) to be able to use the best performing MaskRCNN pretrained model. 
Check this [Issue](https://github.com/pytorch/vision/issues/1606) 
and [PR](https://github.com/pytorch/vision/pull/1609) for more details.

## References
- YOLOV3: [paper](https://pjreddie.com/media/files/papers/YOLOv3.pdf)
- MaskRCNN: [paper](https://arxiv.org/abs/1703.06870)
- SORT: [paper](https://arxiv.org/abs/1602.00763)