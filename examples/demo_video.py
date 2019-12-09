import sys
sys.path.append('.')
import shutil
from multi_person_tracker import MPT
from multi_person_tracker.data import video_to_images

def main():
    vf = sys.argv[1]

    image_folder = video_to_images(vf)
    mot = MPT(
        display=True,
        detector_type='yolo',  # 'maskrcnn'
        batch_size=10,
        detection_threshold=0.7,
        yolo_img_size=416,
    )

    result = mot(image_folder, output_file='sample.mp4')

    shutil.rmtree(image_folder)


if __name__ == '__main__':
    main()

