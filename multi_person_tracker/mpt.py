import os
import cv2
import time
import torch
import shutil
import numpy as np
import os.path as osp
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from yolov3.yolo import YOLOv3

from multi_person_tracker import Sort
from multi_person_tracker.data import ImageFolder, images_to_video

class MPT():
    def __init__(
            self,
            device=None,
            batch_size=12,
            display=False,
            detection_threshold=0.7,
            detector_type='yolo',
            yolo_img_size=608,
            output_format='list',
    ):
        '''
        Multi Person Tracker

        :param device (str, 'cuda' or 'cpu'): torch device for model and inputs
        :param batch_size (int): batch size for detection model
        :param display (bool): display the results of multi person tracking
        :param detection_threshold (float): threshold to filter detector predictions
        :param detector_type (str, 'maskrcnn' or 'yolo'): detector architecture
        :param yolo_img_size (int): yolo detector input image size
        :param output_format (str, 'dict' or 'list'): result output format
        '''

        if device is not None:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.batch_size = batch_size
        self.display = display
        self.detection_threshold = detection_threshold
        self.output_format = output_format

        if detector_type == 'maskrcnn':
            self.detector = keypointrcnn_resnet50_fpn(pretrained=True).to(self.device).eval()
        elif detector_type == 'yolo':
            self.detector = YOLOv3(
                device=self.device, img_size=yolo_img_size, person_detector=True, video=True, return_dict=True
            )
        else:
            raise ModuleNotFoundError

        self.tracker = Sort()

    @torch.no_grad()
    def run_tracker(self, dataloader):
        '''
        Run tracker on an input video

        :param video (ndarray): input video tensor of shape NxHxWxC. Preferable use skvideo to read videos
        :return: trackers (ndarray): output tracklets of shape Nx5 [x1,y1,x2,y2,track_id]
        '''

        # initialize tracker
        self.tracker = Sort()

        start = time.time()
        print('Running Multi-Person-Tracker')
        trackers = []
        for batch in tqdm(dataloader):
            batch = batch.to(self.device)

            predictions = self.detector(batch)

            for pred in predictions:
                bb = pred['boxes'].cpu().numpy()
                sc = pred['scores'].cpu().numpy()[..., None]
                dets = np.hstack([bb,sc])
                dets = dets[sc[:,0] > self.detection_threshold]

                # if nothing detected do not update the tracker
                if dets.shape[0] > 0:
                    track_bbs_ids = self.tracker.update(dets)
                else:
                    track_bbs_ids = np.empty((0, 5))
                trackers.append(track_bbs_ids)

        runtime = time.time() - start
        fps = len(dataloader.dataset) / runtime
        print(f'Finished. Detection + Tracking FPS {fps:.2f}')
        return trackers

    @torch.no_grad()
    def run_detector(self, dataloader):
        '''
        Run tracker on an input video

        :param video (ndarray): input video tensor of shape NxHxWxC. Preferable use skvideo to read videos
        :return: trackers (ndarray): output tracklets of shape Nx5 [x1,y1,x2,y2,track_id]
        '''

        start = time.time()
        print('Running Multi-Person-Tracker')
        detections = []
        for batch in tqdm(dataloader):
            batch = batch.to(self.device)

            predictions = self.detector(batch)

            for pred in predictions:
                bb = pred['boxes'].cpu().numpy()
                sc = pred['scores'].cpu().numpy()[..., None]
                dets = np.hstack([bb, sc])
                dets = dets[sc[:, 0] > self.detection_threshold]

                detections.append(dets)

        runtime = time.time() - start
        fps = len(dataloader.dataset) / runtime
        print(f'Finished. Detection + Tracking FPS {fps:.2f}')
        return detections

    def prepare_output_detections(self, detections):
        new_detections = []
        for frame_idx, dets in enumerate(detections):
            img_dets = []
            for d in dets:
                w, h = d[2] - d[0], d[3] - d[1]
                c_x, c_y = d[0] + w / 2, d[1] + h / 2
                w = h = np.where(w / h > 1, w, h)
                bbox = np.array([c_x, c_y, w, h])
                img_dets.append(bbox)
            new_detections.append(np.array(img_dets))
        return new_detections

    def prepare_output_tracks(self, trackers):
        '''
        Put results into a dictionary consists of detected people
        :param trackers (ndarray): input tracklets of shape Nx5 [x1,y1,x2,y2,track_id]
        :return: dict: of people. each key represent single person with detected bboxes and frame_ids
        '''
        people = dict()

        for frame_idx, tracks in enumerate(trackers):
            for d in tracks:
                person_id = int(d[4])
                # bbox = np.array([d[0], d[1], d[2] - d[0], d[3] - d[1]]) # x1, y1, w, h

                w, h = d[2] - d[0], d[3] - d[1]
                c_x, c_y = d[0] + w/2, d[1] + h/2
                w = h = np.where(w / h > 1, w, h)
                bbox = np.array([c_x, c_y, w, h])

                if person_id in people.keys():
                    people[person_id]['bbox'].append(bbox)
                    people[person_id]['frames'].append(frame_idx)
                else:
                    people[person_id] = {
                        'bbox' : [],
                        'frames' : [],
                    }
                    people[person_id]['bbox'].append(bbox)
                    people[person_id]['frames'].append(frame_idx)
        for k in people.keys():
            people[k]['bbox'] = np.array(people[k]['bbox']).reshape((len(people[k]['bbox']), 4))
            people[k]['frames'] = np.array(people[k]['frames'])

        return people

    def display_results(self, image_folder, trackers, output_file=None):
        '''
        Display the output of multi-person-tracking
        :param video (ndarray): input video tensor of shape NxHxWxC
        :param trackers (ndarray): tracklets of shape Nx5 [x1,y1,x2,y2,track_id]
        :return: None
        '''
        print('Displaying results..')

        save = True if output_file else False
        tmp_write_folder = osp.join('/tmp', f'{osp.basename(image_folder)}_mpt_results')
        os.makedirs(tmp_write_folder, exist_ok=True)

        colours = np.random.rand(32, 3)
        image_file_names = sorted([
            osp.join(image_folder, x)
            for x in os.listdir(image_folder)
            if x.endswith('.png') or x.endswith('.jpg')
        ])

        for idx, (img_fname, tracker) in enumerate(zip(image_file_names, trackers)):

            img = cv2.imread(img_fname)
            for d in tracker:
                d = d.astype(np.int32)
                c = (colours[d[4] % 32, :] * 255).astype(np.uint8).tolist()
                cv2.rectangle(
                    img, (d[0], d[1]), (d[2], d[3]),
                    color=c, thickness=int(round(img.shape[0] / 256))
                )
                cv2.putText(img, f'{d[4]}', (d[0] - 9, d[1] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                cv2.putText(img, f'{d[4]}', (d[0] - 8, d[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

            cv2.imshow('result video', img)

            # time.sleep(0.03)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if save:
                cv2.imwrite(osp.join(tmp_write_folder, f'{idx:06d}.png'), img)

        cv2.destroyAllWindows()

        if save:
            print(f'Saving output video to {output_file}')
            images_to_video(img_folder=tmp_write_folder, output_vid_file=output_file)
            shutil.rmtree(tmp_write_folder)

    def display_detection_results(self, image_folder, detections, output_file=None):
        '''
                Display the output of detector
                :param video (ndarray): input video tensor of shape NxHxWxC
                :param detections (ndarray): detections of shape Nx4 [x1,y1,x2,y2,track_id]
                :return: None
                '''
        print('Displaying results..')

        save = True if output_file else False
        tmp_write_folder = osp.join('/tmp', f'{osp.basename(image_folder)}_mpt_results')
        os.makedirs(tmp_write_folder, exist_ok=True)

        colours = np.random.rand(32, 3)
        image_file_names = sorted([
            osp.join(image_folder, x)
            for x in os.listdir(image_folder)
            if x.endswith('.png') or x.endswith('.jpg')
        ])

        for idx, (img_fname, dets) in enumerate(zip(image_file_names, detections)):
            print(img_fname)
            img = cv2.imread(img_fname)
            for d in dets:
                d = d.astype(np.int32)
                c = (0, 255, 0)
                cv2.rectangle(
                    img, (d[0], d[1]), (d[2], d[3]),
                    color=c, thickness=int(round(img.shape[0] / 256))
                )

            cv2.imshow('result image', img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if save:
                cv2.imwrite(osp.join(tmp_write_folder, f'{idx:06d}.png'), img)

        cv2.destroyAllWindows()

        if save:
            print(f'Saving output video to {output_file}')
            images_to_video(img_folder=tmp_write_folder, output_vid_file=output_file)
            shutil.rmtree(tmp_write_folder)

    def __call__(self, image_folder, output_file=None):
        '''
        Execute MPT and return results as a dictionary of person instances

        :param video (ndarray): input video tensor of shape NxHxWxC
        :return: a dictionary of person instances
        '''

        image_dataset = ImageFolder(image_folder)

        dataloader = DataLoader(image_dataset, batch_size=self.batch_size, num_workers=0)

        trackers = self.run_tracker(dataloader)
        if self.display:
            self.display_results(image_folder, trackers, output_file)

        if self.output_format == 'dict':
            result = self.prepare_output_tracks(trackers)
        elif self.output_format == 'list':
            result = trackers

        return result

    def detect(self, image_folder, output_file=None):
        image_dataset = ImageFolder(image_folder)

        dataloader = DataLoader(image_dataset, batch_size=self.batch_size, num_workers=0)

        detections = self.run_detector(dataloader)
        if self.display:
            self.display_detection_results(image_folder, detections, output_file)
        detections = self.prepare_output_detections(detections)
        return detections
