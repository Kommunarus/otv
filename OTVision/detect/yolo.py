"""
OTVision module to detect objects using yolov5
"""
import json
# Copyright (C) 2022 OpenTrafficCam Contributors
# <https://github.com/OpenTrafficCam
# <team@opentrafficcam.org>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# TODO: docstrings in yolo

from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from cv2 import CAP_PROP_FPS, VideoCapture
import cv2
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import uuid
import json
import os
# import albumentations as A


from OTVision.config import CONFIG
from OTVision.helpers.files import is_in_format
from OTVision.helpers.log import log


class NoVideoError(Exception):
    pass


class VideoFoundError(Exception):
    pass


def detect_video(
    file_path,
    model=None,
    weights: str = CONFIG["DETECT"]["YOLO"]["WEIGHTS"],
    conf: float = CONFIG["DETECT"]["YOLO"]["CONF"],
    iou: float = CONFIG["DETECT"]["YOLO"]["IOU"],
    size: int = CONFIG["DETECT"]["YOLO"]["IMGSIZE"],
    chunksize: int = CONFIG["DETECT"]["YOLO"]["CHUNKSIZE"],
    normalized: bool = CONFIG["DETECT"]["YOLO"]["NORMALIZED"],
    model_siam=None,
    dir_features='',
):
    """Detect and classify bounding boxes in videos using YOLOv5

    Args:
        files (str ot list of str): files to detect.
        model (yolo object): Yolo model to detect with.
        weights (str, optional): Weigths, if no model passed. Defaults to "yolov5s".
        conf (float, optional): Output confidence, if no model passed. Defaults to 0.25.
        iou (float, optional): IOU param, if no model passed. Defaults to 0.45.
        size (int, optional): Frame size for detection. Defaults to 640.
        chunksize (int, optional): Number of files per detection chunk. Defaults to 0.
        normalized (bool, optional): Coords in % of image/frame size (True) or pixels
        (False). Defaults to False.

    Returns:
        [type]: [description]
    """
    if model is None:
        model = loadmodel(weights, conf, iou)

    yolo_detections = []
    yolo_detections_feature = [] # kav 190922
    files_feature = {} # kav 181022
    t1 = perf_counter()

    if not is_in_format(file_path, CONFIG["FILETYPES"]["VID"]):
        raise NoVideoError(f"The file: {file_path} is not a video!")

    cap = VideoCapture(file_path)
    batch_no = 0
    bb = 200

    log.info(f"Run detection on video: {file_path}")

    got_frame = True
    img_batchs = []
    list_f_f = {}

    while got_frame:
        got_frame, img_batch = _get_batch_of_frames(cap, chunksize)

        if not img_batch:
            break

        t_start = perf_counter()

        # What purpose does this transformation have
        transformed_batch = list(map(lambda frame: frame[:, :, ::-1], img_batch))

        t_trans = perf_counter()

        # if batch_no == 477:
        #     pass
        results = model(transformed_batch, size)

        t_det = perf_counter()

        _add_detection_results(yolo_detections, results, normalized)

        t_list = perf_counter()

        _log_batch_performances_stats(
            batch_no, t_start, t_trans, t_det, t_list, len(img_batch)
        )
        batch_no += 1

        width = cap.get(3)  # float
        height = cap.get(4)  # float
        fps = cap.get(CAP_PROP_FPS)  # float
        frames = cap.get(7)  # float

        _delete_small_detection(yolo_detections, width, height, normalized)  # kav 210922
        img_batchs.append(img_batch)
        if batch_no % bb == 0:
            _add_detection_feature_results(yolo_detections_feature, yolo_detections[-bb:], img_batchs, model_siam,
                                           normalized, width, height, dir_features, files_feature)
            img_batchs = []
            n_f_f = 1
            dict_f_f = {}
            name_file = str(uuid.uuid4())
            for k, v in files_feature.items():
                list_f_f[k] = name_file
                dict_f_f[k] = v
                if n_f_f % 1000 == 0:
                    with open(os.path.join(dir_features, name_file), "w") as f:
                        json.dump(dict_f_f, f)
                    dict_f_f = {}
                    name_file = str(uuid.uuid4())
                n_f_f += 1
            else:
                if len(dict_f_f) != 0:
                    with open(os.path.join(dir_features, name_file), "w") as f:
                        json.dump(dict_f_f, f)

            files_feature = {}
        # if batch_no == 10000:
        #     break

    else:
        _add_detection_feature_results(yolo_detections_feature, yolo_detections[-len(img_batchs):], img_batchs, model_siam,
                                       normalized, width, height, dir_features, files_feature)
        n_f_f = 1
        dict_f_f = {}
        name_file = str(uuid.uuid4())
        for k, v in files_feature.items():
            list_f_f[k] = name_file
            dict_f_f[k] = v
            if n_f_f % 1000 == 0:
                with open(os.path.join(dir_features, name_file), "w") as f:
                    json.dump(dict_f_f, f)
                dict_f_f = {}
                name_file = str(uuid.uuid4())
            n_f_f += 1
        else:
            if len(dict_f_f) != 0:
                with open(os.path.join(dir_features, name_file), "w") as f:
                    json.dump(dict_f_f, f)

        # saveimage(transformed_batch, yolo_detections, width, height, normalized, batch_no)

    with open(os.path.join(dir_features, 'all_f'), "w") as f:
        json.dump(list_f_f, f)

    t2 = perf_counter()
    duration = t2 - t1
    det_fps = len(yolo_detections) / duration
    _log_overall_performance_stats(duration, det_fps)

    # class_names = results.names
    class_names = {0: 'pedestrian', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus',
                   6: 'heavy vehicle', 4: 'medium vehicle'}

    det_config = _get_det_config(weights, conf, iou, size, chunksize, normalized)
    vid_config = _get_vidconfig(file_path, width, height, fps, frames)
    return _convert_detections(yolo_detections, class_names, vid_config, det_config, yolo_detections_feature)


def detect_images(
    file_chunks,
    model=None,
    weights: str = CONFIG["DETECT"]["YOLO"]["WEIGHTS"],
    conf: float = CONFIG["DETECT"]["YOLO"]["CONF"],
    iou: float = CONFIG["DETECT"]["YOLO"]["IOU"],
    size: int = CONFIG["DETECT"]["YOLO"]["IMGSIZE"],
    chunksize: int = CONFIG["DETECT"]["YOLO"]["CHUNKSIZE"],
    normalized: bool = CONFIG["DETECT"]["YOLO"]["NORMALIZED"],
    ot_labels_enabled: bool = False,
):
    """Detect and classify bounding boxes in images/frames using YOLOv5

    Args:
        files (str ot list of str): files to detect.
        model (yolo object): Yolo model to detect with.
        weights (str, optional): Weigths, if no model passed. Defaults to "yolov5s".
        conf (float, optional): Output confidence, if no model passed. Defaults to 0.25.
        iou (float, optional): IOU param, if no model passed. Defaults to 0.45.
        size (int, optional): Frame size for detection. Defaults to 640.
        chunksize (int, optional): Number of files per detection chunk. Defaults to 0.
        normalized (bool, optional): Coords in % of image/frame size (True) or pixels
        (False). Defaults to False.
        ot_labels_enabled (bool, optional): returns [detections, names] where detections
        consist of bounding boxes but without any annotations and the class name index
        (True) or returns the detections in otdet format(False). Defaults to False.

    Returns:
        [type]: [description]
    """
    yolo_detections = []
    if not file_chunks:
        return [], [] if ot_labels_enabled else yolo_detections
    if model is None:
        model = loadmodel(weights, conf, iou)
    t1 = perf_counter()
    if _containsvideo(file_chunks):
        raise VideoFoundError(
            "List of paths given to detect_chunks function shouldn't contain any videos"
        )

    log.info("Run detection on images")
    for img_batch, chunk in enumerate(file_chunks, start=1):
        t_start = perf_counter()
        results = model(chunk, size=size)
        t_det = perf_counter()
        _add_detection_results(yolo_detections, results, normalized)
        str_batch_no = f"img_batch_no: {img_batch:d}"
        str_det_time = f"det:{t_det - t_start:0.4f}"
        str_batch_size = f"batch_size: {len(chunk):d}"
        str_fps = f"fps: {chunksize / (t_det - t_start):0.1f}"
        log.info(f"{str_batch_no}, {str_det_time}, {str_batch_size}, {str_fps}")

    t2 = perf_counter()
    duration = t2 - t1
    det_fps = len(yolo_detections) / duration
    _log_overall_performance_stats(duration, det_fps)
    names = results.names
    if ot_labels_enabled:
        return [yolo_detections, names]
    det_config = _get_det_config(weights, conf, iou, size, chunksize, normalized)
    return _convert_detections_chunks(yolo_detections, names, det_config)


def _get_batch_of_frames(video_capture, batch_size):
    """Reads the the next batch_size frames from VideoCapture.

    Args:
        video_capture (obj): VideoCapture instance.
        batch_size (int): batch size.

    Returns:
        gotFrame (bool): True if there are more frames to read.
        False if no more frames can be read.
        batch(list): batch of frames.
    """
    batch = []
    gotFrame = False
    for _ in range(batch_size):
        gotFrame, img = video_capture.read()
        if gotFrame:
            batch.append(img)
        else:
            break
    return gotFrame, batch


def _log_overall_performance_stats(duration, det_fps):
    log.info("All Chunks done in {0:0.2f} s ({1:0.2f} fps)".format(duration, det_fps))


def _log_batch_performances_stats(
    batch_no, t_start, t_trans, t_det, t_list, batch_size
):
    batch_no = "batch_no: {:d}".format(batch_no)
    transformed_batch = "trans: {:0.4f}".format(t_trans - t_start)
    det = "det: {:0.4f}".format(t_det - t_start)
    add_list = "list: {:0.4f}".format(t_list - t_det)
    batch_len = "batch_size: {:d}".format(batch_size)
    fps = "fps: {:0.1f}".format(batch_size / (t_det - t_start))
    log_msg = f"{batch_no}, {transformed_batch}, {det}, {add_list}, {batch_len}, {fps}"
    log.info(
        log_msg
    )  # BUG: #162 Logs twice from yolo.py (with and without formatting)


def _add_detection_results(detections, results, normalized):
    """Adds detection result to an existing list.

    Args:
        detections (list): the existing list containing detections.
        results (list): detection results.
        normalized (bool): True if results are normalized. False otherwise.

    Returns:
        list: the detections list with the newly added
    """
    if normalized:
        detections.extend([i.tolist() for i in results.xywhn])
    else:
        detections.extend([i.tolist() for i in results.xywh])

transform = transforms.Compose(
        [transforms.Normalize((0.5,), (0.5,))])

def _add_detection_feature_results(yolo_detections_feature, yolo_detections, img_batchs, model_siam, normalized,
                                   width, height, dir_features, files_feature):
    all_obj = []
    # name_batch = str(uuid.uuid4())
    for i, frame in enumerate(yolo_detections):
        img_batch = img_batchs[i]
        for obj in frame:
            x = obj[0]
            y = obj[1]
            w = obj[2]
            h = obj[3]
            if normalized:
                x = int(x * width)
                y = int(y * height)
                w = int(w * width)
                h = int(h * height)
            else:
                x = int(x)
                y = int(y)
                w = int(w)
                h = int(h)
            x = x - w // 2
            y = y - h // 2

            img = cv2.resize(img_batch[0][y: y+h, x: x+w, :], (90, 90))

            all_obj.append(img)

    all_obj_np = np.array(all_obj)
    img_tensor = torch.from_numpy(all_obj_np.transpose((0, 3, 1, 2))).contiguous().to(torch.float).div(255)
    # transformed_image1 = transform(img_tensor)
    transformed_image1 = img_tensor.sub_(0.5).div_(0.5)
    if torch.cuda.is_available():
        transformed_image1 = transformed_image1.cuda()
    else:
        transformed_image1 = transformed_image1.cpu()

    feat = model_siam.feature(transformed_image1)
    # f_list = feat.tolist()[0]
    n = 0
    for frame in yolo_detections:
        out = []

        for _ in range(len(frame)):

            name_f = str(uuid.uuid4())
            files_feature[name_f] = feat[n].tolist()
            # with open(os.path.join(dir_features, name_f), "w") as f:
            #     json.dump(feat[n].tolist(), f)

            out.append(name_f)
            n += 1
        yolo_detections_feature.extend([out])

    # with open(os.path.join(dir_features, name_batch), "w") as f:
    #     json.dump(dict_features, f)


def _delete_small_detection(yolo_detections, w, h, normalized):
    end_row = yolo_detections[-1]
    for_delete = []
    for row in end_row:
        if normalized:
            if row[2] < 0.05 and row[3] < 0.05:
                for_delete.append(row)
        else:
            if row[2] < w * 0.05 and row[3] < h * 0.05:
                for_delete.append(row)

    for row in for_delete:
        end_row.remove(row)


def loadmodel(weights, conf, iou):
    log.info(f"Try loading model {weights}")
    t1 = perf_counter()

    if Path(weights).is_file() and Path(weights).suffix == ".pt":
        model = torch.hub.load(
            repo_or_dir="ultralytics/yolov5",  # cv516Buaa/tph-yolov5 ?
            model="custom",
            path=weights,
            # source="local",
            force_reload=True,
        )
        # cv516Buaa/tph-yolov5: model.amp = False ?
        # cv516Buaa/tph-yolov5: model = torch.jit.load(weights) ?
    elif weights == 'our_y5_s': # kav 200922
        model = torch.hub.load(
            repo_or_dir="ultralytics/yolov5",  # cv516Buaa/tph-yolov5 ?
            model="custom",
            path='checkpoints/best_s.pt',
            # source="local",
            # force_reload=True,
        )
        if torch.cuda.is_available():
            model = model.cuda()
        else:
            model = model.cpu()
    elif weights == 'our_y5_l': # kav 200922
        model = torch.hub.load(
            repo_or_dir="ultralytics/yolov5",
            model="custom",
            path='checkpoints/best_l.pt',
        )
        if torch.cuda.is_available():
            model = model.cuda()
        else:
            model = model.cpu()
    elif weights == 'our_y5_x': # kav 200922
        model = torch.hub.load(
            repo_or_dir="ultralytics/yolov5",
            model="custom",
            path='checkpoints/best_x.pt',
        )
        if torch.cuda.is_available():
            model = model.cuda()
        else:
            model = model.cpu()

    elif weights in torch.hub.list(github="ultralytics/yolov5", force_reload=True):

        if torch.cuda.is_available():
            model = torch.hub.load(
                repo_or_dir="ultralytics/yolov5",
                model=weights,
                pretrained=True,
                force_reload=True,
            ).cuda()
        else:
            model = torch.hub.load(
                repo_or_dir="ultralytics/yolov5",
                model=weights,
                pretrained=True,
                force_reload=True,
            ).cpu()
    else:
        raise AttributeError(
            "weights has to be path to .pt or valid model name "
            "from https://pytorch.org/hub/ultralytics_yolov5/"
        )

    model.conf = conf
    model.iou = iou
    model.agnostic = True  # kav 200922

    t2 = perf_counter()
    log.info(f"Model loaded in {round(t2 - t1)} sec")
    return model


def _get_vidconfig(file, width, height, fps, frames):
    return {
        "file": str(Path(file).stem),
        "filetype": str(Path(file).suffix),
        "width": width,
        "height": height,
        "fps": fps,
        "frames": frames,
    }


def _get_det_config(weights, conf, iou, size, chunksize, normalized):
    return {
        "detector": "YOLOv5",
        "weights": weights,
        "conf": conf,
        "iou": iou,
        "size": size,
        "chunksize": chunksize,
        "normalized": normalized,
    }


def _convert_detections_chunks(yolo_detections, names, det_config):
    result = []
    for no, yolo_detection in enumerate(yolo_detections):
        detection = []
        for yolo_bbox in yolo_detection:
            bbox = {
                "class": names[int(yolo_bbox[5])],
                "conf": yolo_bbox[4],
                "x": yolo_bbox[0],
                "y": yolo_bbox[1],
                "w": yolo_bbox[2],
                "h": yolo_bbox[3],
            }

            detection.append(bbox)
        data = {str(no + 1): {"classified": detection}}
        result.append({"det_config": det_config, "data": data})
    return result


def _convert_detections(yolo_detections, names, vid_config, det_config, yolo_detections_features):
    data = {}
    no = 0
    for yolo_detection, yolo_feature in zip(yolo_detections, yolo_detections_features):
        # TODO: #81 Detections: Nested dict instead of dict of lists of dicts
        detection = []
        for yolo_bbox, y_f in zip(yolo_detection, yolo_feature):
            bbox = {
                "class": names[int(yolo_bbox[5])],
                "conf": yolo_bbox[4],
                "x": yolo_bbox[0],
                "y": yolo_bbox[1],
                "w": yolo_bbox[2],
                "h": yolo_bbox[3],
                "feature": y_f
            }
            detection.append(bbox)
        data[str(no)] = {"classified": detection}
        # data[str(no + 1)] = {"classified": detection}
        no += 1
    return {"vid_config": vid_config, "det_config": det_config, "data": data}


def _createchunks(chunksize, files):
    # TODO: Remove method
    if type(files) is str:
        return [files]
    elif _containsvideo(files):
        return files
    elif chunksize == 0:
        return [files]
    else:
        chunk_starts = range(0, len(files), chunksize)
        return [files[i : i + chunksize] for i in chunk_starts]


def _containsvideo(file_chunks):
    if len(file_chunks) == 0:
        return False

    if type(file_chunks[0]) is str:
        file = Path(file_chunks[0])
        vid_formats = [
            ".mov",
            ".avi",
            ".mp4",
            ".mpg",
            ".mpeg",
            ".m4v",
            ".wmv",
            ".mkv",
        ]
        if file.suffix in vid_formats:
            return True
    return False


def saveimage(img_batch, yolo_detections, width, height, normalized, batch_no):
    img = Image.fromarray(img_batch[0])
    d = ImageDraw.Draw(img)
    for box in yolo_detections[-1]:
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        if normalized:
            x = int(x * width)
            y = int(y * height)
            w = int(w * width)
            h = int(h * height)
        else:
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)
        x = x - w // 2
        y = y - h // 2
        d.rectangle(((x, y), (x+w, y+h)), width=3)
    img.save(f'/home/neptun/PycharmProjects/cars/{batch_no-1}.jpg')
    pass
