"""
OTVision module to track road users in frames detected by OTVision
"""
import random

import numpy as np
# based on IOU Tracker written by Erik Bochinski originally licensed under the
# MIT License, see
# https://github.com/bochinski/iou-tracker/blob/master/LICENSE.

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
# GNU General Public License for more detectionsails.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import pandas as pd
import torch
import os
import json

from OTVision.config import CONFIG

from .iou_util import iou


def make_bbox(obj):
    return (
        obj["x"] - obj["w"] / 2,
        obj["y"] - obj["h"] / 2,
        obj["x"] + obj["w"] / 2,
        obj["y"] + obj["h"] / 2,
    )


def center(obj):
    return obj["x"], obj["y"]


def track_iou(
    detections,
    sigma_l=CONFIG["TRACK"]["IOU"]["SIGMA_L"],
    sigma_h=CONFIG["TRACK"]["IOU"]["SIGMA_H"],
    sigma_iou=CONFIG["TRACK"]["IOU"]["SIGMA_IOU"],
    t_min=CONFIG["TRACK"]["IOU"]["T_MIN"],
    t_miss_max=CONFIG["TRACK"]["IOU"]["T_MISS_MAX"],
    model_siam=None, # kav 200922
    w_frame=0,
    h_frame=0,
    dir_features=''
):
    """
    Simple IOU based tracker.
    See "High-Speed Tracking-by-Detection Without Using Image Information
    by E. Bochinski, V. Eiselein, T. Sikora" for
    more information.
    Args:
         detections (list): list of detections per frame, usually generated
         by util.load_mot
         sigma_l (float): low detection threshold.
         sigma_h (float): high detection threshold.
         sigma_iou (float): IOU threshold.
         t_min (float): minimum track length in frames.
    Returns:
        list: list of tracks.
    """

    tracks_active = []
    # tracks_finished = []
    tracks_geojson = {"type": "FeatureCollection", "features": []}
    vehID = 0
    vehIDs_finished = []
    new_detections = {}
    shorts_tracks = []

    with open(os.path.join(dir_features, 'all_f')) as f:
        all_f = json.load(f)

    for n_frame, frame_num in enumerate(detections):
        if n_frame % 1000 == 0:
            cash_features = {}
        detections_frame = detections[frame_num]["classified"]
        # apply low threshold to detections
        dets = [det for det in detections_frame if det["conf"] >= sigma_l]
        new_detections[frame_num] = {}
        updated_tracks = []
        saved_tracks = []
        for track in tracks_active:
            if len(dets) > 0:
                # get det with highest iou
                best_match = max(
                    dets, key=lambda x: iou(track["bboxes"][-1], make_bbox(x))
                )
                if iou(track["bboxes"][-1], make_bbox(best_match)) >= sigma_iou:
                    track["frames"].append(int(frame_num))
                    track["bboxes"].append(make_bbox(best_match))
                    track["center"].append(center(best_match))
                    track["conf"].append(best_match["conf"])
                    track["class"].append(best_match["class"])
                    track["max_conf"] = max(track["max_conf"], best_match["conf"])
                    track["age"] = 0

                    fff = best_match["feature"]
                    if cash_features.get(fff) is None:
                        with open(os.path.join(dir_features, all_f[fff])) as f:
                            cash_features.update(json.load(f))
                    feat = cash_features.get(fff)
                    track["features"].append(feat) # kav 200922

                    updated_tracks.append(track)

                    # remove best matching detection from detections
                    del dets[dets.index(best_match)]
                    # best_match["vehID"] = track["vehID"]
                    best_match["first"] = False
                    best_match["gluing"] = False

                    new_detections[frame_num][track["vehID"]] = best_match

            # if track was not updated
            if len(updated_tracks) == 0 or track is not updated_tracks[-1]:
                # finish track when the conditions are met
                if track["age"] < t_miss_max:
                    track["age"] += 1
                    saved_tracks.append(track)
                elif (
                    track["max_conf"] >= sigma_h
                    and track["frames"][-1] - track["frames"][0] >= t_min
                    and len(track["frames"]) >= t_min
                ):
                    # tracks_finished.append(track)
                    vehIDs_finished.append(track["vehID"])
                    tracks_geojson["features"].append(
                        {
                            "type": "Feature",
                            "geometry": {
                                "type": "LineString",
                                "coordinates": track["center"],
                            },
                            "properties": {
                                "max_conf": track["max_conf"],
                                "ID": track["vehID"],
                                "start_frame": track["frames"][0],
                            },
                        }
                    )
                if track["age"] >= t_miss_max and len(track["frames"]) < t_min:
                    shorts_tracks.append(track["vehID"])

        # kav 200922
        if len(dets) > 0:
            # if frame_num in ['2075',]:
            #     pass
            n_points = 10
            for_del = []
            for_del_det = []
            for iq, det in enumerate(dets):
                dict_score = {}
                for t, trakk in enumerate(saved_tracks):
                    list_score_siam = []
                    list_r = []
                    n_point = 0
                    for nim in range(n_points):
                        if nim == 0:
                            kk = -1
                        else:
                            kk = -nim - 1
                            # kk = - nim * 10
                            # if nim * 10 >= len(trakk["features"]):
                            #     kk = random.choice(list(range(len(trakk["features"]))))
                        if nim == len(trakk["features"]):
                            break
                        w_track = trakk['bboxes'][kk][2] - trakk['bboxes'][kk][0]
                        h_track = trakk['bboxes'][kk][3] - trakk['bboxes'][kk][1]
                        if not (det['w'] * 0.5 < w_track < det['w'] * 2 and
                                det['h'] * 0.5 < h_track < det['h'] * 2):
                            continue
                        # if not (w_track * 0.5 < det['w'] < w_track * 2 and
                        #         h_track * 0.5 < det['h'] < h_track * 2):
                        #     continue

                        fnim = trakk["features"][kk]
                        im1 = torch.from_numpy(np.array(fnim))

                        fff = det["feature"]
                        if cash_features.get(fff) is None:
                            with open(os.path.join(dir_features, all_f[fff])) as f:
                                cash_features.update(json.load(f))
                        feat = cash_features.get(fff)


                        im2 = torch.from_numpy(np.array(feat))
                        if torch.cuda.is_available():
                            im1 = im1.cuda()
                            im2 = im2.cuda()
                        im1 = im1.float().unsqueeze(0)
                        im2 = im2.float().unsqueeze(0)

                        score_siam = model_siam.ev(im1, im2).item()
                        # print(score_siam)
                        list_score_siam.append(score_siam)
                        p1 = trakk["center"][-1]
                        p2 = center(det)
                        r = abs(p1[0] - p2[0]) / w_frame + abs(p1[1] - p2[1]) / h_frame
                        list_r.append(r)
                        n_point += 1
                    dict_score[t] = {'siam': list_score_siam,
                                      'r': list_r,
                                      'trakk': trakk,
                                      'n_point': n_point}
                best_track = None
                best_s = 0
                best_r = 0
                # print(dict_score)
                for k, v in dict_score.items():
                    # mean_score = sum(v['siam'])/v['n_point']
                    # if len(v['siam']) >= 3:
                    #     mean_score = (sum(v['siam']) - max(v['siam']) - min(v['siam']))/(v['n_point'] - 2)
                    # else:
                    if len(v['siam']) != 0:
                        mean_score = max(v['siam'])
                        rr = sum(v['r'])/v['n_point']
                    else:
                        mean_score = 0
                        rr = 1
                    # print(mean_score)
                    if mean_score > 0.1 and rr < 0.3 and best_s < mean_score:
                        best_track = k
                        best_s = mean_score
                        best_r = rr

                if not best_track is None:
                    best_match = dets[iq]
                    track = dict_score[best_track]['trakk']
                    track["frames"].append(int(frame_num))
                    track["bboxes"].append(make_bbox(best_match))
                    track["center"].append(center(best_match))
                    track["conf"].append(best_match["conf"])
                    track["class"].append(best_match["class"])
                    track["max_conf"] = max(track["max_conf"], best_match["conf"])
                    track["age"] = 0

                    fff = best_match["feature"]
                    if cash_features.get(fff) is None:
                        with open(os.path.join(dir_features, all_f[fff])) as f:
                            cash_features.update(json.load(f))
                    feat = cash_features.get(fff)


                    track["features"].append(feat)


                    print(f'frame {frame_num}, returne id {track["vehID"]}, score {round(best_s,3)}, dist {round(best_r,3)}, '
                          f'class: {best_match["class"]}')

                    updated_tracks.append(track)

                    # remove best matching detection from detections
                    # del dets[dets.index(best_match)]
                    for_del_det.append(dets.index(best_match))
                    for_del.append(trakk)

                    # best_match["vehID"] = track["vehID"]
                    best_match["first"] = False
                    best_match["gluing"] = True

                    new_detections[frame_num][track["vehID"]] = best_match
                else:
                    id_tr = None
                    score_tr = 0
                    for k_dict, v_dict in dict_score.items():
                        if len(v_dict['siam']) != 0:
                            max_score = max(v_dict['siam'])
                            rr = sum(v_dict['r']) / v_dict['n_point']

                        else:
                            max_score = 0
                            rr = 1
                        if max_score > score_tr and rr < 0.3:
                            score_tr = max_score
                            id_tr = v_dict['trakk']['vehID']

                    print('frame {}. Best track id {}, score {:.03f}'.format(frame_num, id_tr, score_tr))

                for fd in for_del:
                    saved_tracks.remove(fd)
                for_del = []
            for fd in sorted(for_del_det, reverse=True):
                del dets[fd]

        # TODO: Alter der Tracks
        # create new tracks
        new_tracks = []
        for det in dets:
            vehID += 1

            fff = det["feature"]
            if cash_features.get(fff) is None:
                with open(os.path.join(dir_features, all_f[fff])) as f:
                    cash_features.update(json.load(f))
            feat = cash_features.get(fff)

            new_tracks.append(
                {
                    "frames": [int(frame_num)],
                    "bboxes": [make_bbox(det)],
                    "center": [center(det)],
                    "conf": [det["conf"]],
                    "class": [det["class"]],
                    "max_class": det["class"],
                    "max_conf": det["conf"],
                    "vehID": vehID,
                    "start_frame": int(frame_num),
                    "age": 0,
                    'features': [feat],
                }
            )
            # det["vehID"] = vehID
            det["first"] = True
            det["gluing"] = False
            new_detections[frame_num][vehID] = det
        tracks_active = updated_tracks + saved_tracks + new_tracks

    # finish all remaining active tracks
    # tracks_finished += [
    #     track
    #     for track in tracks_active
    #     if (
    #         track["max_conf"] >= sigma_h
    #         and track["frames"][-1] - track["frames"][0] >= t_min
    #     )
    # ]
    tracks_geojson["features"] += [
        {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": track["center"],
            },
            "properties": {
                "max_conf": track["max_conf"],
                "ID": track["vehID"],
                "start_frame": track["frames"][0],
            },
        }
        for track in tracks_active
        if (
            track["max_conf"] >= sigma_h
            and track["frames"][-1] - track["frames"][0] >= t_min
        )
    ]

    # for track in tracks_finished:
    #     track["max_class"] = pd.Series(track["class"]).mode().iat[0]
    for track_geojson in tracks_geojson["features"]:
        track_geojson["properties"]["max_class"] = (
            pd.Series(track["class"]).mode().iat[0]
        )
    detections = new_detections
    print(sorted(list(set(shorts_tracks))))
    for_de_short = []
    # TODO: #82 Use dict comprehensions in track_iou
    for frame_num, frame_det in new_detections.items():
        for vehID, det in frame_det.items():
            del det["feature"]
            if vehID not in vehIDs_finished:
                det["finished"] = False
            else:
                det["finished"] = True
                # det["label"] = tracks[tracks["vehID"] == det["vehID"]]["max_label"]
            if vehID in list(set(shorts_tracks)):
                for_de_short.append((frame_num, vehID))
    for row in for_de_short:
        del new_detections[row[0]][row[1]]
    # return tracks_finished
    # TODO: #83 Remove unnessecary code (e.g. for tracks_finished) from track_iou
    return (
        new_detections,
        tracks_geojson,
        vehIDs_finished,
    )
