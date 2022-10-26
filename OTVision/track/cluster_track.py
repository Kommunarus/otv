import json
import os
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random
from PIL import Image, ImageDraw
import collections
from OTVision.helpers.log import log
import copy


def dbscan(track):
    log.info('start dbscan')
    data = track['data']
    tracks = {}
    for frame, boxes in data.items():
        for id, box in boxes.items():
            if tracks.get(id) is None:
                tracks[id] = [(box['x'], box['y']),]
            else:
                listt = tracks[id]
                listt.append((box['x'], box['y']))
                tracks[id] = listt

    width = int(track['vid_config']['width'])
    height = int(track['vid_config']['height'])

    images = []
    max_light = 250
    min_light = 150

    dict_id = {}
    for iii, (id, boxes) in enumerate(tracks.items()):
        img = Image.new("L", (width, height))
        img1 = ImageDraw.Draw(img)
        N = len(boxes)

        for i, (x, y) in enumerate(boxes):
            color = int((N - i) / N *(max_light - min_light) + min_light)
            r = 0.1 * height
            img1.ellipse([(x-r, y-r), (x+r, y+r)], fill=color, )

        img = img.resize((64, 64))
        images.append(np.array(img))
        dict_id[iii] = id

    flatimg = []
    for img in images:
        flatimg.append(img.flatten())
    print(len(flatimg))
    new_track = copy.deepcopy(track)

    if len(flatimg) > 10:

        flatimg = np.array(flatimg)
        X = flatimg / 255

        # flatimg = np.array(codes)
        # X = StandardScaler().fit_transform(flatimg)

        pca = PCA(n_components=0.9)
        pcaX = pca.fit_transform(X)

        db = DBSCAN(eps=5, min_samples=5, p=1).fit(pcaX)
        labels = db.labels_
        print(set(labels))

        la = -1
        indx = np.where(labels == la)[0]
        id_bad = [dict_id[i] for i in indx]

        new_dict = {}
        for frame, boxes in data.items():
            for id, box in boxes.items():
                if id in id_bad:
                    if new_dict.get(id) is None:
                        new_dict[id] = [(box['x'], box['y'], box['gluing']), ]
                    else:
                        listt = new_dict[id]
                        listt.append((box['x'], box['y'], box['gluing']))
                        new_dict[id] = listt
        for_del = []
        for id, tr in new_dict.items():
            if any([x[2] for x in tr]) == False:
                for_del.append(id)
        for id in for_del:
            del new_dict[id]

        renew = []
        for id, tr in new_dict.items():
            splitlist = []
            loc = []
            for row in tr:
                if row[2] == False:
                    loc.append(row)
                else:
                    splitlist.append(loc)
                    loc = [row]
            else:
                splitlist.append(loc)

            # print(len(splitlist))
            if len(splitlist) == 2:
                renew.append(id)
        print(renew)
        counter = collections.Counter(labels)
        print(sorted(counter.items(), key=lambda k: k[0]))

        drawing = False
        if drawing:

            Nla = len(set(labels))
            for la in sorted(set(labels)):
                indx = np.where(labels == la)[0]
                ddd = np.array([images[idx] for idx in indx])
                mean_cluster = np.mean(ddd, 0)
                plt.imshow(mean_cluster)
                plt.title('{} / {}'.format(la, counter[la]))
                plt.show()
            for la in sorted(set(labels)):
                fig, ax = plt.subplots(5, 5)
                indx = np.where(labels == la)[0]

                if len(indx) > 25:
                    idx = random.choices(indx, k=min(25, len(indx)))
                else:
                    idx = indx
                for i in range(25):
                    if i < len(indx):
                        n = i // 5
                        m = i % 5
                        ax[n, m].imshow(images[idx[i]])
                fig.suptitle('{} / {}'.format(la, counter[la]))
                plt.show()

            img = Image.new("RGB", (width, height))
            img1 = ImageDraw.Draw(img)
            c = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(Nla)]
            for jj, (id, boxes) in enumerate(tracks.items()):
                N = len(boxes)
                cells = []

                for i, (x, y) in enumerate(boxes):
                    # if i == 0:
                    #     x_old, y_old = x, y
                    #     continue
                    # shape = [(x, y), (x_old, y_old)]
                    color = c[labels[jj]]
                    # r = 0.03 * height
                    # img1.ellipse([(x-r, y-r), (x+r, y+r)], fill=color, )
                    img1.point((x, y), fill=color)

            img.show()

        change = {k:False for k in renew}
        for frame, boxes in data.items():
            for id, box in boxes.items():
                if id in renew:
                    if (box['gluing'] is True) and (change[id] is False):
                        change[id] = True
                    if change[id]:
                        new_track['data'][frame]['{}_2'.format(id)] = new_track['data'][frame].pop(id)


    return new_track


if __name__ == '__main__':
    dir = '/home/neptun/PycharmProjects/otVision_test'
    filename = '1.ottrk'

    with open(os.path.join(dir, filename)) as f:
        track = json.load(f)
    track = dbscan(track)
