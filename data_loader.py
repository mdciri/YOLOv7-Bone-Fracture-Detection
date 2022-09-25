import os
from torch.utils.data import Dataset
from matplotlib.colors import TABLEAU_COLORS
import pandas as pd
import cv2
import glob
import tqdm

def color_list():
    # Return first 10 plt colors as (r,g,b) https://stackoverflow.com/questions/51350872/python-from-color-name-to-rgb
    def hex2rgb(h):
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

    return [hex2rgb(h) for h in TABLEAU_COLORS.values()]  # or BASE_ (8), CSS4_ (148), XKCD_ (949)

class XrayDataset(Dataset):
    def __init__(self, 
                 annotations_file, 
                 img_root,
                 label_root = None,
                 pred_root = None, 
                 conf_thres = 0.2,
                 class_names = [
                    "boneanomaly", "bonelesion", "foreignbody", 
                    "fracture", "metal", "periostealreaction", 
                    "pronatorsign", "softtissue", "text"]):

        self.df = pd.read_csv(annotations_file)
        self.img_root = img_root
        if label_root == None:
            self.label_root = img_root
        else:
            self.label_root = label_root
        self.pred_root = pred_root
        self.conf_thres = conf_thres
        self.files_list = list(self.df["filestem"])
        self.id2names = {k: v for k, v in enumerate(class_names)}

    def __len__(self):
        return len(self.df)

    def _load(self, img_path, ann_path, prd_path=None):

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        ann = open(ann_path, "r")
        lines = ann.read().splitlines()  

        labels, boxes = [], []
        for line in lines:
            value = line.split()
            labels.append(int(value[0]))
            boxes.append([float(x) for x in value[1:]])

        if prd_path:
            pred_ann = open(prd_path, "r")
            lines = pred_ann.read().splitlines()
            pred_labels, pred_boxes, confs = [], [], []
            for line in lines:
                value = line.split()
                conf = float(value[5])
                if conf >= self.conf_thres:
                    pred_labels.append(int(value[0]))
                    pred_boxes.append([float(x) for x in value[1:5]])
                    confs.append(conf)

            return {"img": img, "labels": labels, "boxes": boxes, "pred_labels": pred_labels, "pred_boxes": pred_boxes, "confs": confs}

        else:
            return {"img": img, "labels": labels, "boxes": boxes}
    

    def __getitem__(self, idx):
        img_path = glob.glob(os.path.join(f"{self.img_root}/*/*/", self.files_list[idx] + ".png"))[0]
        ann_path = glob.glob(os.path.join(f"{self.label_root}/*/*/", self.files_list[idx] + ".txt"))[0]
        if self.pred_root:
            prd_path = glob.glob(os.path.join(f"{self.pred_root}", self.files_list[idx] + ".txt"))[0]
        else:
            prd_path = None

        data = self._load(img_path, ann_path, prd_path)

        return data

    def describe(self):

        counts = {k: 0 for k in self.id2names.keys()}

        for f in tqdm.tqdm(self.files_list, total=self.__len__()):

            ann_path = glob.glob(os.path.join(f"{self.label_root}/*/*/", f + ".txt"))[0]
            ann = open(ann_path, "r")
            lines = ann.read().splitlines()  
    
            for line in lines:
                value = line.split()
                label = int(value[0])
                counts[label] += 1

        print("This dataset contains:")
        for i, n in counts.items():
            print(f"  - {n} labels for {self.id2names[i]} class.")


    def blend_data(self, idx):
        data = self.__getitem__(idx)
        height, width = data["img"].shape
        colors = color_list()

        out = cv2.cvtColor(data["img"], cv2.COLOR_GRAY2RGB)

        if "pred_labels" not in data.keys():

            for label, box in zip(data["labels"], data["boxes"]):
                xc, yc, w, h = box

                start_point = tuple([int((xc-w) * width), int((yc-h) * height)])
                end_point = tuple([int((xc+w) * width), int((yc+h) * height)])
                (text_width, text_height), _ = cv2.getTextSize(self.id2names[label], cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

                cv2.rectangle(out, start_point, end_point, colors[label], 4)
                cv2.rectangle(out, (start_point[0]-2, start_point[1]-text_height-15), (start_point[0] + text_width+2, start_point[1]), colors[label], -1)
                cv2.putText(out, self.id2names[label], (start_point[0], start_point[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        else:

            for label, box in zip(data["labels"], data["boxes"]):
                xc, yc, w, h = box

                start_point = tuple([int((xc-w) * width), int((yc-h) * height)])
                end_point = tuple([int((xc+w) * width), int((yc+h) * height)])
                txt = self.id2names[label] + " (GT)"
                (text_width, text_height), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

                cv2.rectangle(out, start_point, end_point, colors[-1], 4)
                cv2.rectangle(out, (start_point[0]-2, end_point[1]), (start_point[0] + text_width+2, end_point[1]+text_height+15), colors[-1], -1)
                cv2.putText(out, txt, (start_point[0], end_point[1]+25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            for label, box, conf in zip(data["pred_labels"], data["pred_boxes"], data["confs"]):
                xc, yc, w, h = box

                start_point = tuple([int((xc-w) * width), int((yc-h) * height)])
                end_point = tuple([int((xc+w) * width), int((yc+h) * height)])
                txt = self.id2names[label] + f" {conf:.2f}"
                (text_width, text_height), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

                cv2.rectangle(out, start_point, end_point, colors[label], 4)
                cv2.rectangle(out, (start_point[0]-2, start_point[1]-text_height-15), (start_point[0] + text_width+2, start_point[1]), colors[label], -1)
                cv2.putText(out, txt, (start_point[0], start_point[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return out