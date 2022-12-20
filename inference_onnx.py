import os
import argparse
import time
import numpy as np
import cv2
import argparse
import onnxruntime
from matplotlib.colors import TABLEAU_COLORS


def color_list():
    # Return first 10 plt colors as (r,g,b) https://stackoverflow.com/questions/51350872/python-from-color-name-to-rgb
    def hex2rgb(h):
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

    return [hex2rgb(h) for h in TABLEAU_COLORS.values()]

colors = color_list()

def xyxy2xywh(bbox, H, W):

    x1, y1, x2, y2 = bbox

    return [0.5*(x1+x2)/W, 0.5*(y1+y2)/H, (x2-x1)/W, (y2-y1)/H]

def load_img(img_file, img_mean=0, img_scale=1/255):
    img = cv2.imread(img_file)[:, :, ::-1]
    img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
    img = (img - img_mean) * img_scale
    img = np.asarray(img, dtype=np.float32)
    img = np.expand_dims(img,0)
    img = img.transpose(0,3,1,2)
    return img


def model_inference(model_path, image_np, device="cpu"):

    providers = ['CUDAExecutionProvider'] if device=="cuda" else ['CPUExecutionProvider']
    session = onnxruntime.InferenceSession(model_path, providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    output = session.run([output_name], {input_name: image_np})

    return output[0][:, :6]


def post_process(img_file, output, score_threshold=0.3, format="xywh"):
    """
    Draw bounding boxes on the input image. Dump boxes in a txt file.
    """
    assert format == "xyxy" or format == "xywh"

    det_bboxes, det_scores, det_labels = output[:, 0:4], output[:, 4], output[:, 5]
    id2names = {
        0: "boneanomaly", 1: "bonelesion", 2: "foreignbody", 
        3: "fracture", 4: "metal", 5: "periostealreaction", 
        6: "pronatorsign", 7:"softtissue", 8:"text"
    }

    img = cv2.imread(img_file)
    H, W = img.shape[:2]
    h, w = 640, 640
    label_txt = ""

    for idx in range(len(det_bboxes)):
        if det_scores[idx]>score_threshold:
            bbox = det_bboxes[idx]
            bbox = bbox @ np.array([[W/w, 0, 0, 0], [0, H/h, 0, 0], [0, 0, W/w, 0], [0, 0, 0, H/h]])
            bbox_int = [int(x) for x in bbox]
            label = det_labels[idx]
            
            if format=="xywh":
                bbox = xyxy2xywh(bbox, H, W)
            label_txt += f"{int(label)} {det_scores[idx]:.5f} {bbox[0]:.5f} {bbox[1]:.5f} {bbox[2]:.5f} {bbox[3]:.5f}\n"

            color_map = colors[int(label)]
            txt = f"{id2names[label]} {det_scores[idx]:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            cv2.rectangle(img, (bbox_int[0], bbox_int[1]), (bbox_int[2], bbox_int[3]), color_map, 2)
            cv2.rectangle(img, (bbox_int[0]-2, bbox_int[1]-text_height-10), (bbox_int[0] + text_width+2, bbox_int[1]), color_map, -1)
            cv2.putText(img, txt, (bbox_int[0], bbox_int[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return img, label_txt

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./yolov7-p6-bonefracture.onnx", help="ONNX model path")
    parser.add_argument("--img-path", type=str, help="input image path")
    parser.add_argument("--dst-path", type=str, default="./predictions", help="folder path destination")
    parser.add_argument("--device", type=str, default="cpu", help="device for onnxruntime provider")
    parser.add_argument("--score-tresh", type=float, default=0.3, help="score treshold")
    parser.add_argument("--bbox-format", type=str, default="xywh", help="bounding box format to save annotation (or xyxy)")
    args = parser.parse_args()

    assert args.device == "cpu" or args.device == "cuda"

    # laod image
    img = load_img(args.img_path)

    # inference
    out = model_inference(args.model_path, img, args.device)

    # post-processing
    start_time = time.time()
    out_img, out_txt = post_process(args.img_path, out, args.score_tresh, args.bbox_format)
    elapsed_time = time.time() - start_time
    print(f"Inferece completed in {elapsed_time:.3f} secs.")

    # save prediciton
    os.makedirs(args.dst_path, exist_ok=True)
    bn = os.path.basename(args.img_path).split(".")[0]
    cv2.imwrite(os.path.join(args.dst_path, bn + ".png"), out_img[..., ::-1])
    with open(os.path.join(args.dst_path, bn + ".txt"), 'w') as f:
        f.write(out_txt)

    print(f"Predicted image and annotations are now saved in {args.dst_path}.")
