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

def load_img(img_file, img_mean=0, img_scale=1/255):
    img = cv2.imread(img_file)[:, :, ::-1]
    img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
    img = (img - img_mean) * img_scale
    img = np.asarray(img, dtype=np.float32)
    img = np.expand_dims(img,0)
    img = img.transpose(0,3,1,2)
    return img


def model_inference(model_path, image_np, cuda=False):

    providers = ['CUDAExecutionProvider'] if cuda else ['CPUExecutionProvider']
    session = onnxruntime.InferenceSession(model_path, providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    output = session.run([output_name], {input_name: image_np})

    return output[0][:, :6]


def post_process(img_file, output, score_threshold=0.3):
    """
    Draw bounding boxes on the input image. Dump boxes in a txt file.
    """
    det_bboxes, det_scores, det_labels = output[:, 0:4], output[:, 4], output[:, 5]
    id2names = {
        0: "boneanomaly", 1: "bonelesion", 2: "foreignbody", 
        3: "fracture", 4: "metal", 5: "periostealreaction", 
        6: "pronatorsign", 7:"softtissue", 8:"text"
    }

    img = cv2.imread(img_file)
    org_size = img.shape[:2]
    img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
    label_txt = ""

    for idx in range(len(det_bboxes)):
        if det_scores[idx]>score_threshold:
            bbox = det_bboxes[idx]
            label = det_labels[idx]

            label_txt += f"{int(label)} {bbox[0]:.5f} {bbox[1]:.5f} {bbox[2]:.5f} {bbox[3]:.5f} {det_scores[idx]:.5f}\n"

            color_map = colors[int(label)]
            txt = f"{id2names[label]} {det_scores[idx]:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            bbox = [int(x) for x in bbox]
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color_map, 2)
            cv2.rectangle(img, (bbox[0]-2, bbox[1]-text_height-10), (bbox[0] + text_width+2, bbox[1]), color_map, -1)
            cv2.putText(img, txt, (bbox[0], bbox[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
    img = cv2.resize(img, org_size[::-1], interpolation=cv2.INTER_LINEAR)
    
    return img, label_txt

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="runs/train/yolov7/weights/yolov7-p6-bonefracture.onnx")
    parser.add_argument("--img-path", type=str)
    parser.add_argument("--dst-path", type=str, default=None)
    args = parser.parse_args()

    # laod image
    img = load_img(args.img_path)

    # inference
    out = model_inference(args.model_path, img)

    # post-processing
    start_time = time.time()
    out_img, out_txt = post_process(args.img_path, out)
    elapsed_time = time.time() - start_time
    print(f"Inferece completed in {elapsed_time:.3f} secs.")

    # save prediciton
    if args.dst_path:
        os.makedirs(args.dst_path, exist_ok=True)
        bn = os.path.basename(args.img_path).split(".")[0]
        cv2.imwrite(os.path.join(args.dst_path, bn + ".png"), out_img[..., ::-1])
        with open(os.path.join(args.dst_path, bn + ".txt"), 'w') as f:
            f.write(out_txt)

        print(f"Predicted image and annotations are now saved in {args.dst_path}.")
