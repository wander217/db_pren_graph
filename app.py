import time
import pandas
import streamlit as st
import torch
from PIL import Image
import numpy as np
import cv2 as cv
import networkx as nx
import pandas as pd
from models import loader

st.set_page_config(layout="wide")


@st.cache(allow_output_mutation=True, show_spinner=False)
def load_model():
    det, rec, kie = loader()
    return det, rec, kie


def join_bbox(bbox_list):
    pairs = []
    threshold_y = 4
    threshold_x = 20
    check = np.zeros(len(bbox_list))
    for i in range(len(bbox_list)):
        x_min_i, y_min_i, x_max_i, y_max_i = bbox_list[i]
        for j in range(i + 1, len(bbox_list)):
            x_min_j, y_min_j, x_max_j, y_max_j = bbox_list[j]
            cond1 = abs(y_min_i - y_min_j) < threshold_y
            cond2 = abs(x_min_i - x_max_j) < threshold_x
            cond3 = abs(x_max_i - x_min_j) < threshold_x
            if cond1 and (cond2 or cond3):
                pairs.append((i, j))
                check[i] = check[j] = 1
    g = nx.Graph()
    g.add_edges_from(pairs)
    merge_pair = [list(a) for a in list(nx.connected_components(g))]
    out_final = []
    INF = 999999999
    for idx in merge_pair:
        c_bbox = []
        for i in idx:
            c_bbox.append(bbox_list[i])
        sorted_x = sorted(c_bbox, key=lambda x: x[0])
        new_bbox = [INF, INF, -INF, -INF]
        for item in sorted_x:
            new_bbox[0] = min(new_bbox[0], item[0])
            new_bbox[1] = min(new_bbox[1], item[1])
            new_bbox[2] = max(new_bbox[2], item[2])
            new_bbox[3] = max(new_bbox[3], item[3])
        out_final.append(new_bbox)
    for i in range(len(check)):
        if check[i] == 0:
            out_final.append(bbox_list[i])
    return out_final


def process(det, rec, kie, image):
    start = time.time()
    bbox = det.predict(image)
    det_time = time.time() - start
    start = time.time()
    bbox_list = []
    bbox = join_bbox(bbox)
    for i in range(len(bbox)):
        x_min, y_min, x_max, y_max = bbox[i]
        crop_image = image[y_min:y_max + 1, x_min:x_max + 1, :]
        w, h, _ = crop_image.shape
        if w == 0 or h == 0:
            continue
        crop_image = cv.cvtColor(crop_image, cv.COLOR_BGR2RGB)
        label = rec.predict(Image.fromarray(crop_image))
        bbox_list.append([
            x_min, y_min, x_max, y_max, label
        ])
    rec_time = time.time() - start
    start = time.time()
    result = kie.predict(bbox_list)
    kie_time = time.time() - start
    new_data = []
    for i in range(len(bbox_list)):
        x_min, y_min, x_max, y_max, text = bbox_list[i]
        label, _ = result[i]
        if label != "OTHER":
            cv.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            cv.putText(image, str(len(new_data)), (x_min, y_min),
                       cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255))
            new_data.append((text, label))
    return image, new_data, det_time, rec_time, kie_time


def main():
    det, rec, kie = load_model()
    st.title("Invoice extraction")
    option_col1, option_col2 = st.columns(2)

    col_name = ["TEXT", "LABEL"]
    pd_data = pandas.DataFrame(data=[[("\t" * 30) + "Empty" + ("\t" * 30),
                                      ("\t" * 25) + "Empty" + ("\t" * 25)]], columns=col_name)
    with option_col2:
        #     st.table(pd_data)
        placeholder = st.dataframe(pd_data)

    total_text = None
    det_text = None
    rec_text = None
    kie_text = None
    with option_col1:
        container1 = st.container()
        container2 = st.container()
        with container1:
            with st.form("form1", clear_on_submit=True):
                content_file = st.file_uploader("Upload your image here",
                                                type=["jpg", "jpeg", "png"])
                submit = st.form_submit_button("Upload")
                if torch.cuda.is_available():
                    st.text("Your device: GPU")
                else:
                    st.text("Your device: CPU (You can you GPU to archive good performance)")
                if content_file is not None:
                    pil_img = Image.open(content_file)
                    image = np.array(pil_img)
                    # if image.shape[0] < image.shape[1]:
                    #     image = cv.rotate(image, cv.ROTATE_90_CLOCKWISE)

                    if submit:
                        st.balloons()
                        if total_text is not None:
                            total_text.empty()
                        if det_text is not None:
                            det_text.empty()
                        if rec_text is not None:
                            rec_text.empty()
                        if kie_text is not None:
                            kie_text.empty()
                        print(">" * 100)
                        wait_text = st.text("Please wait ...")
                        image, result, det_time, rec_time, kie_time = process(det, rec, kie, image)
                        pd_data = pd.DataFrame(data=result, columns=col_name)
                        with container2:
                            st.image(image, width=500)
                        with option_col2:
                            placeholder.empty()
                            placeholder = st.dataframe(pd_data, width=1200)
                        wait_text.empty()
                        total_text = st.text("Total time: {}s".format(det_time + rec_time + kie_time))
                        det_text = st.text("Detection time: {}s".format(det_time))
                        rec_text = st.text("Recognition time: {}s".format(rec_time))
                        kie_text = st.text("Key extract time: {}s".format(kie_time))


if __name__ == "__main__":
    main()
