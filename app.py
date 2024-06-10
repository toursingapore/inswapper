import os
import cv2
import copy
import insightface
import onnxruntime
import numpy as np
from PIL import Image
from typing import List, Union
import streamlit as st

# Function definitions remain the same

def getFaceSwapModel(model_path: str):
    model = insightface.model_zoo.get_model(model_path)
    return model

def getFaceAnalyser(model_path: str, providers, det_size=(320, 320)):
    face_analyser = insightface.app.FaceAnalysis(name="buffalo_l", root="./checkpoints", providers=providers)
    face_analyser.prepare(ctx_id=0, det_size=det_size)
    return face_analyser

def get_one_face(face_analyser, frame:np.ndarray):
    face = face_analyser.get(frame)
    try:
        return min(face, key=lambda x: x.bbox[0])
    except ValueError:
        return None

def get_many_faces(face_analyser, frame:np.ndarray):
    try:
        face = face_analyser.get(frame)
        return sorted(face, key=lambda x: x.bbox[0])
    except IndexError:
        return None

def swap_face(face_swapper, source_faces, target_faces, source_index, target_index, temp_frame):
    source_face = source_faces[source_index]
    target_face = target_faces[target_index]
    return face_swapper.get(temp_frame, target_face, source_face, paste_back=True)

def process(source_img: Union[Image.Image, List], target_img: Image.Image, source_indexes: str, target_indexes: str, model: str):
    providers = onnxruntime.get_available_providers()
    face_analyser = getFaceAnalyser(model, providers)
    model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), model)
    face_swapper = getFaceSwapModel(model_path)
    target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
    target_faces = get_many_faces(face_analyser, target_img)
    num_target_faces = len(target_faces)
    num_source_images = len(source_img)

    if target_faces is not None:
        temp_frame = copy.deepcopy(target_img)
        if isinstance(source_img, list) and num_source_images == num_target_faces:
            for i in range(num_target_faces):
                source_faces = get_many_faces(face_analyser, cv2.cvtColor(np.array(source_img[i]), cv2.COLOR_RGB2BGR))
                if source_faces is None:
                    raise Exception("No source faces found!")
                temp_frame = swap_face(face_swapper, source_faces, target_faces, i, i, temp_frame)
        elif num_source_images == 1:
            source_faces = get_many_faces(face_analyser, cv2.cvtColor(np.array(source_img[0]), cv2.COLOR_RGB2BGR))
            if source_faces is None:
                raise Exception("No source faces found!")
            if target_indexes == "-1":
                num_iterations = min(len(source_faces), num_target_faces)
                for i in range(num_iterations):
                    source_index = 0 if len(source_faces) == 1 else i
                    temp_frame = swap_face(face_swapper, source_faces, target_faces, source_index, i, temp_frame)
            else:
                source_indexes = source_indexes.split(',')
                target_indexes = target_indexes.split(',')
                for source_index, target_index in zip(source_indexes, target_indexes):
                    source_index = int(source_index)
                    target_index = int(target_index)
                    if source_index >= len(source_faces) or target_index >= len(target_faces):
                        raise ValueError("Invalid face index")
                    temp_frame = swap_face(face_swapper, source_faces, target_faces, source_index, target_index, temp_frame)
        else:
            raise Exception("Unsupported face configuration")
        result = temp_frame
    else:
        raise Exception("No target faces found!")
    
    result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    return result_image

# Streamlit UI
st.title("Face Swap Application")
st.write("This project is developed by Haofan Wang to support face swap in single frame. Multi-frame will be supported soon!")

uploaded_source_images = st.file_uploader("Upload Source Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
uploaded_target_image = st.file_uploader("Upload Target Image", type=["jpg", "jpeg", "png"])
source_indexes = st.text_input("Source Indexes (comma-separated)", "-1")
target_indexes = st.text_input("Target Indexes (comma-separated)", "-1")
face_restore = st.checkbox("Face Restore", False)
background_enhance = st.checkbox("Background Enhance", False)
face_upsample = st.checkbox("Face Upsample", False)
upscale = st.slider("Upscale", 1, 4, 1)
codeformer_fidelity = st.slider("Codeformer Fidelity", 0.0, 1.0, 0.5)
model_path = "./checkpoints/inswapper_128.onnx"

if st.button("Run Face Swap"):
    if uploaded_source_images and uploaded_target_image:
        source_images = [Image.open(img) for img in uploaded_source_images]
        target_image = Image.open(uploaded_target_image)
        result_image = process(source_images, target_image, source_indexes, target_indexes, model_path)
        
        if face_restore:
            from restoration import *
            check_ckpts()
            upsampler = set_realesrgan()
            device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
            codeformer_net = ARCH_REGISTRY.get("CodeFormer")(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, connect_list=["32", "64", "128", "256"]).to(device)
            ckpt_path = "CodeFormer/CodeFormer/weights/CodeFormer/codeformer.pth"
            checkpoint = torch.load(ckpt_path)["params_ema"]
            codeformer_net.load_state_dict(checkpoint)
            codeformer_net.eval()
            result_image = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
            result_image = face_restoration(result_image, background_enhance, face_upsample, upscale, codeformer_fidelity, upsampler, codeformer_net, device)
            result_image = Image.fromarray(result_image)
        
        st.image(result_image, caption="Result Image", use_column_width=True)
        st.success("Face swap completed successfully!")
    else:
        st.error("Please upload both source and target images.")
