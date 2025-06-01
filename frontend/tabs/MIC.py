import os
import yaml
from pathlib import Path
from dataclasses import fields
from typing import BinaryIO

import streamlit as st
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch import Tensor

from backend.exp.exp_main import Exp_Main
from backend.utils.ExpConfigs import ExpConfigs
from backend.data.data_provider.data_factory import data_provider

PYOMNITS_PATH = "backend/"

async def MIC():
    st.header(":wave: Let's Start From Here!")

    uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "ogg", "flac", "aac", "m4a"])

    audio_bytes: BinaryIO = None
    if uploaded_file is not None:
        # To read file as bytes:
        audio_bytes = uploaded_file.getvalue()
        audio_type: str = os.path.splitext(uploaded_file.name)[-1]

        st.audio(audio_bytes, format=f"audio/{audio_type}")

    model_option = st.selectbox(
        "Choose a model",
        [
            "Pyraformer",
            "Linear"
        ]
    )
    dataset_option = "OpenMIC"

    n_classes_option = st.slider("Number of instruments to display", min_value=1, max_value=20, value=2, step=1)

    if st.button("Analyze", type="primary"):
        yaml_configs_path = Path(f"{PYOMNITS_PATH}configs/{model_option}/{dataset_option}.yaml")

        if yaml_configs_path.exists():
            with open(yaml_configs_path, 'r', encoding="utf-8") as stream:
                try:
                    yaml_configs: dict = yaml.safe_load(stream)

                    unknown_key_list: list[str] = []
                    known_key_list: list[str] = [field.name for field in fields(ExpConfigs)]
                    for key in yaml_configs.keys():
                        if key not in known_key_list:
                            st.warning(f"Remove unknown key {key} from model's configs")
                            unknown_key_list.append(key)
                    for key in unknown_key_list:
                        yaml_configs.pop(key)

                    configs = ExpConfigs(**yaml_configs)
                    configs.checkpoints_test = f"{PYOMNITS_PATH}storage/pretrained/{dataset_option}/{model_option}"

                    if audio_bytes is None:
                        st.error("Upload an audio file first to continue")
                    else:
                        _, data_loader = data_provider(configs, "test")
                        data_loader.dataset.load_custom_data(audio_bytes) # OpenMIC.load_custom_data

                        class_dict = {
                            "🪗 Accordion": {
                                "wiki": "https://en.wikipedia.org/wiki/Accordion"
                            }, 
                            "🪕 Banjo": {
                                "wiki": "https://en.wikipedia.org/wiki/Banjo"
                            }, 
                            "Bass": {
                                "wiki": "https://en.wikipedia.org/wiki/Bass_(sound)"
                            }, 
                            "Cello": {
                                "wiki": "https://en.wikipedia.org/wiki/Cello"
                            }, 
                            "Clarinet": {
                                "wiki": "https://en.wikipedia.org/wiki/Clarinet"
                            }, 
                            "Cymbals": {
                                "wiki": "https://en.wikipedia.org/wiki/Cymbals"
                            }, 
                            "🥁 Drums": {
                                "wiki": "https://en.wikipedia.org/wiki/Drum"
                            }, 
                            "Flute": {
                                "wiki": "https://en.wikipedia.org/wiki/Flute"
                            }, 
                            "🎸 Guitar": {
                                "wiki": "https://en.wikipedia.org/wiki/Guitar"
                            }, 
                            "Mallet Percussion": {
                                "wiki": "https://en.wikipedia.org/wiki/Keyboard_percussion_instrument"
                            }, 
                            "Mandolin": {
                                "wiki": "https://en.wikipedia.org/wiki/Mandolin"
                            }, 
                            "Organ": {
                                "wiki": "https://en.wikipedia.org/wiki/Organ_(music)"
                            }, 
                            "🎹 Piano": {
                                "wiki": "https://en.wikipedia.org/wiki/Piano"
                            }, 
                            "🎷 Saxophone": {
                                "wiki": "https://en.wikipedia.org/wiki/Saxophone"
                            }, 
                            "Synthesizer": {
                                "wiki": "https://en.wikipedia.org/wiki/Synthesizer"
                            }, 
                            "Trombone": {
                                "wiki": "https://en.wikipedia.org/wiki/Trombone"
                            }, 
                            "🎺 Trumpet": {
                                "wiki": "https://en.wikipedia.org/wiki/Trumpet"
                            }, 
                            "Ukulele": {
                                "wiki": "https://en.wikipedia.org/wiki/Ukulele"
                            }, 
                            "🎻 Violin": {
                                "wiki": "https://en.wikipedia.org/wiki/Violin"
                            }, 
                            "🗣️ Voice": {
                                "wiki": "https://en.wikipedia.org/wiki/Human_voice"
                            }
                        } # obtained from class-map.json of OpenMIC dataset

                        exp = Exp_Main(configs)
                        pred_class_all = []
                        total_length = 0 # seconds
                        batch: dict[Tensor] # typing support
                        model_output: dict[Tensor] # typing support
                        for batch, model_output in exp.inference(data_loader):
                            pred_class: Tensor = model_output["pred_class"] # (BATCH_SIZE, N_CLASSES)
                            probabilities: np.ndarray = F.softmax(pred_class, dim=1).detach().cpu().numpy() # (BATCH_SIZE, N_CLASSES)

                            x_mask: Tensor = batch["x_mask"] # (BATCH_SIZE, SEQ_LEN), used to indicate padding values
                            is_padding: np.ndarray = (x_mask.sum(dim=1) > 0).int().detach().cpu().numpy() # (BATCH_SIZE)
                            probabilities_filtered = probabilities[is_padding > 0] # reemove padded samples in batch

                            pred_class_all.append(probabilities_filtered)
                            total_length += int(x_mask.sum().detach().cpu().item() / 16000) # divide by 16k sampling rate
                        pred_class_all = np.concat(pred_class_all, axis=0) # (N_SAMPLES, N_CLASSES)

                        pred_class_all_sum = np.sum(pred_class_all, axis=0) # (N_CLASSES)
                        top_k_indices = np.argsort(pred_class_all_sum)[-n_classes_option:] # (n_classes_option)
                        top_k_class_names = [list(class_dict.keys())[index] for index in top_k_indices] # (n_classes_option)
                        pred_class_all_top_k = pred_class_all[:, top_k_indices] # (N_SAMPLES, n_classes_option)
                        pred_class_all_top_k_percentage = pred_class_all_top_k * 100 # (N_SAMPLES, n_classes_option)

                        pred_class_all_top_k_percentage = pred_class_all_top_k_percentage.repeat(10, axis=0)[:total_length]

                        st.metric(label="Most likely", value=list(class_dict.keys())[top_k_indices[-1]], delta=f"{np.average(pred_class_all[:, top_k_indices[-1]]) * 100:.2f} %")
                        st.link_button("wiki", class_dict[list(class_dict.keys())[top_k_indices[-1]]]["wiki"], icon=":material/open_in_new:")
                        st.line_chart(
                            pd.DataFrame(
                                pred_class_all_top_k_percentage,
                                columns=top_k_class_names
                            ),
                            x_label="Time (seconds)",
                            y_label="Probability (%)"
                        )
                except yaml.YAMLError as exc:
                    print(f"{PYOMNITS_PATH}utils/configs.py: Exception when parsing {yaml_configs_path}: {exc}")
                    exit(1) 
        else:
            st.error(f"""Config file for model '{model_option}' does not exist under '{yaml_configs_path}'. Follow these procedures to generate one:

- `cd backend`
- `sh scripts/{model_option}/{dataset_option}.sh` Then kill it via ctrl+c after you see any outputs in terminal.""")
