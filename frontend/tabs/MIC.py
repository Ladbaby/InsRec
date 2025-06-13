import io
import os
import yaml
from pathlib import Path
from dataclasses import fields
from typing import BinaryIO

import torch
import streamlit as st
import numpy as np
import pandas as pd
import torch.nn.functional as F
import altair as alt
from torch import Tensor
from pydub import AudioSegment

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
            "Reformer",
            "Informer",
            "Nonstationary_Transformer",
            "Hi_Patch",
            "GRU_D",
            "TimesNet",
            "MambaSimple",
            "TSMixer",
            "Raindrop",
            "LightTS",
            "Transformer",
            "FEDformer",
            "FreTS",
            "DLinear",
            "Linear",
            "Leddam",
            "iTransformer",
            "PrimeNet",
            "SegRNN",
            "Autoformer",
            "PatchTST",
            "MICN",
            "TiDE",
            "Crossformer",
            "FiLM",
        ]
    )
    dataset_option = "OpenMIC"

    n_classes_option = st.slider("Number of instruments to display", min_value=1, max_value=20, value=5, step=1)

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
                        total_length = round(len(AudioSegment.from_file(io.BytesIO(audio_bytes))) / 1000) # seconds
                        model_output: dict[Tensor] # typing support
                        for batch, model_output in exp.inference(data_loader):
                            pred_class: Tensor = model_output["pred_class"] # (BATCH_SIZE, N_CLASSES)
                            probabilities: np.ndarray = F.softmax(pred_class, dim=1).detach().cpu().numpy() # (BATCH_SIZE, N_CLASSES)

                            pred_class_all.append(probabilities)

                        torch.cuda.empty_cache()
                        del exp

                        pred_class_all = np.concat(pred_class_all, axis=0).repeat(10, axis=0)[:total_length] # (TIME_LENGTH, N_CLASSES) each sample is 10s

                        pred_class_all_sum = np.sum(pred_class_all, axis=0) # (TIME_LENGTH)
                        top_k_indices = np.argsort(pred_class_all_sum)[::-1][:n_classes_option] # (n_classes_option)
                        top_k_class_names = [list(class_dict.keys())[index] for index in top_k_indices] # (n_classes_option)
                        pred_class_all_top_k = pred_class_all[:, top_k_indices] # (TIME_LENGTH, n_classes_option)
                        pred_class_all_top_k_percentage = pred_class_all_top_k * 100 # (TIME_LENGTH, n_classes_option)

                        st.metric(label="Most likely", value=top_k_class_names[0], delta=f"{np.average(pred_class_all_top_k_percentage[:, 0]):.2f} %")
                        st.link_button("wiki", class_dict[list(class_dict.keys())[top_k_indices[0]]]["wiki"], icon=":material/open_in_new:")

                        # Prepare data for Altair
                        n_samples, n_classes = pred_class_all_top_k_percentage.shape

                        # Create a long-format DataFrame
                        data_list = []
                        for sample_idx in range(n_samples):
                            for class_idx, class_name in enumerate(top_k_class_names):
                                data_list.append({
                                    'Time (seconds)': sample_idx,
                                    'Instrument': class_name,
                                    'Similarity (%)': pred_class_all_top_k_percentage[sample_idx, class_idx]
                                })

                        df = pd.DataFrame(data_list)

                        # Create the Altair chart
                        chart = alt.Chart(df).mark_line(
                            point=False,
                            strokeWidth=3
                        ).add_selection(
                            alt.selection_multi(fields=['Instrument'])
                        ).encode(
                            x=alt.X('Time (seconds):O', 
                                    title='Time (seconds)',
                                    axis=alt.Axis(labelAngle=0)),
                            y=alt.Y('Similarity (%):Q', 
                                    title='Similarity (%)',
                                    scale=alt.Scale(zero=False)),
                            color=alt.Color('Instrument:N', 
                                            title='Instrument',
                                            sort=top_k_class_names,  # Maintain sorted order in legend
                                            scale=alt.Scale(scheme='category10')),
                            # opacity=alt.condition(alt.datum.Class, alt.value(0.8), alt.value(0.2)),
                            tooltip=['Time (seconds):O', 'Instrument:N', 'Similarity (%):Q']
                        ).interactive()

                        st.altair_chart(chart, use_container_width=True)
                except yaml.YAMLError as exc:
                    print(f"{PYOMNITS_PATH}utils/configs.py: Exception when parsing {yaml_configs_path}: {exc}")
                    exit(1) 
        else:
            st.error(f"""Config file for model '{model_option}' does not exist under '{yaml_configs_path}'. Follow these procedures to generate one:

- `cd backend`
- `sh scripts/{model_option}/{dataset_option}.sh` Then kill it via ctrl+c after you see any outputs in terminal.""")
