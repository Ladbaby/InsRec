import os

import streamlit as st

async def MIC():
    st.header(":wave: Let's Start From Here!")

    uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "ogg", "flac", "aac", "m4a"])

    if uploaded_file is not None:
        # To read file as bytes:
        audio_bytes = uploaded_file.getvalue()
        audio_type: str = os.path.splitext(uploaded_file.name)[-1]

        st.audio(audio_bytes, format=f"audio/{audio_type}")
