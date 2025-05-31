import sys
import os
import asyncio

import streamlit as st
import streamlit_antd_components as sac

# somewhat tricky way to solve the import error in backend 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

async def main():
    st.set_page_config(
        page_title="InsRec",
        page_icon="🎹",
        layout="wide",
        initial_sidebar_state="expanded", 
        menu_items={
            'Get Help': 'https://github.com/Ladbaby/InsRec',
            'Report a bug': "https://github.com/Ladbaby/InsRec",
            'About': "## Made by Ladbaby"
        }
    )

    from frontend.tabs.MIC import MIC

    with st.sidebar:
        menu = sac.menu(
            items=[
                sac.MenuItem('InsRec', icon='house-fill'),
            ],
            key='menu',
            open_index=[1]
        )

    if menu == "InsRec":
        await MIC()

asyncio.run(main())