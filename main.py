import asyncio

import streamlit as st
import streamlit_antd_components as sac

async def main():
    st.set_page_config(
        page_title="MIC",
        page_icon="🎹",
        layout="wide",
        initial_sidebar_state="expanded", 
        menu_items={
            'Get Help': 'https://www.google.com',
            'Report a bug': "https://www.google.com",
            'About': "## Made by Ladbaby"
        }
    )

    from frontend.tabs.MIC import MIC

    with st.sidebar:
        menu = sac.menu(
            items=[
                sac.MenuItem('MIC', icon='house-fill'),
            ],
            key='menu',
            open_index=[1]
        )

    if menu == "MIC":
        await MIC()

asyncio.run(main())