# File: streamlit_app.py
# Last Updated: Tuesday, February 10, 2026
# Description: Main entry point and navigation controller.

import streamlit as st
import single_mode
import batch_mode
from app_settings import ensure_session_state_defaults, hydrate_session_state_from_disk, persist_app_settings

st.set_page_config(page_title="SAXS Simulator", layout="wide", page_icon="⚛️")

# --- Global Session State Initialization ---
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if '_reload_settings_from_disk' not in st.session_state:
    st.session_state._reload_settings_from_disk = True

ensure_session_state_defaults(st.session_state)

if st.session_state.get("_reload_settings_from_disk", False):
    hydrate_session_state_from_disk(st.session_state)
    st.session_state._reload_settings_from_disk = False

# --- Navigation Logic ---
if st.session_state.page == 'home':
    st.title("SAXS Simulator & Analysis Tool")
    st.markdown("### Choose your workflow")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("**Single Run & Interactive Analysis**")
        st.markdown("Simulate one dataset, adjust parameters in real-time, and visualize 1D/2D results interactively.")
        if st.button("Start Single Mode", width="stretch"):
            st.session_state._reload_settings_from_disk = True
            st.session_state.page = 'single'
            st.rerun()
            
    with col2:
        st.success("**Batch Processing**")
        st.markdown("Run multiple simulations automatically by defining parameter sweeps in a table.")
        if st.button("Start Batch Mode", width="stretch"):
            st.session_state._reload_settings_from_disk = True
            st.session_state.page = 'batch'
            st.rerun()

    st.markdown("---")
    st.markdown("""
    **Modules:**
    * **Single Mode:** Real-time feedback, detailed plots, 2D/1D toggles.
    * **Batch Mode:** Upload CSV or use grid editor to run permutations of parameters. Downloads results as ZIP.
    """)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.caption("© Roy Beck Barkai - 2026")

elif st.session_state.page == 'single':
    single_mode.run()

elif st.session_state.page == 'batch':
    batch_mode.run()

persist_app_settings(st.session_state)
