"""
In the RHS, upon selecting a run 
See stats of run (acc, etc)
See file its stored in
See map of prediction at each timestemp (can scroll through or click across top bar, map updates)
- Prediction on top
- Ground Truth at bottom 
See a specific location and get the time chart of temperature at these locations as well 


"""

import streamlit as st

from comps.colS import colS
from comps.colR import colR 

from const import INITIAL_STATE 
 

st.set_page_config(page_title='ClimaXplainability', layout="wide")

if 'demo_state' not in st.session_state: 
    st.session_state.state = INITIAL_STATE 

#lhs, _, rhs = st.columns([0.3, 0.1, 0.6]) 

colS() 

colR()