import streamlit as st 

def colR():
    st.header("ClimaXplainability Visualizations")

    st.caption("Run Metrics (both averaged across runs, and invidiaul for predictions)")
    st.table()


    st.subheader("Visual Predictions vs. Ground Truth")
    st.image('/localhome/advit/aug30_exps/all_vars/pred/pred_at_0006_hrs.png') 

    
    """ Analyzing Specific Areas Within the Map"""

    st.subheader(f"Temperature vs. Time Plots for Specific Locations")
    
    location = st.selectbox("Select a location", ["Location 1", "Location 2", "Location 3"])

