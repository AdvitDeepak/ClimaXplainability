import streamlit as st 
from geopy.geocoders import Nominatim


def get_lat_long(location_name):
    geolocator = Nominatim(user_agent="location-locator")
    location = geolocator.geocode(location_name)
    if location:
        return location.latitude, location.longitude
    else:
        return None, None


def colR():
    st.header("ClimaXplainability Visualizations")

    col1, col2 = st.columns([0.5, 0.5])



    col1.caption("Run Metrics (both averaged across runs, and individual for predictions)")
    #st.table(st.session_state.state["metrics_table"])

    if st.session_state.state["run_df"] is None or st.session_state.state["run_df"].empty: 
        col1.info('Waiting for dataframe...', icon="ℹ️")
    else: 
        col1.dataframe(st.session_state.state["run_df"], hide_index=True)

    col2.caption("Prediction Metrics (both averaged across runs, and individual for predictions)")

    if st.session_state.state["run_df"] is None or st.session_state.state["run_df"].empty: 
        col2.info('Waiting for dataframe...', icon="ℹ️")
    else: 
        col2.dataframe(st.session_state.state["metrics_df"], hide_index=True)

    st.subheader("Ground Truth (not shown for now) vs ClimaX Preds")
    #st.image('/localhome/advit/aug30_exps/all_vars/pred/pred_at_0006_hrs.png') 

    if st.session_state.state["plotly_truth"] is None and st.session_state.state["plotly_pred"] is None: 
        st.info("Waiting for Plotly interactive figures...", icon="ℹ️")
    else: 
        if st.session_state.state["plotly_truth"]: 
            st.plotly_chart(st.session_state.state["plotly_truth"])
        
        if st.session_state.state["plotly_pred"]: 
            st.plotly_chart(st.session_state.state["plotly_pred"])
        


    
    location_name = st.text_input("Enter a location (e.g., country, city, state, etc.):")
    if location_name:
        if st.button("Find"):
            lat, lon = get_lat_long(location_name)
            if lat is not None and lon is not None:
                st.success(f"Latitude: {lat}, Longitude: {lon}")
            else:
                st.error("Location not found or coordinates not available.")

    """ Analyzing Specific Areas Within the Map"""

    st.subheader(f"Temperature vs. Time for Specific Locations")
    
    location = st.selectbox("Select a location", ["Location 1", "Location 2", "Location 3"])

