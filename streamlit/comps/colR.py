import streamlit as st 

def colR():
    st.header("ClimaXplainability Visualizations")

    col1, col2 = st.columns([0.5, 0.5])



    col1.caption("Run Metrics (both averaged across runs, and individual for predictions)")
    #st.table(st.session_state.state["metrics_table"])
    col1.dataframe(st.session_state.state["run_df"], hide_index=True)


    col2.caption("Prediction Metrics (both averaged across runs, and individual for predictions)")
    #st.table(st.session_state.state["metrics_table"])
    col2.dataframe(st.session_state.state["metrics_df"], hide_index=True)

    st.subheader("Ground Truth vs ClimaX Preds")
    #st.image('/localhome/advit/aug30_exps/all_vars/pred/pred_at_0006_hrs.png') 

    if st.session_state.state["plotly_truth"]: 
        st.plotly_chart(st.session_state.state["plotly_truth"])
    
    if st.session_state.state["plotly_pred"]: 
        st.plotly_chart(st.session_state.state["plotly_pred"])
    


    
    """ Analyzing Specific Areas Within the Map"""

    st.subheader(f"Temperature vs. Time for Specific Locations")
    
    location = st.selectbox("Select a location", ["Location 1", "Location 2", "Location 3"])

