"""

<will have LHS column w/ dials (like a sidebar collapsible, and then RHS where actual runs are shown> 


In the LHS, should be able to: 
Select your NPZ dataset (we have ALL npz files stored0 
Select your input variables
Select your output variables 
Select your input year
Select your lead time(s)
Select run via a button 
Select a dump directory for run json files 

OR, select an existing directory containing run json files 

"""

import datetime
import streamlit as st 
from const import SIDEBAR


# The sidebar 

def colS():
    st.sidebar.header("ClimaXplainability 🌎")
    st.sidebar.caption(SIDEBAR['caption'])


    # Default: set to run run ClimaX + Visualize Results 
    st.session_state.state["selected_mode"] = st.sidebar.toggle('Run ClimaX + Visualize Results', value=True)
    st.sidebar.divider() 


    # Case 1) If turned off, we can load an existing directory + ONLY visualize results 
    if not st.session_state.state["selected_mode"]: 

        st.session_state.state["run_json_dir"] = st.sidebar.text_input("Existing Run Directory Path (where are prediction logs stored?)", "/home/advit/sep12_exps")

        st.sidebar.header("Existing Runs Detected")
        st.session_state.state["selected_run"] = st.sidebar.selectbox("Select an existing run of those identified in Dir Path", ["Run 1", "Run 2", "Run 3"])


    # Case 2) If left on (default), we input the parameters into ClimaX + Run and Visualize Results 
    else:  

        st.sidebar.caption("Paths for NPZ datasets, possible inputs, and possible outputs are specified in `streamlit/const.py`. Please modify as needed.")

        st.session_state.state["selected_npz"] = st.sidebar.selectbox("Select NPZ dataset", ["Dataset 1", "Dataset 2", "Dataset 3"])

        st.session_state.state["selected_inp_var"] = st.sidebar.multiselect("Select input variables", ["Var1", "Var2", "Var3"])
        st.session_state.state["selected_out_var"] = st.sidebar.multiselect("Select output variables", ["VarA", "VarB", "VarC"])


        """ Start Time Selection """ 

        st.sidebar.caption("Based on the year/shard of the selected NPZ file, please choose a start time for ClimaX's predictions from the following range:")

        col1, col2 = st.sidebar.columns([0.6, 0.4])

        with col1: 
            # TODO: select specific day/time down to 6 hour granularity based on selected npz 
            st.session_state.state["selected_start"] = col1.date_input("Select a Date and Time Below:", datetime.date(2019, 7, 6))
        
        with col2: 
            time_select = col2.time_input("(6 hour granularity)", datetime.time(6, 00))
        

        col3, col4 = st.sidebar.columns([0.5, 0.5])

        col3.number_input("Select Lead Time (6-hr steps)", key=1, step=6)
        col4.number_input("Select Number of Predictions", key=2, step=1)

        st.sidebar.caption("For example, `start = Mon`, `lead_time = 24` and `num_preds = 2` means Climax will predict **Tues given Mon** and **Wed given Tues**")

        st.session_state.state["run_json_dir"] = st.sidebar.text_input("Dump Directory Path (where should predictions be stored?)", "")


    st.sidebar.divider() 

    # Run button to trigger model run
    if st.sidebar.button("Run ClimaXplainability 🌎", type="primary"):
        # Add logic to trigger the model run and save results
        st.sidebar.text("Model is running...")

 