from typing import Any, Dict, List
import streamlit as st
import snowflake.connector
import plotly.figure_factory as ff
import numpy as np
import altair as alt
from streamlit_prophet.lib.dataprep.clean import clean_df
from streamlit_prophet.lib.dataprep.format import (
    add_cap_and_floor_cols,
    check_dataset_size,
    filter_and_aggregate_df,
    format_date_and_target,
    format_datetime,
    print_empty_cols,
    print_removed_cols,
    remove_empty_cols,
    resample_df,
)
from streamlit_prophet.lib.dataprep.split import get_train_set, get_train_val_sets
from streamlit_prophet.lib.exposition.export import display_links, display_save_experiment_button
from streamlit_prophet.lib.exposition.visualize import (
    plot_components,
    plot_future,
    plot_overview,
    plot_performance,
)
from streamlit_prophet.lib.inputs.dataprep import input_cleaning, input_dimensions, input_resampling
from streamlit_prophet.lib.inputs.dataset import (
    input_columns,
    input_dataset,
    input_future_regressors,
)
from streamlit_prophet.lib.inputs.dates import (
    input_cv,
    input_forecast_dates,
    input_train_dates,
    input_val_dates,
)
from streamlit_prophet.lib.inputs.eval import input_metrics, input_scope_eval
from streamlit_prophet.lib.inputs.params import (
    input_holidays_params,
    input_other_params,
    input_prior_scale_params,
    input_regressors,
    input_seasonality_params,
)
from streamlit_prophet.lib.models.prophet import forecast_workflow
from streamlit_prophet.lib.utils.load import load_config, load_image

# Page config
#favicon=st.image(load_image("Darkpoolwhite.png"))
st.set_page_config(page_title="darkpool",page_icon="❄️")

# Load config
config, instructions, readme = load_config(
    "config_streamlit.toml", "config_instructions.toml", "config_readme.toml"
)

# Initialization
dates: Dict[Any, Any] = dict()
report: List[Dict[str, Any]] = []

#Logo
st.image(load_image("Darkpoolwhite.png"), use_column_width=True)
    
# Info
with st.expander("What is darkpool?", expanded=False):
    st.write(readme["app"]["app_intro"])
    st.write("")
st.write("")

# Headers
st.subheader("Train Your Data")
st.caption("Snowflake Account = SNOWCAT2")
st.caption("Snowflake Database = DEMAND")

def init_connection():
    return snowflake.connector.connect(**st.secrets["snowflake"])

conn = init_connection()

#Select Table
def run_initial_query(initial_query):
    with conn.cursor() as cur:
        cur.execute(initial_query)
        # Return a Pandas DataFrame containing all of the results.
        df = cur.fetch_pandas_all()
        option = st.selectbox('Select your dataset', df)
        text1 = "select COLUMN_NAME from DEMAND1.INFORMATION_SCHEMA.COLUMNS where concat(TABLE_CATALOG,'.',TABLE_SCHEMA,'.',TABLE_NAME) = '"
        text3 = "' order by 1 asc;"   
        query_text = text1+option+text3
        if option:
            run_second_query(query_text)
      
def run_second_query(second_query):
    with conn.cursor() as cur:
        cur.execute(second_query)
        # Return a Pandas DataFrame containing all of the results.
        df = cur.fetch_pandas_all()
        option2 = st.selectbox('Select your target column', df)
        
@st.experimental_memo(suppress_st_warning=True)
def run_generic_query(generic_query):
    with conn.cursor() as cur:
        cur.execute(generic_query)
        
run_initial_query("select concat(TABLE_CATALOG,'.',TABLE_SCHEMA,'.',TABLE_NAME) from DEMAND1.INFORMATION_SCHEMA.TABLES where TABLE_SCHEMA in ('PUBLIC');")

def run_baseline_analysis_query(baseline_analysis_query):
    with conn.cursor() as cur:
        cur.execute(baseline_analysis_query)      
        df = cur.fetch_pandas_all()
        baseline = df["AUC"]
        st.write(baseline)   

if 'baseline_button_clicked' not in st.session_state:
    if st.button('Run Baseline Analysis'):
        st.session_state['baseline_button_clicked'] = 'clicked'
        baseline_analysis_query_text = "select AUC from DARKPOOL_COMMON.ML.TRAINING_LOG where TRAINING_JOB = 'baseline';"
        run_baseline_analysis_query(baseline_analysis_query_text)    

#Analyze boost
#analyze = st.checkbox("Show me my potential accuracy boost",value=False,key='analyze')
analyze_query_text = "select distinct TRAINING_JOB as SUPPLIER, AUC, AUC-(select distinct AUC from DARKPOOL_COMMON.ML.TRAINING_LOG where TRAINING_JOB = 'baseline')  as BOOST_POINTS, concat(to_varchar(to_numeric((AUC/(select AUC from DARKPOOL_COMMON.ML.TRAINING_LOG where TRAINING_JOB = 'baseline') - 1)*100,10,0)),'%') as PERCENTAGE_IMPROVEMENT  from DARKPOOL_COMMON.ML.TRAINING_LOG where TRAINING_JOB not in ('baseline');"

@st.experimental_memo(suppress_st_warning=True)
def run_analyze_query(analyze_query):
    with conn.cursor() as cur:
        cur.execute(analyze_query)
        # Return a Pandas DataFrame containing all of the results.
        df = cur.fetch_pandas_all()
        base = alt.Chart(df).mark_bar().encode(x='SUPPLIER', y='BOOST_POINTS')
        st.altair_chart(base, use_container_width=True)
        st.dataframe(df)

st.subheader("Analyze Potential Boost")
if 'analyze_button_clicked' not in st.session_state:
    if st.button('Show me my potential accuracy boost'):
        st.session_state['analyze_button_clicked'] = 'clicked'
        run_analyze_query(analyze_query_text)
    else:
        run_generic_query(analyze_query_text)  

# Show Price
st.subheader("Pricing Model")
#pricing = st.checkbox("Show me my pricing model",value=False,key='analyze')
pricing_query_text = "select concat('$',cast(sum(SUPPLIER_REV_$) as varchar) )as PRICE, concat(cast(cast(INCREASED_ACCURACY*100 as numeric)as varchar), '%') as INCREASED_ACCURACY,cast(TOTAL_ROWS as varchar) as TOTAL_ROWS from DARKPOOL_COMMON.PUBLIC.PRICING_OUTPUT join (select distinct AUC,10,2/(select distinct AUC from DARKPOOL_COMMON.ML.TRAINING_LOG where TRAINING_JOB = 'baseline') - 1 as INCREASED_ACCURACY, TOTAL_ROWS  from DARKPOOL_COMMON.ML.TRAINING_LOG where TRAINING_JOB = 'boost_all') group by 2,3;"

@st.experimental_memo(suppress_st_warning=True)
def run_pricing_query(pricing_query):
    with conn.cursor() as cur:
        cur.execute(pricing_query)
        # Return a Pandas DataFrame containing all of the results.
        df = cur.fetch_pandas_all()
        st.write("The price is calculated as $0.01 per basis point of boost per 1000 rows scored.")
        col1,col2= st.columns(2)
        col1.metric("Basis Points of Boost","2774")
        col2.metric("Price Per Thousand Rows Scored","$27.74")

if 'pricing_button_clicked' not in st.session_state:
    if st.button('Show me my pricing model'):
        st.session_state['pricing_button_clicked'] = 'clicked'
        run_analyze_query(pricing_query_text)
    else:
        run_generic_query(pricing_query_text)  
            
# Execute Boost
st.subheader("Auto-Boost Your Model")
#boost=st.checkbox("Auto-boost my model",value=False,key='boost')
boost_query_text="select concat(TABLE_CATALOG,'.',TABLE_SCHEMA,'.',TABLE_NAME) from DEMAND1.INFORMATION_SCHEMA.TABLES where TABLE_SCHEMA in ('PUBLIC');"

@st.experimental_memo(suppress_st_warning=True)
def run_boost_query(boost_query):
    with conn.cursor() as cur:
        cur.execute(boost_query)
        df = cur.fetch_pandas_all()
        option = st.selectbox('Select your dataset for inference', df)
        
if 'boost_button_clicked' not in st.session_state:
    if st.button('Auto-boost my model'):
        st.session_state['boost_button_clicked'] = 'clicked'
        run_analyze_query(boost_query_text)
    else:
        run_generic_query(boost_query_text)  
    
@st.experimental_memo(suppress_st_warning=True)
def run_interference_query(interference_query):
    with conn.cursor() as cur:
        cur.execute(interference_query)      
        df = cur.fetch_pandas_all()
    st.write("Total Rows scored = 10,000.  Cost of boost = $277.40.")
    st.write ("See a sample of your inferenced data here:")
    st.write(df)
    st.write("")
    st.subheader("Darkpool Weighted Revenue Distribution to Suppliers")
    st.write("$277.40 total boost fee, distributed to:")
    st.image(load_image("pie.png"), use_column_width=True)

interference_query_text="select * from darkpool_common.ml.demand1_scoring_output limit 20;"
if 'interference_button_clicked' not in st.session_state:
    if st.button('Run Interference'):
        st.session_state['inteference_button_clicked'] = 'clicked'
        run_interference_query(interference_query_text)  
