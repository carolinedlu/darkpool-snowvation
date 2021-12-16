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
st.set_page_config(page_title="darkpool")

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

def run_query(query):
    with conn.cursor() as cur:
        cur.execute(query)

        # Return a Pandas DataFrame containing all of the results.
        df = cur.fetch_pandas_all()
        option = st.selectbox('Select your dataset', df)
        text1 = "select COLUMN_NAME from DEMAND1.INFORMATION_SCHEMA.COLUMNS where concat(TABLE_CATALOG,'.',TABLE_SCHEMA,'.',TABLE_NAME) = '"
        #text2 = "DEMAND.DATA.CUSTOMERS"
        text2 = option
        text3 = "' order by 1 asc;"   
        query_text = text1+text2+text3
        #st.write(query_text)
        if option:
            run_query2(query_text)
        
def run_query2(query_text):
    with conn.cursor() as cur:
        cur.execute(query_text)

        # Return a Pandas DataFrame containing all of the results.
        df = cur.fetch_pandas_all()
        option2 = st.selectbox('Select your dependent variable', df)
        #if option2:
        #    st.write('You have selected dependent variable ',option2)
        #st.write(option)       
        
run_query("select concat(TABLE_CATALOG,'.',TABLE_SCHEMA,'.',TABLE_NAME) from DEMAND1.INFORMATION_SCHEMA.TABLES where TABLE_SCHEMA in ('PUBLIC');") 

if st.button('Run Baseline Analysis'):
    def run_query(query_text2):
      with conn.cursor() as cur:
        cur.execute(query_text2)      
        df = cur.fetch_pandas_all()
        baseline = df["AUC"]
        st.write(baseline)

    run_query("select AUC from DARKPOOL_COMMON.ML.TRAINING_LOG where TRAINING_JOB = 'baseline';")            



#Analyze boost

## Add column + line chart 

st.subheader("Analyze Potential Boost")
analyze = st.checkbox("Show me my potential accuracy boost",value=False,key='analyze')

if analyze==True:
    def run_query(query):
        with conn.cursor() as cur:
            cur.execute(query)

            # Return a Pandas DataFrame containing all of the results.
            df = cur.fetch_pandas_all()
            base = alt.Chart(df).mark_bar().encode(x='TRAINING_JOB', y='POINTS')
            st.altair_chart(base, use_container_width=True)
            st.dataframe(df)

            
            
            
if analyze==False:
    def run_query(query):
        with conn.cursor() as cur:
            cur.execute(query)
                


run_query("select distinct TRAINING_JOB, AUC, AUC-(select distinct AUC from DARKPOOL_COMMON.ML.TRAINING_LOG where TRAINING_JOB = 'baseline')  as POINTS, concat(to_varchar(to_numeric((AUC/(select AUC from DARKPOOL_COMMON.ML.TRAINING_LOG where TRAINING_JOB = 'baseline') - 1)*100,10,0)),'%') as PERCENTAGE_IMPROVEMENT  from DARKPOOL_COMMON.ML.TRAINING_LOG where TRAINING_JOB not in ('baseline');")


# Show Price

st.subheader("Pricing Model")

pricing = st.checkbox("Show me my pricing model",value=False,key='analyze')

if pricing==True:
    def run_query(query):
        with conn.cursor() as cur:
            cur.execute(query)

            # Return a Pandas DataFrame containing all of the results.
            df = cur.fetch_pandas_all()
            col1,col2,col3 = st.columns(3)
            col1.metric("Increased Accuracy", "47%")
            col2.metric("Rows","1,000,023")
            col3.metric("Price","$1,000")
#            st.write("TESTING HERE:")
#            st.dataframe(df)
#            accuracy ='"'+df['INCREASED_ACCURACY']+'"'
#            aw = st.write(accuracy)
#            st.metric("Increased Accuracy",value = aw )
if pricing==False:
    def run_query(query):
        with conn.cursor() as cur:
            cur.execute(query)
            
run_query("select concat('$',cast(sum(SUPPLIER_REV_$) as varchar) )as PRICE, concat(cast(cast(INCREASED_ACCURACY*100 as numeric)as varchar), '%') as INCREASED_ACCURACY,cast(TOTAL_ROWS as varchar) as TOTAL_ROWS from DARKPOOL_COMMON.PUBLIC.PRICING_OUTPUT join (select distinct AUC,10,2/(select distinct AUC from DARKPOOL_COMMON.ML.TRAINING_LOG where TRAINING_JOB = 'baseline') - 1 as INCREASED_ACCURACY, TOTAL_ROWS  from DARKPOOL_COMMON.ML.TRAINING_LOG where TRAINING_JOB = 'boost_all') group by 2,3;") 



# Execute Boost

st.subheader("Auto-Boost Your Model")

boost=st.checkbox("Auto-boost my model",value=False,key='boost')
if boost==True:
    def run_query(query_text):
        with conn.cursor() as cur:
            cur.execute(query_text)      
            df = cur.fetch_pandas_all()
            option = st.selectbox('Select your dataset for inference', df)
if boost==False:
     def run_query(query):
        with conn.cursor() as cur:
            cur.execute(query)         
        
run_query("select concat(TABLE_CATALOG,'.',TABLE_SCHEMA,'.',TABLE_NAME) from DEMAND1.INFORMATION_SCHEMA.TABLES where TABLE_SCHEMA in ('PUBLIC');")        
    
if st.button('Run Inference'):
    def run_query(query_text2):
      with conn.cursor() as cur:
        cur.execute(query_text2)      
        df = cur.fetch_pandas_all()
        st.write(df)

    run_query("select * from darkpool_common.ml.demand1_scoring_output limit 20;")            
