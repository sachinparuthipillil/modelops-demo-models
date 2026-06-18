from teradataml import *
from aoa import (aoa_create_context, ModelContext, DatasetInfo)

import pandas as pd
import numpy as np
import lightgbm as lgb

def drop_view(view_name, schema_name):            
    try:
        execute_sql("drop view {}.{};".format(schema_name,view_name))
    except:
        pass
    
def drop_table(table_name, schema_name):            
    try:
        execute_sql("drop table {}.{};".format(schema_name,table_name))
    except:
        pass    

def run_task(context: ModelContext, **kwargs):
    aoa_create_context()
    print(kwargs)
    table_name_tb = 'cd_score_tmp_tb'
    schema_name_tb = 'ADLTRD_TB'
    schema_name_cse = 'demo_user'
    
    # Original table for the features
    table_name = table_name_tb #kwargs.get("table_name")
    schema_name = schema_name_cse #kwargs.get("schema_name") 
    
    drop_view(table_name,schema_name)
    drop_table(table_name, schema_name)
    
    execute_sql("database {};".format(schema_name))

    print("Creating view "+table_name +" In schema: "+ schema_name)
    
    qry ='''REPLACE VIEW {} AS 
        SEL * FROM  ANTISELECT( 
             ON (SEL * FROM 
                 (SEL PRIM_PARTY_ID 
                -----------------------------  
                -- 5. Transformations and missing values 
                ----------------------------- 
                --def transform_features 
                ,AGE   as age_at_acquisition 
                ,date '2025-01-01' + random(0,300) as last_qff_transaction_date 
                ,ROUND(COALESCE(member_tenure_at_acquisition, 68),0) as  member_tenure_at_acquisition 
                ,COALESCE(dom_segs_one_year,0) as dom_segs_one_year 
                ,COALESCE(int_segs_one_year,0) as int_segs_one_year 
                ,COALESCE(dom_eco_segs_one_year,0) as dom_eco_segs_one_year 
                ,COALESCE(int_eco_segs_one_year,0) as int_eco_segs_one_year 
                ,COALESCE(dom_bus_segs_one_year,0) as dom_bus_segs_one_year 
                ,COALESCE(int_bus_segs_one_year,0) as int_bus_segs_one_year 
                ,COALESCE(ppp_flight_one_year,0) as ppp_flight_one_year 
                ,COALESCE(cl_flight_one_year,0) as cl_flight_one_year 
                ,COALESCE(cl_pl_flight_one_year,0) as cl_pl_flight_one_year 
                ,COALESCE(flight_spend_one_year,0) as flight_spend_one_year 
                ,COALESCE(CURR_POINT_BAL,0) as CURR_POINT_BAL 

                ,GENDER_DESC 
                ,PSTL_CODE 

                ----------------------------- 
                -- 6. Tier mapping 
               ----------------------------- 
                ,CASE WHEN TIER_CLUB_DESC IS IN('FFBR', 'QCBR', 'BR', 'BRQC') THEN 1 
                      WHEN TIER_CLUB_DESC IS IN('FFSL', 'QCSL', 'SL', 'SLQC') THEN 2 
                      WHEN TIER_CLUB_DESC IS IN('FFGD', 'GD') THEN 3 
                      WHEN TIER_CLUB_DESC IS IN('FFPL', 'FFPO', 'CLPO', 'CLQF', 'PL', 'CL') THEN 4 
                      ELSE 0 
                 END as  tier 
                ,new_to_qff 
                ,indirect_earn_last_2_year 
                ,RANDOM(0,1) as recovery 
                ,COALESCE(Decile,0) as Decile 
                ,COALESCE(Percentile,0) as Percentile 
            FROM cc_scoring_tb as c 
 
            ----------------------------- 
            -- 3. SEIFA merge 
            ----------------------------- 
            LEFT OUTER JOIN 
            (SEL Postcode 
                    ,Score 
                    ,COALESCE(Decile, 5) as Decile 
                    ,Percentile  
            FROM seifa) as a 
            on CAST(c.PSTL_CODE as INT)= a.Postcode 

            --This are the filtering functions from python 
            ----------------------------- 
            -- 2. Filtering  
            ----------------------------- 
            --def filter_scored_data 
            -- no need for testing uncomment when deploy 
            WHERE 1=1 
            -- AND AGE between 18 and 70 
            -- AND member_tenure_at_acquisition <3  

            ) as tb 
            WHERE 1=1 
               -- AND last_qff_transaction_date > date '2025-01-01' 
            )  
        USING 
          Exclude ('PSTL_CODE','last_qff_transaction_date') 
    )AS dt;'''.format(table_name)
                 

    execute_sql(qry)
    # Get features from the model
    print("loading model")
    
    cur = get_context().raw_connection().cursor()
    cur.execute(f"SELECT file_content FROM qa_model_table")
    row = cur.fetchone()
  
    print(row[0][:100])
    content = row[0]
    
    print("writing model to the file")
    with open("temp_file.txt", "w", encoding="utf-8") as f:
        f.write(content)
    print("converting model to LGB")
    lgb_model=lgb.Booster(model_file='temp_file.txt')

    #Light GB features
    feature_names_lgb = lgb_model.feature_name()
    print(feature_names_lgb)
    
    # Data Set
    cc_score_td = DataFrame.from_query(context.dataset_info.sql)
    td_columns_list= cc_score_td.columns
    
    
     #-----------------------------
     # 6. Prepare and Aligh features
     # -----------------------------

    # Change this to retrive all the columns from pickle file and the lgb model. For now work around is a list
    for col in feature_names_lgb:
        if col not in td_columns_list:
            print(f"⚠️ Missing column '{col}', filling with 0 or blank.")
            drop_table('cd_score_tmp1',schema_name)
            drop_view('cd_score_tmp1',schema_name)
            
            qry = '''CREATE TABLE cd_score_tmp1 AS(
                    SEL a.* 
                     ,0 as {}
                    FROM cd_score_tmp_tb as a)
                    WITH DATA PRIMARY INDEX(PRIM_PARTY_ID);'''.format(col)
            
            execute_sql(qry)
            
            #drop old vew
            drop_table(table_name, schema_name)
            drop_view(table_name,schema_name)
         
            #recreate with a new column
            qry='''CREATE TABLE {} AS(
                  SEL * FROM cd_score_tmp1
            )WITH DATA PRIMARY INDEX(PRIM_PARTY_ID);'''.format(table_name)
            execute_sql(qry)
           
  
    cc_td = DataFrame(in_schema(schema_name, table_name))
    print(cc_td.head())
    
