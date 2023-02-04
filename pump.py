import streamlit as st
import pandas as pd
from scipy import optimize
import matplotlib.pyplot as plt
import numpy as np

def add_form(session_df,keys,columns):
            num_new_rows = st.sidebar.number_input("Add Rows",columns,5, key = keys[0])
            ncol = session_df.shape[1]  # col count
            rw = -1

            with st.form(key=keys[1], clear_on_submit= True):
                cols = st.columns(ncol)
                rwdta = []

                for i in range(ncol):
                    
                    rwdta.append(cols[i].number_input(session_df.columns[i]))
                # you can insert code for a list comprehension here to change the data (rwdta) 
                # values into integer / float, if required

                if st.form_submit_button("Add"):
                    if session_df.shape[0] == num_new_rows:
                        st.error("Add row limit reached. Cant add any more records..")
                    else:
                        rw = session_df.shape[0] + 1
                        st.info(f"Row: {rw} / {num_new_rows} added")
                        session_df.loc[rw] = rwdta

                        if session_df.shape[0] == num_new_rows:
                            st.error("Add row limit reached...")

def objective(x, a, b, c):
                return a * x + b * x**2 + c
def df_fit(df):
                x = df['flow_rate']
                y =  df['head']
                z =  df['efficiency']
                popt, _ = optimize.curve_fit(objective, x, y)
                a_h, b_h, c_h = popt
                popt_e, _ = optimize.curve_fit(objective, x, z)
                a, b, c = popt_e
                max_flow_rate = x.max() + 0.1*x.max()
                calc_head = []
                calc_eff = []
                df_fitted = pd.DataFrame()

                for i in range(0,int(max_flow_rate+1),1):
                    calc_head_i = c_h + b_h*(i**2) + a_h*i
                    calc_head.append(calc_head_i)
                    calc_eff_j = c + b*(i**2) + a*i
                    calc_eff.append(calc_eff_j)
                
                df_fitted['fitted_eff'] = pd.Series(calc_eff).astype('float64')
                df_fitted['head'] = pd.Series(calc_head).astype('float64')
                df_fitted['flow_rate'] = pd.Series(np.arange(max_flow_rate+1))
                max_head = df_fitted['head'].max() + 0.1*df_fitted['head'].max()
                return df_fitted, max_head, max_flow_rate,a_h, b_h, c_h
def calculate_head(q,df):
                df_fitted, max_head, max_flow_rate,a_h, b_h, c_h  = df_fit(df)
                calc_head = c_h + b_h*(q**2) + a_h*q
                return calc_head

def test_calculations(df,df_power,df_uploaded):
                df['head'] = (((df['p_discharge']-df['p_suction'])*10)/df['specific_gravity']).astype('float64')
                df['hydraulic_power'] = (df['flow_rate']*df['head']*df['specific_gravity']*9.81)/(3600)
                df['load%'] = 100*(df['current'] / float(df_power[df_power['load%']==100]['current']))
                pf_inter =  np.interp(np.array(df.loc[:,'load%']),np.array(df_power['load%']),np.array(df_power['power_factor']))  
                motor_inter =  np.interp(np.array(df.loc[:,'load%']),np.array(df_power['load%']),np.array(df_power['motor_efficiency']))
                df['power_factor'] = pf_inter
                df['motor_efficiency'] = motor_inter
                df['power'] = (1.73*df['current']*df['voltage']*df['power_factor']*df['motor_efficiency'])/100000
                df['efficiency'] = (df['hydraulic_power']/df['power'])*100
                df['calculated_head'] = calculate_head(df['flow_rate'],df_uploaded)
                df['tolerance'] = ((df['head']-df['calculated_head'])/df['calculated_head'])*100
                return df
def plot_fitted_curves(df_fitted,df_uploaded,max_flow_rate, max_head):
            fig,axs = plt.subplots(1,2)
            axs[0].plot(df_fitted['flow_rate'],df_fitted['head'])
            axs[0].scatter(df_uploaded['flow_rate'],df_uploaded['head'])
            axs[0].set_xlim([0, max_flow_rate])
            axs[0].set_ylim([0, max_head])
            axs[1].plot(df_fitted['flow_rate'],df_fitted['fitted_eff'])
            axs[1].scatter(df_uploaded['flow_rate'],df_uploaded['efficiency'])
            axs[1].set_xlim([0, max_flow_rate])
            st.pyplot(fig)
    
def plot_charts(df,df_test,df_fitted,max_head,max_flow):

                if df_fitted['head'].max() >  df_test['head'].max():
                    max_head = df_fitted['head'].max() + 0.1*df_fitted['head'].max()
                else:
                    max_head = df_test['head'].max() + 0.1*df_test['head'].max()

                fig,axs = plt.subplots(2,2)
                axs[0,0].plot(df_fitted['flow_rate'],df_fitted['head'])
                axs[0,0].scatter(df_test['flow_rate'],df_test['head'])
                axs[0,0].set_xlim([0, max_flow])
                axs[0,0].set_ylim([0, max_head])
                
                axs[0,1].plot(df_fitted['flow_rate'],df_fitted['fitted_eff'])
                axs[0,1].scatter(df_test['flow_rate'],df_test['efficiency'])
                
                axs[1,0].plot(df['flow_rate'],df['power'])
                axs[1,0].scatter(df_test['flow_rate'],df_test['power'])

                return st.pyplot(fig)
def main():
    html_temp="""
    <div style="background-color:lightblue;padding:16px">
    <h2 style="color:black"; text-align:center> Centrifugal Pump Test Run Calculator </h2>
    </div>
    <style>
    table {
    font-family: arial, sans-serif;
    border-collapse: collapse;
    width: 100%;
    }

    td, th {
    border: 1px solid #dddddd;
    text-align: left;
    padding: 8px;
    }


    </style>



    <h3>Units used</h3>

    <table>
    <tr>
        <th>Parameter</th>
        <th>Unit</th>
    </tr>
    <tr>
        <td>Flow rate</td>
        <td>M3/hr</td>
    </tr>
    <tr>
        <td>Pressure (Suction/Discharge)</td>
        <td>Kg/cm2.g</td>
    </tr>
    <tr>
        <td>Head</td>
        <td>m</td>
    </tr>
    <tr>
        <td>Current</td>
        <td>Ampere</td>
    </tr>
    <tr>
        <td>Power (Hydraulic/ motor)</td>
        <td>Kw</td>
    </tr>
    <tr>
        <td>NPSHr</td>
        <td>m</td>
    </tr>
    </table>
        """
    st.markdown(html_temp, unsafe_allow_html=True)

    
    
    if "df" not in st.session_state:
        st.session_state.df = pd.DataFrame(columns=["flow_rate", 
                                                "head", 
                                                "power", 
                                                "efficiency", 
                                                "NPSHr"])
        

    st.subheader("Add Pump manufacturing data")


    s1 = st.selectbox('Using CSV file?',('No','Yes'), key = 'fat')
    #Case 1 no pump FAT csv file
    if s1 == 'No':
        keys = ['fat_input',"add form",]
        
        
        add_form(st.session_state.df,keys,5)


        
        df_uploaded = pd.DataFrame(st.session_state.df)
        st.dataframe(df_uploaded)

        #power_factor_input
        st.subheader("Add Pump power factors data")
        if "df_test" not in st.session_state:
            st.session_state.df_factor = pd.DataFrame(columns=["load%","current","motor_efficiency","power_factor"])

        keys = ['fat_power_input',"add form power factor"]
        add_form(st.session_state.df_factor,keys, 4)
        
        df_uploaded_power= pd.DataFrame(st.session_state.df_factor)
        if 100 not in df_uploaded_power['load%'].values and st.session_state.df_factor.shape[0] == 3:
            rated_current = st.number_input('Insert rated current')
            pf = st.number_input('Insert rated current power factor')
            moto_eff = st.number_input('Insert rated current motor efficiency')
            
            df_uploaded_power.loc[len(df_uploaded_power.index)+1] = [100,rated_current,moto_eff,pf] 
            st.info(f'{len(df_uploaded_power.index)}')  
            st.dataframe(df_uploaded_power) 
        else: st.dataframe(df_uploaded_power) 
        
        
        
        if st.button("Plot fitted Pump curves"):
            try:
                df_fitted, max_head, max_flow_rate,a_h, b_h, c_h = df_fit(df_uploaded)
                #Plot recorded data
                plot_fitted_curves(df_fitted,df_uploaded,max_flow_rate, max_head)
            except (ValueError, TypeError): st.write('Please Check your dataset')

    else: #FAT TABLE
        try:
            html_temp_fat="""
            <style>
            table {
            font-family: arial, sans-serif;
            border-collapse: collapse;
            width: 100%;
            }

            td, th {
            border: 1px solid #dddddd;
            text-align: left;
            padding: 8px;
            }


            </style>
            <p>Your tables should commit to the following form</p>

            <table>
            <tr>
                <th>flow_rate</th>
                <th>head</th>
                <th>power</th>
                <th>efficiency</th>
                
            </tr>
            </table>
            <br>
            <table>
            <tr>
                <th>load%</th>
                <th>current</th>
                <th>motor_efficiency</th>
                <th>power_factor</th>
                
            </tr>
            </table>"""
            st.markdown(html_temp_fat, unsafe_allow_html=True)
            uploaded_file = st.file_uploader('Choose a file', key = 1)
            df_uploaded=pd.read_csv(uploaded_file).sort_values('flow_rate')
            st.dataframe(df_uploaded)
            df_fitted, max_head, max_flow_rate,a_h, b_h, c_h = df_fit(df_uploaded)

           
        except ValueError:
            st.write('Please Choose your pump curve file')

        # power factor data calculations
        try:
            uploaded_file_power = st.file_uploader('Choose a file', key = 2)
            df_uploaded_power=pd.read_csv(uploaded_file_power).sort_values('load%')
            if 100 not in df_uploaded_power['load%'].values:
                rated_current = st.number_input('Insert rated current')
                pf = st.number_input('Insert rated current power factor')
                moto_eff = st.number_input('Insert rated current motor efficiency')
                df_uploaded_power.loc[len(df_uploaded_power.index)] = [100, rated_current,pf,moto_eff]
                st.dataframe(df_uploaded_power)
            else: st.dataframe(df_uploaded_power)
        except ValueError:
            st.write('Please Choose your factor power file')

        #Plot obtained data from file
        if st.button("Plot fitted Pump curves"):
            try:
                plot_fitted_curves(df_fitted,df_uploaded,max_flow_rate, max_head)
            except (ValueError, TypeError): st.write('Please Check your dataset')
    
    #Test Run data gathering
    st.subheader("Add Pump Test Run data")
    if "df_test" not in st.session_state:
        st.session_state.df_test = pd.DataFrame(columns=["flow_rate", 
                                                "p_suction",
                                                "p_discharge", 
                                                "current",
                                                "voltage",
                                                "specific_gravity"
                                                ])
    s1 = st.selectbox('Using CSV file?',('No','Yes'), key = 'test')
    
    if s1 == 'No':
        keys = ['test_input',"addform_test"]
        add_form(st.session_state.df_test,keys,5)
        st.dataframe(st.session_state.df_test)

        if st.button("Reveal Calculations Table", key = 'calculations_table'):
            try:
                df_test_uploaded = pd.DataFrame(st.session_state.df_test).copy().sort_values('flow_rate')
                df_test_uploaded = test_calculations(df_test_uploaded,df_uploaded_power,df_uploaded) 
                st.dataframe(df_test_uploaded)
                plot_charts(df_uploaded,df_test_uploaded,df_fitted,max_head,max_flow_rate)
                
            except (ValueError, TypeError): st.write('Please Check your dataset')

        
            
    else:
        try:
            html_temp_test="""
            <style>
            table {
            font-family: arial, sans-serif;
            border-collapse: collapse;
            width: 100%;
            }

            td, th {
            border: 1px solid #dddddd;
            text-align: left;
            padding: 8px;
            }


            </style>
            <p>Your table should commit to the following form</p>

            <table>
            <tr>
                <th>flow_rate</th>
                <th>p_suction</th>
                <th>p_discharge</th>
                <th>current</th>
                <th>voltage</th>
                <th>specific_gravity</th>
                
            </tr>
            </table>
            """
            st.markdown(html_temp_test, unsafe_allow_html=True)
            uploaded_test_file = st.file_uploader('Choose a file', key = "test_run_data")
            df_test_uploaded=pd.read_csv(uploaded_test_file).sort_values('flow_rate')
            st.dataframe(df_test_uploaded.loc[:,["flow_rate", 
                                                "p_suction",
                                                "p_discharge", 
                                                "current",
                                                "voltage",
                                                "specific_gravity"]])
            

            df_test_uploaded = test_calculations(df_test_uploaded,df_uploaded_power,df_uploaded)    

            
            

            if st.button("Reveal Calculations Table", key = 'calculations_table'):
                try:
                    st.dataframe(df_test_uploaded)
                except (ValueError, TypeError): st.write('Please Check your dataset')
                      
                
            if st.button("Plot Pump curves", key = 'test_fat_curve'):
                plot_charts(df_uploaded,df_test_uploaded,df_fitted,max_head,max_flow_rate)
                
        except ValueError:
            st.write('Please Choose your test run data file')
        
        


if __name__ == '__main__':
    main()