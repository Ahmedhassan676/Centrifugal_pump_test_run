import streamlit as st
import pandas as pd
from scipy import optimize
import matplotlib.pyplot as plt
import numpy as np

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



<h2>Units used</h2>

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
  
</table>
    """
    st.markdown(html_temp, unsafe_allow_html=True)


    
    
    if "df" not in st.session_state:
        st.session_state.df = pd.DataFrame(columns=["flow_rate", 
                                                "head", 
                                                "power", 
                                                "efficiency", 
                                                "NPSHr"])
        #

    st.subheader("Add Pump manufacturing data")


    s1 = st.selectbox('Using CSV file?',('No','Yes'), key = 'fat')
    #Case 1 no pump FAT csv file
    if s1 == 'No':

        num_new_rows = st.sidebar.number_input("Add Rows",5,5, key = 'fat_input')
        ncol = st.session_state.df.shape[1]  # col count
        rw = -1

        with st.form(key="add form", clear_on_submit= True):
            cols = st.columns(ncol)
            rwdta = []

            for i in range(ncol):
                #rwdta.append(cols[i].text_input(st.session_state.df.columns[i]))
                rwdta.append(cols[i].number_input(st.session_state.df.columns[i]))
            # you can insert code for a list comprehension here to change the data (rwdta) 
            # values into integer / float, if required

            if st.form_submit_button("Add"):
                if st.session_state.df.shape[0] == num_new_rows:
                    st.error("Add row limit reached. Cant add any more records..")
                else:
                    rw = st.session_state.df.shape[0] + 1
                    st.info(f"Row: {rw} / {num_new_rows} added")
                    st.session_state.df.loc[rw] = rwdta

                    if st.session_state.df.shape[0] == num_new_rows:
                        st.error("Add row limit reached...")
        df_uploaded = pd.DataFrame(st.session_state.df)
        st.dataframe(df_uploaded)

        #power_factor_input
        st.subheader("Add Pump power factors data")
        if "df_test" not in st.session_state:
            st.session_state.df_factor = pd.DataFrame(columns=["load%","current","motor_efficiency","power_factor"])
        num_new_rows = st.sidebar.number_input("Add Rows",4,5, key = 'fat_power_input')
        ncol = st.session_state.df_factor.shape[1]  # col count
        rw = -1
        #df_uploaded_power= pd.DataFrame(st.session_state.df_factor)
        
        #else: st.dataframe(df_uploaded_power)
        
        with st.form(key="add form power factor", clear_on_submit= True):
            cols = st.columns(ncol)
            rwdta = []

            for i in range(ncol):
                #rwdta.append(cols[i].text_input(st.session_state.df.columns[i]))
                rwdta.append(cols[i].number_input(st.session_state.df_factor.columns[i]))
            # you can insert code for a list comprehension here to change the data (rwdta) 
            # values into integer / float, if required

            if st.form_submit_button("Add"):
                if st.session_state.df_factor.shape[0] == num_new_rows:
                    st.error("Add row limit reached. Cant add any more records..")
                else:
                    if st.session_state.df_factor.shape[0] == 0:
                        rw = st.session_state.df_factor.shape[0] + 1    
                        st.info(f"Row: {rw} / {num_new_rows} added")
                        st.session_state.df_factor.loc[rw] = rwdta
                        st.info(f'{st.session_state.df_factor.shape[0]}, {rw}')
                    else:
                        rw = st.session_state.df_factor.shape[0] + 1
                        st.info(f"Row: {rw} / {num_new_rows} added")
                        st.session_state.df_factor.loc[rw] = rwdta

                    if st.session_state.df_factor.shape[0] == num_new_rows:
                        st.error("Add row limit reached...")
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
            x = df_uploaded['flow_rate']
            y =  df_uploaded['head']
            def objective(x, a, b, c):
                return a * x + b * x**2 + c
            popt, _ = optimize.curve_fit(objective, x, y)
                # summarize the parameter values
            a_h, b_h, c_h = popt

            def calculate_head(q):
                calc_head = c_h + b_h*(q**2) + a_h*q
                return calc_head
            #print('y = %.5f * x + %.5f * x^2 + %.5f' % (a, b, c))
            max_flow_rate = x.max() + 0.1*x.max()
            calc_head = []
            df_fitted = pd.DataFrame()

            for i in range(0,int(max_flow_rate+1),1):
                calc_head_i = c_h + b_h*(i**2) + a_h*i
                calc_head.append(calc_head_i)
            
            # Efficiency fitting
            x = df_uploaded['flow_rate']
            y =  df_uploaded['efficiency']
            popt, _ = optimize.curve_fit(objective, x, y)
            a, b, c = popt
            calc_eff = []
            for j in range(0,int(max_flow_rate+1),1):
                calc_eff_j = c + b*(j**2) + a*j
                calc_eff.append(calc_eff_j)
        

            df_fitted['fitted_eff'] = pd.Series(calc_eff).astype('float64')
            df_fitted['head'] = pd.Series(calc_head).astype('float64')
            df_fitted['flow_rate'] = pd.Series(np.arange(max_flow_rate+1))
            max_head = df_fitted['head'].max() + 0.1*df_fitted['head'].max()

            #Plot recorded data
            fig,axs = plt.subplots(1,2)
            axs[0].plot(df_fitted['flow_rate'],df_fitted['head'])
            axs[0].scatter(df_uploaded['flow_rate'],df_uploaded['head'])
            axs[0].set_xlim([0, max_flow_rate])
            axs[0].set_ylim([0, max_head])
            axs[1].plot(df_fitted['flow_rate'],df_fitted['fitted_eff'])
            axs[1].scatter(df_uploaded['flow_rate'],df_uploaded['efficiency'])
            axs[1].set_xlim([0, max_flow_rate])
            st.pyplot(fig)

    else:
        try:
            uploaded_file = st.file_uploader('Choose a file', key = 1)
            df_uploaded=pd.read_csv(uploaded_file)
            st.dataframe(df_uploaded)
            x = df_uploaded['flow_rate']
            y =  df_uploaded['head']
            def objective(x, a, b, c):
	            return a * x + b * x**2 + c
            popt, _ = optimize.curve_fit(objective, x, y)
            # summarize the parameter values
            a_h, b_h, c_h = popt

            def calculate_head(q):
                calc_head = c_h + b_h*(q**2) + a_h*q
                return calc_head
            #print('y = %.5f * x + %.5f * x^2 + %.5f' % (a, b, c))
            max_flow_rate = x.max() + 0.1*x.max()
            calc_head = []
            df_fitted = pd.DataFrame()

            for i in range(0,int(max_flow_rate+1),1):
                calc_head_i = c_h + b_h*(i**2) + a_h*i
                calc_head.append(calc_head_i)
            
            # Efficiency fitting
            x = df_uploaded['flow_rate']
            y =  df_uploaded['efficiency']
            popt, _ = optimize.curve_fit(objective, x, y)
            a, b, c = popt
            calc_eff = []
            for j in range(0,int(max_flow_rate+1),1):
                calc_eff_j = c + b*(j**2) + a*j
                calc_eff.append(calc_eff_j)
            

            df_fitted['fitted_eff'] = pd.Series(calc_eff).astype('float64')
            df_fitted['head'] = pd.Series(calc_head).astype('float64')
            df_fitted['flow_rate'] = pd.Series(np.arange(max_flow_rate+1))
            max_head = df_fitted['head'].max() + 0.1*df_fitted['head'].max()
        except ValueError:
            st.write('Please Choose your pump curve file')

        # power factor data calculations
        try:
            uploaded_file_power = st.file_uploader('Choose a file', key = 2)
            df_uploaded_power=pd.read_csv(uploaded_file_power)
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
            fig,axs = plt.subplots(1,2)
            axs[0].plot(df_fitted['flow_rate'],df_fitted['head'])
            axs[0].scatter(df_uploaded['flow_rate'],df_uploaded['head'])
            axs[0].set_xlim([0, max_flow_rate])
            axs[0].set_ylim([0, max_head])
            axs[1].plot(df_fitted['flow_rate'],df_fitted['fitted_eff'])
            axs[1].scatter(df_uploaded['flow_rate'],df_uploaded['efficiency'])
            axs[1].set_xlim([0, max_flow_rate])
            st.pyplot(fig)
    
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

        num_new_rows = st.sidebar.number_input("Add Rows",5,5, key = 'test_input')
        ncol = st.session_state.df_test.shape[1]  # col count
        rw = -1

        with st.form(key="addform_test", clear_on_submit= True):
            cols = st.columns(ncol)
            rwdta = []

            for i in range(ncol):
                #rwdta.append(cols[i].text_input(st.session_state.df.columns[i]))
                rwdta.append(cols[i].number_input(st.session_state.df_test.columns[i]))
            # you can insert code for a list comprehension here to change the data (rwdta) 
            # values into integer / float, if required

            if st.form_submit_button("Add"):
                if st.session_state.df_test.shape[0] == num_new_rows:
                    st.error("Add row limit reached. Cant add any more records..")
                else:
                    rw = st.session_state.df_test.shape[0] + 1
                    st.info(f"Row: {rw} / {num_new_rows} added")
                    st.session_state.df_test.loc[rw] = rwdta

                    if st.session_state.df_test.shape[0] == num_new_rows:
                        st.error("Add row limit reached...")

        st.dataframe(st.session_state.df_test)
        
        #if not df_test_uploaded.empty:

    

        if st.button("Reveal Calculations Table", key = 'calculations_table'):
            df_test_uploaded = pd.DataFrame(st.session_state.df_test).copy()
            df_test_uploaded['head'] = (((df_test_uploaded['p_discharge']-df_test_uploaded['p_suction'])*10)/df_test_uploaded['specific_gravity']).astype('float64')
            df_test_uploaded['hydraulic_power'] = (df_test_uploaded['flow_rate']*df_test_uploaded['head']*df_test_uploaded['specific_gravity']*9.81)/(3600)
            df_test_uploaded['load%'] = 100*(df_test_uploaded['current'] / float(df_uploaded_power[df_uploaded_power['load%']==100]['current']))
            pf_inter =  np.interp(np.array(df_test_uploaded.loc[:,'load%']),np.array(df_uploaded_power['load%']),np.array(df_uploaded_power['power_factor']))  
            motor_inter =  np.interp(np.array(df_test_uploaded.loc[:,'load%']),np.array(df_uploaded_power['load%']),np.array(df_uploaded_power['motor_efficiency']))
            df_test_uploaded['power_factor'] = pf_inter
            df_test_uploaded['motor_efficiency'] = motor_inter
            df_test_uploaded['power'] = (1.73*df_test_uploaded['current']*df_test_uploaded['voltage']*df_test_uploaded['power_factor']*df_test_uploaded['motor_efficiency'])/100000
            df_test_uploaded['efficiency'] = (df_test_uploaded['hydraulic_power']/df_test_uploaded['power'])*100
            df_test_uploaded['calculated_head'] = calculate_head(df_test_uploaded['flow_rate'])
            df_test_uploaded['tolerance'] = ((df_test_uploaded['head']-df_test_uploaded['calculated_head'])/df_test_uploaded['calculated_head'])*100
            st.dataframe(df_test_uploaded)
            

        
            if df_fitted['head'].max() >  df_test_uploaded['head'].max():
                max_head = df_fitted['head'].max() + 0.1*df_fitted['head'].max()
            else:
                max_head = df_test_uploaded['head'].max() + 0.1*df_test_uploaded['head'].max()
            
            #if st.button("Plot Pump curves", key = 'test_fat_curve'):
            fig,axs = plt.subplots(2,2)
            axs[0,0].plot(df_fitted['flow_rate'],df_fitted['head'])
            axs[0,0].scatter(df_test_uploaded['flow_rate'],df_test_uploaded['head'])
            axs[0,0].set_xlim([0, max_flow_rate])
            axs[0,0].set_ylim([0, max_head])
            
            axs[0,1].plot(df_fitted['flow_rate'],df_fitted['fitted_eff'])
            axs[0,1].scatter(df_test_uploaded['flow_rate'],df_test_uploaded['efficiency'])
            
            axs[1,0].plot(df_uploaded['flow_rate'],df_uploaded['power'])
            axs[1,0].scatter(df_test_uploaded['flow_rate'],df_test_uploaded['power'])

            st.pyplot(fig)
    else:
        try:
            uploaded_test_file = st.file_uploader('Choose a file', key = "test_run_data")
            df_test_uploaded=pd.read_csv(uploaded_test_file)
            #st.dataframe(df_test_uploaded)
            df_test_uploaded['head'] = (((df_test_uploaded['p_discharge']-df_test_uploaded['p_suction'])*10)/df_test_uploaded['specific_gravity']).astype('float64')
            df_test_uploaded['hydraulic_power'] = (df_test_uploaded['flow_rate']*df_test_uploaded['head']*df_test_uploaded['specific_gravity']*9.81)/(3600)
            df_test_uploaded['load%'] = 100*(df_test_uploaded['current'] / float(df_uploaded_power[df_uploaded_power['load%']==100]['current']))
            
       

            pf_inter =  np.interp(np.array(df_test_uploaded['load%']),np.array(df_uploaded_power['load%']),np.array(df_uploaded_power['power_factor']))  
            motor_inter =  np.interp(np.array(df_test_uploaded['load%']),np.array(df_uploaded_power['load%']),np.array(df_uploaded_power['motor_efficiency']))
            df_test_uploaded['power_factor'] = pd.Series(pf_inter)
            df_test_uploaded['motor_efficiency'] = pd.Series(motor_inter)
            df_test_uploaded['power'] = (1.73*df_test_uploaded['current']*df_test_uploaded['voltage']*df_test_uploaded['power_factor']*df_test_uploaded['motor_efficiency'])/100000
            df_test_uploaded['efficiency'] = (df_test_uploaded['hydraulic_power']/df_test_uploaded['power'])*100
            df_test_uploaded['calculated_head'] = calculate_head(df_test_uploaded['flow_rate'])
            df_test_uploaded['tolerance'] = ((df_test_uploaded['head']-df_test_uploaded['calculated_head'])/df_test_uploaded['calculated_head'])*100

            st.dataframe(df_test_uploaded.loc[:,["flow_rate", 
                                                "p_suction",
                                                "p_discharge", 
                                                "current",
                                                "voltage",
                                                "specific_gravity"]])
            

            if st.button("Reveal Calculations Table", key = 'calculations_table'):
                st.dataframe(df_test_uploaded)

            
            if df_fitted['head'].max() >  df_test_uploaded['head'].max():
                max_head = df_fitted['head'].max() + 0.1*df_fitted['head'].max()
            else:
                max_head = df_test_uploaded['head'].max() + 0.1*df_test_uploaded['head'].max()
            
            if st.button("Plot Pump curves", key = 'test_fat_curve'):
                fig,axs = plt.subplots(2,2)
                axs[0,0].plot(df_fitted['flow_rate'],df_fitted['head'])
                axs[0,0].scatter(df_test_uploaded['flow_rate'],df_test_uploaded['head'])
                axs[0,0].set_xlim([0, max_flow_rate])
                axs[0,0].set_ylim([0, max_head])
                
                axs[0,1].plot(df_fitted['flow_rate'],df_fitted['fitted_eff'])
                axs[0,1].scatter(df_test_uploaded['flow_rate'],df_test_uploaded['efficiency'])
                
                axs[1,0].plot(df_uploaded['flow_rate'],df_uploaded['power'])
                axs[1,0].scatter(df_test_uploaded['flow_rate'],df_test_uploaded['power'])

                st.pyplot(fig)
        except ValueError:
            st.write('Please Choose your test run data file')
        
        


if __name__ == '__main__':
    main()