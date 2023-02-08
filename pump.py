import streamlit as st
import pandas as pd
from scipy import optimize
import matplotlib.pyplot as plt
import numpy as np

def add_form(session_df,keys,columns):
            '''This function is used to create the input forms used (FAT data, power factor data and test run data
            Input to this function is:
            1. the session dataframe created
            2. special keys for each form
            3. number of columns for the specified dataframe
            Outputs are displayed directly as you input your observations to the dataframe it updates itself with 
            the observation
            '''
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
                '''This is the objective function used to fit your input FAT data into curves'''
                return a * x + b * x**2 + c
def df_fit(df):
                '''This function is used to fit the input of FAT data into polynomial curves (Q-H and Q-Eff curves)
            Input to this function is:
            1. your FAT data (through input forms of csv file)
            Outputs are as follows:
            1. a fitted dataframe of the FAT data
            2. max head
            3. max flow rate
            4. the objective function parameters a,b and c 
            '''
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
                '''this function is used to calculate any point head given the flow rate based on the fitted curve parameters acquired earlier
                the function takes the flow rate point and the original dataframe where it obtains the fitting parameters a,b and c
                and calculates the required head
                inputs: 
                1. flow rate point
                2. the uploaded / form dataframe of the test run data
                outputs:
                1. Calculated head for the flow rate point
                '''
                df_fitted, max_head, max_flow_rate,a_h, b_h, c_h  = df_fit(df)
                calc_head = c_h + b_h*(q**2) + a_h*q
                return calc_head

def test_calculations(df,df_power,df_uploaded):
                '''This is the function which performs the test run calculations
                Input:
                1. the uploaded FAT data
                2. the uploaded test run data
                3. The uploaded power factor data
                '''
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
            ''' This function plots the fitted curves along with the uploaded FAT data
            input: 
            1. fitted FAT data 
            2. uploaded FAT data
            3. max flow rate
            4. max head
            output:
            two plots of:
            1. fitted Q-H curve and FAT datapoints
            2. fitted Q-Eff curve and FAT data points
            '''
            fig,axs = plt.subplots(1,2)
            fig.set_figheight(8)
            fig.set_figwidth(8)
            axs[0].plot(df_fitted['flow_rate'],df_fitted['head'])
            axs[0].scatter(df_uploaded['flow_rate'],df_uploaded['head'])
            axs[0].set_xlim([0, max_flow_rate])
            axs[0].set_ylim([0, max_head])
            axs[0].set_xlabel("Flow rate (m3/hr)")
            axs[0].set_ylabel("Head (m)")

            axs[1].plot(df_fitted['flow_rate'],df_fitted['fitted_eff'])
            axs[1].scatter(df_uploaded['flow_rate'],df_uploaded['efficiency'])
            axs[1].set_xlim([0, max_flow_rate])
            axs[1].set_xlabel("Flow rate (m3/hr)")
            axs[1].set_ylabel("Efficiency (%)")
            st.pyplot(fig)
    
def plot_charts(df,df_test,df_fitted,max_head,max_flow):
            ''' This function plots the fitted curves along with the uploaded test run data
            input: 
            1. uploaded FAT datafram
            2. uploaded test run dataframe
            3. fitted FAT dataframe
            4. max flow rate
            5. max head
            output:
            3 plots of:
            1. fitted Q-H curve and test run points
            2. fitted Q-Eff curve and test run points
            3. Q-power curve in the original uploaded FAT dataframe and test run points
            '''
            if df_fitted['head'].max() >  df_test['head'].max():
                max_head = df_fitted['head'].max() + 0.1*df_fitted['head'].max()
            else:
                max_head = df_test['head'].max() + 0.1*df_test['head'].max()

            fig,axs = plt.subplots(2,2)
            fig.set_figheight(8)
            fig.set_figwidth(8)
            axs[0,0].plot(df_fitted['flow_rate'],df_fitted['head'])
            axs[0,0].scatter(df_test['flow_rate'],df_test['head'])
            axs[0,0].set_xlim([0, max_flow])
            axs[0,0].set_ylim([0, max_head])
            axs[0,0].set_xlabel("Flow rate (m3/hr)")
            axs[0,0].set_ylabel("Head (m)")


            axs[0,1].plot(df_fitted['flow_rate'],df_fitted['fitted_eff'])
            axs[0,1].scatter(df_test['flow_rate'],df_test['efficiency'])
            axs[0,1].set_xlabel("Flow rate (m3/hr)")
            axs[0,1].set_ylabel("Efficiency (%)")

            axs[1,0].plot(df['flow_rate'],df['power'])
            axs[1,0].scatter(df_test['flow_rate'],df_test['power'])
            axs[1,0].set_xlabel("Flow rate (m3/hr)")
            axs[1,0].set_ylabel("Power (Kw)")

            return st.pyplot(fig)
def convert_data(df):
     csv = df.to_csv(index=False).encode('utf-8')
     return csv

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
    #Case 1.1: no pump FAT csv file and using form
    if s1 == 'No':
        keys = ['fat_input',"add form",]
        
        # Calling the form function for the user to input his own FAT data
        add_form(st.session_state.df,keys,5)
        df_uploaded = pd.DataFrame(st.session_state.df)
        #Displaying the observations table
        st.dataframe(df_uploaded)

        #power_factor_input using form
        st.subheader("Add Pump power factors data")
        if "df_test" not in st.session_state:
            st.session_state.df_factor = pd.DataFrame(columns=["load%","current","motor_efficiency","power_factor"])

        keys = ['fat_power_input',"add form power factor"]
        # Calling the form function for the user to input his own data
        add_form(st.session_state.df_factor,keys, 4)
        
        df_uploaded_power= pd.DataFrame(st.session_state.df_factor)
        # Making sure that the user input for the pump motor data form includes a 100% load point
        if 100 not in df_uploaded_power['load%'].values and st.session_state.df_factor.shape[0] == 3:
            rated_current = st.number_input('Insert rated current')
            pf = st.number_input('Insert rated current power factor')
            moto_eff = st.number_input('Insert rated current motor efficiency')
            
            df_uploaded_power.loc[len(df_uploaded_power.index)+1] = [100,rated_current,moto_eff,pf] 
            st.info(f'{len(df_uploaded_power.index)}')  
            #Displaying the observations table for pump motor data
            st.dataframe(df_uploaded_power) 
        else: st.dataframe(df_uploaded_power)  #Displaying the observations table
        
        
        
        if st.button("Plot fitted Pump curves"):
            try:
                #obatining the required parameters for plotting fitted curves along with obatined FAT data
                #by calling the df_fit() function
                df_fitted, max_head, max_flow_rate,a_h, b_h, c_h = df_fit(df_uploaded)
                #Plot recorded data using the plot_fitted_curves() function
                plot_fitted_curves(df_fitted,df_uploaded,max_flow_rate, max_head)
            except (ValueError, TypeError): st.write('Please Check your dataset')

    else: #Case 1.2: FAT Table from CSV file
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
            cols = ["flow_rate","head","power","efficiency"]
            if uploaded_file:
                df_uploaded=pd.read_csv(uploaded_file)
                try:
                    #Checking if uploaded columns matches the specified columns' names
                    if set(list(df_uploaded.columns)) == set(cols):
                    
                        #Sorting uploaded FAT data by flow rate
                        df_uploaded=df_uploaded.sort_values('flow_rate')
                        #Displaying the observations table
                        st.dataframe(df_uploaded)
                        df_fitted, max_head, max_flow_rate,a_h, b_h, c_h = df_fit(df_uploaded)
                    else: st.write('Please Check your FAT column names')    
                except TypeError: st.write('Please Check your FAT dataset')
            
            
            

           
        except ValueError:
            st.write('Please Choose your pump curve file')

        # power factor data calculations
        try:
            uploaded_file_power = st.file_uploader('Choose a file', key = 2)
            cols = ["load%","current","motor_efficiency","power_factor"]
            if uploaded_file_power:
                df_uploaded_power=pd.read_csv(uploaded_file_power)
                try:
                    #Checking if uploaded columns matches the specified columns' names
                    if set(list(df_uploaded_power.columns)) == set(cols):
                        #Sorting uploaded power data by load percentage
                        df_uploaded_power=df_uploaded_power.sort_values("load%")
                        # Making sure that the user uploaded table for the pump motor data form includes a 100% load point
                        if 100 not in df_uploaded_power["load%"].values:
                            rated_current = st.number_input('Insert rated current')
                            pf = st.number_input('Insert rated current power factor')
                            moto_eff = st.number_input('Insert rated current motor efficiency')
                            df_uploaded_power.loc[len(df_uploaded_power.index)] = [100, rated_current,pf,moto_eff]
                            #Displaying the observations table for pump motor data
                            st.dataframe(df_uploaded_power)
                        else: st.dataframe(df_uploaded_power) #Displaying the observations table for pump motor data
                        
                    else: st.write('Please Check your motor data columns names')    
                except TypeError: st.write('Please Check your dataset')
        except ValueError:
            st.write('Please Choose your factor power file')

        #Plot obtained data from file
        if st.button("Plot fitted Pump curves"):
            try:
                #Plot recorded data using the plot_fitted_curves() function
                plot_fitted_curves(df_fitted,df_uploaded,max_flow_rate, max_head)
            except (ValueError, TypeError, UnboundLocalError): st.write('Please Check your dataset(s)')
    
    
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
    #Case 2.1: Test Run data gathering no CSV file and using form
    if s1 == 'No': 
        keys = ['test_input',"addform_test"]
        # Calling the form function for the user to input his own test run data
        add_form(st.session_state.df_test,keys,5)
        st.dataframe(st.session_state.df_test)

        if st.button("Reveal Calculations Table", key = 'calculations_table'):
            try:
                df_test_uploaded = pd.DataFrame(st.session_state.df_test).copy().sort_values('flow_rate')
                #Updating the uploaded test run data with the test run calculations
                df_test_uploaded = test_calculations(df_test_uploaded,df_uploaded_power,df_uploaded) 
                st.dataframe(df_test_uploaded)
                #Calling plot_charts() function which displays pump data against test run data
                plot_charts(df_uploaded,df_test_uploaded,df_fitted,max_head,max_flow_rate)
                st.download_button("Click to download your calculations table!", convert_data(df_test_uploaded),"calculations.csv","text/csv", key = "download1")
                
            except (ValueError, TypeError): st.write('Please Check your dataset')
            
            
    else: #Case 2.2: Test Run data gathering no CSV file and using form
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
            #Code to upload gathered test run data from CSV file
            uploaded_test_file = st.file_uploader('Choose a file', key = "test_run_data")
            cols = ["flow_rate", "p_suction","p_discharge", "current", "voltage","specific_gravity"]
            if uploaded_test_file:
                df_test_uploaded=pd.read_csv(uploaded_test_file)
                #Checking if uploaded test run data columns matches the specified columns' names
                if set(list(df_test_uploaded.columns)) == set(cols):
                    df_test_uploaded=df_test_uploaded.sort_values('flow_rate')
                    #Display uploaded test run data
                    st.dataframe(df_test_uploaded.loc[:,["flow_rate", 
                                                    "p_suction",
                                                    "p_discharge", 
                                                    "current",
                                                    "voltage",
                                                    "specific_gravity"]])
                    try:
                        # Updating the uploaded test run data with the test run calculations
                        df_test_uploaded = test_calculations(df_test_uploaded,df_uploaded_power,df_uploaded)    
                    except TypeError: st.write('Please Check your FAT and power factor datasets')
                    
                    

                    if st.button("Reveal Calculations Table", key = 'calculations_table'):
                        try:
                            #Displaying uploaded test run data
                            st.dataframe(df_test_uploaded)
                            st.download_button("Click to download your calculations table!", convert_data(df_test_uploaded),"calculations.csv","text/csv", key = "download2")
                        except (ValueError, TypeError, KeyError): st.write('Please Check your uploaded Test run dataset')
                            
                        
                    if st.button("Plot Pump curves", key = 'test_fat_curve'):
                        try:
                            #Calling plot_charts() function which displays pump data against test run data
                            plot_charts(df_uploaded,df_test_uploaded,df_fitted,max_head,max_flow_rate)
                        except UnboundLocalError: st.write('Please Check your FAT and power factor datasets')
                    
                else: st.write('Please Check that your columns names are as specified')
            
            
                
        except ValueError:
            st.write('Please Choose your test run data file')
        
        


if __name__ == '__main__':
    main()