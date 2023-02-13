# Centrifugal_pump_performance_evaluation
This tool was developed for the APRCO engineers to evaluate the performance of their centrifugal pumps in their respective units.

You need 3 files/entries to use this tool:
1. FAT (factory acceptance test) results of you pump (Q, H, Power, Efficiency%)
2. your pump motor datasheet data (Load%, Current, motor efficiency and power factors)
3. performance test data (Q, Psuc.,Pdisch., Current, Voltage and Specific gravity)[1] 

The App will fit your FAT Q-H and Q-Efficiency data into two curves using the Equation described as follows [2]:
H = a*Q + b*Q^2 + C
E = a*Q + b*Q^2 + C

Then performs your performance test calculations and calculate %Error in your readings (Compare it with API 610 tolerance table) 
Additionally, the app will display your test results in graphs and you can also download your calculations table!

[1] P.S: you can make an estimate of your fluid Specific gravity using the shut-off head in your characteristic curve where dP (Pa) = rho*9.81*H (m)
usually makes +-8% Error in S.G.
[2] Working Guide to Pump and Pumping Stations: Calculations and Simulations, E. Shashi Menon - Pages (25-27)

Note 1: These calculations neglects the effects of elevations differences between your to pressure gauges.
Note 2: add form code credits: Mr. Shawn_Pereira reply in stackflow
link: https://discuss.streamlit.io/t/how-to-add-records-to-a-dataframe-using-python-and-streamlit/19164/5
