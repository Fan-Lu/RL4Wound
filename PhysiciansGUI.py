#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 12:48:43 2023

@author: Manasa Kesapragada
"""

# Run this app with `python PhysiciansGUI.py` and
# visit http://127.0.0.1:8055/ in your web browser.

from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
from dash import Input, Output, State, html, no_update 
from dash_iconify import DashIconify
import pdb
import matplotlib.pyplot as plt
import numpy as np
import plotly.io as pio
import datetime
from skimage import io
import shutil
import os


# Global name for csv file
# csv_file_name = "./data_save/exp_25/comb_wound_1.csv"


wound_num = input('Please select wound number, for example: 1'
                  '\n Wound #: ')
wound_num = int(wound_num.replace(' ', ''))
print('Wound number is set to {} !!!'.format(wound_num))

csv_file_name = "./data_save/exp_23/comb_wound_{}.csv".format(wound_num)


dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"

app = Dash(__name__, external_stylesheets=[dbc.themes.ZEPHYR, dbc_css, dbc.themes.BOOTSTRAP])

app.title = "Interface for Physicians"

def reset_time_stampe(df):
    lnu = len(df)
    day_change_idx = []
    df['time'] = df['time(s)'] - df['time(s)'].iloc[0]

    for idx in range(1, lnu):
        if df.loc[idx, 'time'] < df.loc[idx - 1, 'time']:
            day_change_idx.append(idx)
    for day_idx in day_change_idx:
        for idx in range(day_idx, lnu):
            df.loc[idx, 'time'] = df.loc[idx, 'time'] + df.loc[day_idx - 1, 'time']
    return df

df_combined = pd.read_csv(csv_file_name)
df_combined_forDosage = pd.read_csv(csv_file_name)
# TODO: Added by Fan Lu
df_combined = reset_time_stampe(df_combined)
df_combined_forDosage = reset_time_stampe(df_combined_forDosage)

selected_columns = ['Hemostasis', 'Inflammation', 'Proliferation', 'Maturation']

datetime_str = df_combined['Image'].str.extract(r'(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})')
df_combined['conv_time'] = pd.to_datetime(datetime_str[0], format='%Y-%m-%d-%H-%M-%S', errors='coerce')

pred_time = px.line(df_combined, x="conv_time", y=selected_columns)

# df_m1m2 = pd.read_csv("out_wound_6_lowEF_2023-03-03-10-07.csv")


df_main = pd.read_csv('woundprogress.csv')

columns_woundprogression = ['wound_progress_target', 'wound_progress_target_+30%',
       'wound_progress_target_-30%', 'wound_progress_noctr']
      # 'wound_progress_DRLctr']#, 'wound_progress_healnet']


#df['Time Processed'] = pd.to_datetime(df['Time Processed'])

# Calculate the +30% and -30% values
df_main['Upper Line'] = df_main['wound_progress_target_+30%'] 
df_main['Lower Line'] = df_main['wound_progress_target_-30%'] 

##For Drug concentration vs time
F = 96485.3321   # Faraday constant -- coulombs per mole (C/mol) = (Amp*sec)/mol
eta = 0.2        # Pump efficiency
g_per_mol = 309.33  # Molecular weight of flx
#charge = current_in_amperes * time_in_seconds  # Calculate charge in Coulombs
#dose = eta * charge * g_per_mol / (F * 1e3)

def cum_drug_conc(df, name):
    doses = np.zeros((8, len(df) - 1))
    for ch in (1, 3, 5, 7):
        currents = df[f'{name}_ch_{ch}'].iloc[1:].values
        df['time'] = df['time(s)'] - df['time(s)'].iloc[0]

        times = df['time'].iloc[1:].values - df['time'].iloc[:-1].values
        charges = currents * times
        doses[ch-1] = eta * charges * g_per_mol / (F * 1e3)
    cum_doses = np.cumsum(doses, axis=1)
    return cum_doses


##Live plot --Current
df_combined[df_combined.columns[9:17]]  = df_combined[df_combined.columns[9:17]] * 5e6
df_combined['time'] = df_combined['time(s)'] - df_combined['time(s)'].iloc[0]

df_combined['time(h)'] = df_combined['time'] / 3600
current_time = px.line(df_combined, x="time(h)", y=df_combined.columns[9:17])
current_time.update_layout(
   yaxis_title="EF Strength (mV/mm)",
   xaxis_title="Time (hours)",
   legend_title="",
   legend=dict(
   orientation="h",
   yanchor="bottom",
   y=1.02,
   xanchor="right",
   x=1
   )
   )
current_time.update_yaxes(showticksuffix="none")

labels = ['Channel 1', 'Channel 2','Channel 3','Channel 4','Channel 5','Channel 6','Channel 7','Channel 8']
# Update the traces with custom labels and colors
for i, trace in enumerate(current_time.data):
    trace.name = labels[i]  # i + 1 to skip the 'Time' column



###Live plot --Dosage
#dose_df = df_combined[df_combined.columns[9:17]].apply(drug_conc, args=(df_combined['time(s)'],))

cum_dosage = cum_drug_conc(df_combined_forDosage, 'dc')
cum_dosageT = cum_dosage.T

# Creating a DataFrame from the transposed array
dose_df = pd.DataFrame(cum_dosageT)
dose_df.insert(0, 'time(s)', df_combined_forDosage['time(s)'])
dose_df['time'] = dose_df['time(s)'] - dose_df['time(s)'].iloc[0]
dose_df['time(h)'] = dose_df['time'] / 3600
dosage_time = px.line(dose_df, x=dose_df['time(h)'], y=dose_df.columns[1:9])
dosage_time.update_layout(
    yaxis_title="Dose (mg)",
    xaxis_title="Time (hours)",
    legend_title="",
    legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
)
)




labels = ['Channel 1', 'Channel 2','Channel 3','Channel 4','Channel 5','Channel 6','Channel 7','Channel 8']
# Update the traces with custom labels and colors
for i, trace in enumerate(dosage_time.data):
    trace.name = labels[i]  # i + 1 to skip the 'Time' column

##Live plot --Current
controller_odes = px.line(df_main, x="Time", y="wound_progress_target")
controller_odes.update_layout(
       yaxis_title="Wound progression towards closure",
       xaxis_title="Days from the wound onset",
       legend_title="Model",
       legend=dict(
       orientation="h",
       yanchor="bottom",
       y=1.02,
       xanchor="right",
       x=1
       )
       )

# Add area fill for each line
for i, trace in enumerate(controller_odes.data):
    if i > 0:
        trace.fill = 'tonexty'

##Current time wound stage prob bar plot
colors = ['blue','yellow', 'green', 'orange']

pred_stage = px.bar(x=df_combined[selected_columns].iloc[-1].index, y = df_combined[selected_columns].iloc[-1].values, color=colors)
pred_stage.update_layout(
    showlegend=False,
    xaxis_title="Wound stages",
    yaxis_title="Probability",
)

image_link = df_combined['Image'].iloc[-1]

tab1_content = dbc.Card([
    dbc.CardHeader("Wound Image"),
    dbc.CardBody([
        html.P("Wound Image",className="card-text"),
        dbc.CardImg(id="wound-image",src=""),
        dcc.Interval(id = "update-woundimage", interval = 10000, n_intervals = 0)
    ])
   
])


tab2_content = dbc.Card([
    dbc.CardBody([
        dcc.Graph(
            id='dosage_day',
            figure=dosage_time),
        dcc.Interval(id = "update-doseVStime", interval = 10000, n_intervals = 0)
    ])
], className="p-4 border border-0 ")

tab3_content = dbc.Card([
    dbc.CardBody([
            dcc.Graph(
                id='current-time-closedloop',
                figure=current_time),
            dcc.Interval(id = "update-efVStime", interval = 10000, n_intervals = 0)
    ])
], className="p-4 border border-0 ")


tab4_content = dbc.Card([
                dbc.CardBody([
                    dcc.Graph(
                        id='wound-stage-pred-time'),
                    dcc.Interval(
                        id='wound-stage-interval',
                        interval=10000,
                        n_intervals=0
                        )
                    ])
], className="p-4 border border-0 ")

## Masked screen
# Read the contents of the local text file

## Stat icon on the button
go_to_details_icon = DashIconify(icon="carbon:time-plot", style = {'marginRight':5 }) #ic:baseline-timeline
go_back = DashIconify(icon="material-symbols:arrow-back", style = {'marginRight':5 }) #material-symbols:home-health
control_panel_icon = DashIconify(icon="material-symbols:left-panel-open", style = {'marginRight':5 }) 


##The initial popup screen displaying the wound state and the go to details button
maskedScreen = html.Div([        
         dbc.Card([
                dbc.CardBody([
                    html.H2("Wound State", className="card-title", id = 'woundState', style={'width': '50%','display': 'inline-block','text-align': 'center'}),
                    html.P(id="wound_state", className="card-text",style={'width': '50%','display': 'inline-block','text-align': 'center', 'font-size': '2em'}),
                    dcc.Interval(id = "update-state", interval = 100000, n_intervals = 0)
                     ])
                ], color="dark", inverse=True, className='card border-primary mb-3'),
        
    ], id='popUpDiv')



app.layout = dbc.Container([
    html.Div(maskedScreen,  className="w-50 p-5", style={'position': 'absolute','top': '50%','left': '50%','transform': 'translate(-50%, -50%)'}),
    html.Div(dbc.Button('Toggle', className = "me-1", id = "goStatBtn"),style ={'text-align': 'right', 'margin-top': '1%'}),
    html.Div([
        dbc.Row([
           dbc.Col([
               dbc.Card([
                dbc.CardHeader("Manual Control Panel: Adjust Drug Concentration"),
                dbc.CardBody([
                    html.P("Adjust proposed drug concentrations at next treatment time", className="card-text"),
                    dbc.Col([
                        dbc.Input(id="drug-concentration-input", type="number", min=0, max=0.025, step=0.001, placeholder="Enter the Drug Conc [0-0.1 mg]..."),
                    ],style={'width': '50%','display': 'inline-block'}),
                    dbc.Col([
                        dbc.Button("Set Drug Concentration", id="set-drug-concentration-btn", color="info"),
                    ],style={'width': '50%','display': 'inline-block'}),
                    
                    ## This is seen after the button is clicked
                    dbc.Col([
                        html.Div(id='dc-output')
                    ])

                    ])
                ]),
                dbc.Card([
                    dbc.CardHeader("Adjust Electric Field"),
                    dbc.CardBody([
                        
                        html.P("Adjust proposed EF Strength at next treatment time", className="card-text"),
                        dbc.Col([
                            dbc.Input(id="ef-strength-input", type="number", min=0, max=50, step=1, placeholder="Enter the EF Strength [0-35 mV/mm]..."),
                        ],style={'width': '50%','display': 'inline-block'}),
                        dbc.Col([
                            dbc.Button("Set Electric Field Strength", id="set-efstrength-btn", color="info"),
                         ],style={'width': '50%','display': 'inline-block'}),
                        
                        ## This is seen after the button is clicked
                         dbc.Col([
                             html.Div(id='ef-output')
                         ])
                        ])
                    ]),
                dbc.Card([
                    dbc.CardHeader("Emergency Shut-off"),
                    dbc.CardBody([
                        dbc.Col([
                            html.P("Set Current to 0!  ", className="card-text"),
                        ],style={'width': '14%','display': 'inline-block'}),
                        dbc.Col([
                            dbc.Button("Emergency Shut-off", id="shutoff-btn", color="danger",  className="me-1"),
                         ],style={'display': 'inline-block'}),
                       
                       ## This is seen after the button is clicked
                        dbc.Col([
                            html.Div(id='shutoff-output')
                        ])
                        ])
                    ]),
                ],width=8)
        
        ],className = "mb-2 mt-2"),

        dbc.Row([
          dbc.Col([
            dbc.Card([
                dbc.CardHeader("Wound stage probability at current time"),
                dbc.CardBody([
                    dcc.Graph(
                        id='wound-stage-bar',
                        figure=pred_stage), 
                    dcc.Interval(id = "update-wound-prob-bar", interval = 10000, n_intervals = 0)
                    ])
                ]),
            ],width=4),

            dbc.Col([
                tab1_content
              ],width=4)
     
      
        ],className = "mb-2 mt-2"),
     html.Div(dbc.Button([go_to_details_icon,'Go to Detailed View'], className = "me-1", id = "goDetails")),

    ],id = 'mainDiv'),
    
     html.Div([
         dbc.Row([
           dbc.Col([
            dbc.Card([
                dbc.CardHeader("Tracking Wound Progression"),
                 dbc.CardBody([
                    dcc.Graph(
                        id='healerAI',
                        figure= controller_odes),
                    dcc.Interval(id = "update-healerAI", interval = 10000, n_intervals = 0)
                ]),
            ]),
           ],className = "mb-2 mt-2",width=8),
          dbc.Row([
            dbc.Col([
            dbc.Tabs([
                dbc.Tab(tab2_content, label="Drug Concentration vs time"),
                dbc.Tab(tab3_content, label="EF Strength vs time"),
                dbc.Tab(tab4_content, label="Wound Stage Predictions")
            ])  
        ], width=8),
       ],className = "mb-2 mt-2"),  
    ],id = 'detailDiv',style={'display': 'none'}) ])
       
    
],fluid = True)


##Call back for navigate to control panel button
@app.callback(
    Output('popUpDiv','style'), 
    Output('mainDiv','style'),
    Input('goStatBtn', 'n_clicks')
    )
def hide_and_fade(n_clicks):
    if n_clicks is None:
        return {'display': 'block'}, {'opacity': 0.08, 'background-color': 'white'} #, {'display': 'none'}
    elif n_clicks % 2 == 0:
        return {'display': 'block'}, {'opacity': 0.08, 'background-color': 'white'} #, {'display': 'none'}
    else:
        return {'display': 'none'}, {'display': 'block'} #, {'display': 'block'}


##Call back to change the label of the button
@app.callback(
    Output("goStatBtn", "children"),
    Input("goStatBtn", "n_clicks")
)
def update_button_label(n_clicks):
    if n_clicks is None:
        return [control_panel_icon,"Manual Control Panel"]
    elif n_clicks % 2 == 0:
        return [control_panel_icon,"Manual Control Panel"]
    else:
        return [go_back,"Navigate to the home screen"]

## Call back to update wound state in real time
@app.callback(
    Output("wound_state", "children"),
    Input("update-state", "n_intervals")
    )
def update_wound_state(n):
    ## Read the wound state from the file
    with open('state.txt', 'r') as file:
        data = file.read().strip()
    
    return data

##Call back to go to the details screen
@app.callback(
    Output('detailDiv','style'), 
    Output("goDetails", "children"),
    Input('goDetails', 'n_clicks')
    )
def hide_detail_screen(n_clicks):
    if n_clicks is None:
        return  {'display': 'none'}, ["Go to Detailed View"]
    elif n_clicks % 2 == 0:
        return  {'display': 'none'}, ["Go to Detailed View"]
    else:
        return {'display': 'block'}, ["Hide Detailed View"]


# Callback to save drug concentration to CSV
@app.callback(
    Output("dc-output", "children"),
    Output('drug-concentration-input', 'value'),
    Input("set-drug-concentration-btn", "n_clicks"), 
    State("drug-concentration-input", "value")
)
def save_to_dccsv(n_clicks, drug_concentration_value):
    if n_clicks is not None:
        # Save the drug concentration to a CSV file
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        df = pd.DataFrame({"Datetime": [current_datetime], "Drug Concentration": [drug_concentration_value]})
        df.to_csv("drug_concentration_data.csv", index=False)

        # Returning a message to update the "dc-output" div
        message = f"Drug Concentration {drug_concentration_value} mg saved to csv."

        # Clearing the input value after clicking the button
        drug_concentration_value = None

        return message, drug_concentration_value
    else:
        return no_update, no_update


# Callback to save ef value to CSV
@app.callback(
    Output("ef-output", "children"),
    Output('ef-strength-input', 'value'),
    Input("set-efstrength-btn", "n_clicks"), 
    State("ef-strength-input", "value")
)
def save_to_efcsv(n_clicks, ef_strength_value):
    if n_clicks is not None:
        # Save the ef strength to a CSV file
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        df = pd.DataFrame({"Datetime": [current_datetime],"EF Strength": [ef_strength_value]})
        df.to_csv("ef_strength_data.csv", index=False)

        # Returning a message to update the "dc-output" div
        message = f"EF Strength {ef_strength_value} mV/mm saved to csv."

        # Clearing the input value after clicking the button
        ef_strength_value = None

        return message, ef_strength_value
    else:
        return no_update, no_update

# Callback to save shut off value to CSV
@app.callback(
    Output("shutoff-output", "children"),
    Input("shutoff-btn", "n_clicks")
)
def save_to_shutOffcsv(n_clicks):
    current_value = 0
    if n_clicks is not None:
        # Save the shutoff current to a CSV file
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        df = pd.DataFrame({"Datetime": [current_datetime],"Current": [current_value]})
        df.to_csv("shutoff.csv", index=False)

        # Returning a message to update the "dc-output" div
        message = f"Current {current_value} saved to csv."

        return message






##Call back for Wound Healer ODES and Controller vs time plot
## Accelerated vs Delayed
@app.callback(
    Output('healerAI','figure'),
    [Input('update-healerAI', 'n_intervals')]  
    )    
def updatehealerAI_plot(n_intervals):
    df_main = pd.read_csv('woundprogress.csv')
    for column in columns_woundprogression:
        df_main[column].replace('', np.nan, inplace = True)
        
     # Read the desired column 'wound_progress_DRLctr' from another CSV file
    df_online = pd.read_csv(csv_file_name, usecols=['wound_progress_DRLctr', 'time(s)'])
    # TODO: Fan Lu: reset timestamp using reset_time_stamp function will cause wound stage not displaying
    # df_online = reset_time_stampe(df_online)

    df_combined['time'] = df_combined['time(s)'] - df_combined['time(s)'].iloc[0]

    df_online['time(d)'] = df_combined['time'] / 86400 

    df_main = df_main.sort_values(by='Time')

    controller_odes = px.scatter(df_main, x="Time", y=columns_woundprogression)
    controller_odes.add_scatter(x=df_online['time(d)'],  y=df_online['wound_progress_DRLctr'], mode='markers', name='wound_progress_DRLctr')

    controller_odes.update_layout(
       yaxis_title="Wound progression towards closure",
       xaxis_title="Days from the wound onset",
        legend_title="",
          legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1,
            xanchor="right",
            x=1
        ),
        annotations=[
         dict(
             x=14,  # X coordinate of the annotation
             y=0.3,  # Y coordinate of the annotation
             text='Delayed healing',  # The text to be displayed
             showarrow=False,  # Hide the arrow
             font=dict(
                 size=20,
                 color='#8B0000' #lightgoldenrodyellow'
             )
         ),
         dict(
             x=3,  # X coordinate of the annotation
             y=0.8,  # Y coordinate of the annotation
             text='Accelerated healing',  # The text to be displayed
             showarrow=False,  # Hide the arrow
             font=dict(
                 size=20,
                 color='Black'
             )
         )
         ]
       )
    
     # Color the area above and below the line differently
    for i, trace in enumerate(controller_odes.data):
         if i == 3:
             trace.fill = 'tozeroy'
       

    # Customize the labels for each trace
    custom_labels = {
        'wound_progress_target': 'Target',
        'wound_progress_target_+30%': '+/-30% Target',
        'wound_progress_target_-30%': '',
        'wound_progress_noctr': 'w/o Treatment',
        'wound_progress_DRLctr': 'w/ DRL treatment'
    }

    # Update the traces with custom labels and colors
    for i, trace in enumerate(controller_odes.data):
        trace.name = custom_labels[df_main.columns[i + 2]]  # i + 1 to skip the 'Time' column

    # Define custom colors for each trace
    custom_colors = ['#636EFA', 'rgba(0, 0, 0, 0.4)', 'rgba(0, 0, 0, 0.4)', '#AB63FA', '#00CC96', '#CC5800']

    # Update trace colors
    for i, trace in enumerate(controller_odes.data):
        trace.marker.color = custom_colors[i]
    
    # Update the traces with custom labels and colors
    for i, trace in enumerate(controller_odes.data):
        label = custom_labels.get(df_main.columns[i + 2], '')  # Get the custom label or an empty string
        trace.name = label  # Set the name to the custom label
        trace.showlegend = label != ''  # Hide legend if the label is an empty string
    
    #controller_odes.update_traces(mode='lines+markers')#, marker=dict(symbol='square', size=10))
    marker_modes = ['lines', 'markers', 'markers', 'lines', 'lines', 'markers+lines']
    marker_sizes = [None, 4, 4, None, None, 10]  # Adjust the sizes as needed
    # marker_symbols = [None, 'cross', 'x', None, None, 'triangle-up']
    marker_symbols = [None, 'cross', 'x', None, None, 'triangle-up']


    for i, trace in enumerate(controller_odes.data):
        if i < 4:
            trace.mode = marker_modes[i]

            if 'markers' in marker_modes[i]:
                trace.marker.size = marker_sizes[i]
                trace.marker.symbol = marker_symbols[i]


    return controller_odes



##Call back for Dosage vs time plot
@app.callback(
    Output('wound-stage-bar','figure'),
    [Input('update-wound-prob-bar', 'n_intervals')]  
    )    
def update_woundBar(n_intervals):
    df_combined = pd.read_csv(csv_file_name)
    # TODO: Fan Lu: reset timestamp using reset_time_stamp function will cause wound stage not displaying
    # df_combined = reset_time_stampe(df_combined)
    pred_stage = px.bar(x=df_combined[selected_columns].iloc[-1].index, y = df_combined[selected_columns].iloc[-1].values, color=colors)
    pred_stage.update_layout(
        showlegend=False,
        xaxis_title="Wound stages",
        yaxis_title="Probability",
    )
    return pred_stage


##Call back for updating wound image
@app.callback(
    Output('wound-image', 'src'),
    Input('update-woundimage', 'n_intervals')
)

def update_image(n_intervals):
    df_combined = pd.read_csv(csv_file_name)
    df_combined = reset_time_stampe(df_combined)
    image_link = df_combined['Image'].iloc[-1]
    # Full path to the source image
    src_image_path = image_link
    # File name of the source image
    image_file = os.path.basename(image_link)
    # Full path to the destination image in the assets folder
    dest_image_path = os.path.join("assets/", image_file)
    # Copy the image file to the assets folder
    shutil.copy(src_image_path, dest_image_path)
    # Relative path for the image to be displayed in the app
    relative_image_path = f'/assets/{image_file}'
    return relative_image_path


##Call back for Dosage vs time plot
@app.callback(
    Output('dosage_day','figure'),
    [Input('update-doseVStime', 'n_intervals')]  
    )    
def update_DosagevsT_plot(n_intervals):
    df_combined_forDosage = pd.read_csv(csv_file_name)

    # TODO: Added by Fan Lu
    df_combined_forDosage = reset_time_stampe(df_combined_forDosage)

    cum_dosage = cum_drug_conc(df_combined_forDosage, 'dc')
    cum_dosageT = cum_dosage.T

    # Creating a DataFrame from the transposed array
    dose_df = pd.DataFrame(cum_dosageT)
    dose_df.insert(0, 'time(s)', df_combined['time(s)'])
    dose_df['time'] = dose_df['time(s)'] - dose_df['time(s)'].iloc[0]
    
    dose_df['time(h)'] = dose_df['time'] / 3600
    dosage_time = px.line(dose_df, x=dose_df['time(h)'], y=dose_df.columns[1:9])
    dosage_time.update_layout(
        yaxis_title="Dose (mg)",
        xaxis_title="Time (hours)",
        legend_title="",
        yaxis_range=[0, 0.025],
        legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
    )
    #dosage_time.update_yaxes(showticksuffix="none")
    #dosage_time.update_yaxes(tickformat=".5f", tickprefix=" ", ticksuffix=" ")
    dosage_time.update_yaxes(tickformat=".2r")


    labels = ['Channel 1', 'Channel 2','Channel 3','Channel 4','Channel 5','Channel 6','Channel 7','Channel 8']
    ##Update the traces with custom labels and colors
    for i, trace in enumerate(dosage_time.data):
        trace.name = labels[i]  # i + 1 to skip the 'Time' column

    return dosage_time



##Call back for EF vs time plot
@app.callback(
    Output('current-time-closedloop','figure'),
    [Input('update-efVStime', 'n_intervals')]  
    )    
def update_EFvsT_plot(n_intervals):
    df_combined = pd.read_csv(csv_file_name)

    # TODO: Added by Fan Lu
    df_combined = reset_time_stampe(df_combined)

    df_combined[df_combined.columns[9:17]]  = df_combined[df_combined.columns[9:17]] * 5e6
    df_combined['time'] = df_combined['time(s)'] - df_combined['time(s)'].iloc[0]

    df_combined['time(h)'] = df_combined['time'] / 3600
    current_time = px.line(df_combined, x="time(h)", y=df_combined.columns[9:17])
    current_time.update_layout(
       yaxis_title="EF Strength (mV/mm)",
       xaxis_title="Time (hours)",
       legend_title="",
       legend=dict(
       orientation="h",
       yanchor="bottom",
       y=1.02,
       xanchor="right",
       x=1
       )
       )
    current_time.update_yaxes(showticksuffix="none")
    
    labels = ['Channel 1', 'Channel 2','Channel 3','Channel 4','Channel 5','Channel 6','Channel 7','Channel 8']
    # Update the traces with custom labels and colors
    for i, trace in enumerate(current_time.data):
        trace.name = labels[i]  # i + 1 to skip the 'Time' column

    return current_time


##Call back for wound stage prediction plot --This was using previous healnet probabilities table
@app.callback(
    Output('wound-stage-pred-time','figure'),
    [Input('wound-stage-interval', 'n_intervals')]  
    )    
def update_woundstage_plot(n_intervals):
    df_combined = pd.read_csv(csv_file_name)
    # TODO: Fan Lu: reset timestamp using reset_time_stamp function will cause wound stage not displaying
    # df_combined = reset_time_stampe(df_combined)
    datetime_str = df_combined['Image'].str.extract(r'(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})')
    df_combined['conv_time'] = (df_combined['time(s)'] - df_combined['time(s)'].iloc[0]) / (86400.0)
    # df_combined['conv_time'] = pd.to_datetime(datetime_str[0], format='%Y-%m-%d-%H-%M-%S', errors='coerce')
    pred_time = px.line(df_combined, x="conv_time", y=selected_columns)
    pred_time.update_layout(
        yaxis_title="Wound stage probability",
        xaxis_title="Hours from the wound onset",
        legend_title="Wound stages",
       # xaxis=dict(fixedrange=True),  # Disable zoom on the x-axis
       # yaxis=dict(scaleanchor="x"),   # Allow zoom on the y-axis
        legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
        )
    )
    return pred_time



if __name__ == '__main__':
    app.run_server(port=8057,debug=False, dev_tools_hot_reload=False)