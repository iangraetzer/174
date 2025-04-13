import streamlit as st

# st.cache_data.clear() # debug
import time
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from urllib.request import urlopen
from urllib.error import HTTPError
import json
import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go

from streamlit_dynamic_filters import DynamicFilters

# ian's image packages
from PIL import Image
from io import BytesIO
from streamlit.components.v1 import html
import requests

import os
import pathlib

# will need `pyarrow`if using parquet files (recommended)




#from ipyvizzu import Chart, Data, Config, Style, DisplayTarget

st.set_page_config(page_title="CSE 6242 App", layout="wide")

# os.chdir('/Users/oscar.martinez/Library/CloudStorage/OneDrive-CoxAutomotive/Documents/GT/CSE6242/project/my_app')
current_dir = pathlib.Path(__file__).parent.resolve()
lap_tire_df_no_outliers_path = os.path.join(current_dir, 'lap_tire_df_no_outliers.parquet')
driver_bio_2_path = os.path.join(current_dir, 'driver_bio_2.parquet')

@st.cache_data(ttl=28800)
def get_info_data():
    """
    function to pull info df
    """
    # query the API
    driver_api = 'https://api.openf1.org/v1/drivers' # driver data
    sess_api = 'https://api.openf1.org/v1/sessions' # session data
    meet_api = 'https://api.openf1.org/v1/meetings' # meeting/location data

    ok = False
    while not ok:   
        try:
            # bring in driver data
            response = urlopen(driver_api)
            data = json.loads(response.read().decode('utf-8'))
            drv_df = pd.DataFrame(data)

            # bring in session data
            response = urlopen(sess_api)
            data = json.loads(response.read().decode('utf-8'))
            ses_df = pd.DataFrame(data)

            # bring in meeting data (to use Ian's work)
            response = urlopen(meet_api)
            data = json.loads(response.read().decode('utf-8'))
            met_df = pd.DataFrame(data)    
            
            ok = True
        except HTTPError as e:
            time.sleep(1)           

    f_df = drv_df.merge(ses_df, how='left', on=['session_key'])
    f_df = f_df.merge(met_df[['meeting_name','circuit_key','year']], how='left', on=['circuit_key','year'])
    f_df = f_df.loc[f_df['session_type']=='Race',:].copy() # only races have the necessary info
    
    return f_df

# -------------------- code to bring in the data from the API and create a df for the plots -------------------- 
# -------------------- should convert to cacheable function in prod --------------------
@st.cache_data(ttl=28800)
def pull_f1_data(ses_num, drv_num):
    """
    function to pull f1 data
    
    + ses_num = session number
    + drv_num = driver_num
    """
    # query the API
    brk_rec = f'https://api.openf1.org/v1/car_data?session_key={ses_num}&driver_number={drv_num}'
    loc_rec = f'https://api.openf1.org/v1/location?session_key={ses_num}&driver_number={drv_num}'
    lap_rec = f'https://api.openf1.org/v1/laps?session_key={ses_num}&driver_number={drv_num}'
    pos_rec = f'https://api.openf1.org/v1/position?session_key={ses_num}&driver_number={drv_num}'

    # bring in brake data
    ok = False
    while not ok:
        try:
            response = urlopen(brk_rec)
            data = json.loads(response.read().decode('utf-8'))
            brk_df = pd.DataFrame(data)
            ok = True
        except HTTPError as e:
            time.sleep(1)    

    # bring in location data
    ok = False
    while not ok:    
        try:
            response = urlopen(loc_rec)
            data = json.loads(response.read().decode('utf-8'))
            loc_df = pd.DataFrame(data)
            ok = True
        except HTTPError as e:
            time.sleep(1)      

    # bring in lap data
    ok = False
    while not ok:       
        try:
            response = urlopen(lap_rec)
            data = json.loads(response.read().decode('utf-8'))
            lap_df = pd.DataFrame(data)
            ok = True
        except HTTPError as e:
            time.sleep(1)  
            
    # bring in position data
    ok = False
    while not ok:       
        try:
            response = urlopen(pos_rec)
            data = json.loads(response.read().decode('utf-8'))
            pos_df = pd.DataFrame(data)
            ok = True
        except HTTPError as e:
            time.sleep(1)               

    #  convert to a different date format
    brk_df['date'] = pd.to_datetime(brk_df['date'], format='ISO8601')
    # brk_df['date'] = brk_df['date'].dt.floor('S') # Floor to seconds (removes microseconds)
    brk_df['date'] = brk_df['date'].dt.round('100ms') # Round to nearest tenth of a second (100 milliseconds)
    brk_df.drop_duplicates(subset=['date'], inplace=True, ignore_index=True, keep='last') # dedup by seconds

    # clean location df
    loc_df['date'] = pd.to_datetime(loc_df['date'], format='ISO8601')
    # loc_df['date'] = loc_df['date'].dt.floor('S') # Floor to seconds (removes microseconds)
    loc_df['date'] = loc_df['date'].dt.round('100ms') # Round to nearest tenth of a second (100 milliseconds)
    loc_df.drop_duplicates(subset=['date'], inplace=True, ignore_index=True, keep='last') # dedup by seconds

    # clean lap df
    lap_df['date_start'] = pd.to_datetime(lap_df['date_start'], format='ISO8601')
    # lap_df['date_start'] = lap_df['date_start'].dt.floor('S') # Floor to seconds (removes microseconds)
    lap_df['date_start'] = lap_df['date_start'].dt.round('100ms') # Round to nearest tenth of a second (100 milliseconds)
    lap_df.drop_duplicates(subset=['date_start'], inplace=True, ignore_index=True, keep='last') # dedup by seconds
    
    # clean position df
    pos_df['date'] = pd.to_datetime(pos_df['date'], format='ISO8601')
    # lap_df['date_start'] = lap_df['date_start'].dt.floor('S') # Floor to seconds (removes microseconds)
    pos_df['date'] = pos_df['date'].dt.round('100ms') # Round to nearest tenth of a second (100 milliseconds)
    pos_df.drop_duplicates(subset=['date'], inplace=True, ignore_index=True, keep='last') # dedup by seconds    

    # merge the df to create my df
    my_df = brk_df.merge(loc_df[['x','y','z','date']], how='inner', on='date'
                        ).merge(lap_df[['date_start','lap_number']], how='left', left_on='date', right_on='date_start')
    
    my_df.sort_values(by=['date'], inplace=True, ignore_index=True) # need to sort to do a near merge
    my_df = pd.merge_asof(my_df, pos_df[['date','position']], left_on='date', right_on='date', direction='nearest') # near merge
    
    my_df['lap_number'] = my_df['lap_number'].ffill() # could have done a near merge as well but would have lost the lap num = 0
    my_df['lap_number'].fillna(value=0, inplace=True)

    # convert it to binary and discuss with group whether 104 means anything
    my_df['brake'] = np.floor(my_df['brake']/100)

    # column to chart the track
    my_df['ones'] = 1
    
    return my_df.copy()

# Display the image in the left column
@st.cache_data(ttl=28800)
def load_and_crop_image(image_url, target_ratio=3/4):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    
    w, h = img.size
    current_ratio = w / h
    
    if current_ratio > target_ratio:  # Too wide
        new_w = int(h * target_ratio)
        left = (w - new_w) // 2
        img = img.crop((left, 0, left + new_w, h))
    elif current_ratio < target_ratio:  # Too tall
        new_h = int(w / target_ratio)
        top = (h - new_h) // 2
        img = img.crop((0, top, w, top + new_h))
    
    return img

# Load data
@st.cache_data(ttl=28800)
# def load_data(file_path="/Users/iangraetzer/Desktop/Georgia_Tech/data_vis_CSE_6040/group_project/lap_tire_df_no_outliers.csv"):


def load_data(file_path=lap_tire_df_no_outliers_path):
    tire_df = pd.read_parquet(file_path)
    
    if tire_df["Lap Time"].dtype == object:
        def convert_lap_time(time_str):
            if pd.isna(time_str):
                return np.nan
            
            try:
                if ':' in str(time_str):
                    parts = str(time_str).split(':')
                    minutes = float(parts[0])
                    seconds = float(parts[1])
                    return minutes * 60 + seconds
                else:
                    return float(time_str)
            except (ValueError, TypeError):
                return np.nan
        
        tire_df["Lap Time"] = tire_df["Lap Time"].apply(convert_lap_time)
    
    if tire_df["Lap Number"].dtype == object:
        tire_df["Lap Number"] = pd.to_numeric(tire_df["Lap Number"], errors="coerce")
    

    if "meeting_key" in tire_df.columns and "meeting_name" not in tire_df.columns:
        tire_df["meeting_name"] = tire_df["meeting_key"]

    
    return tire_df




# st.write(os.listdir())

# # Set page title
# st.markdown("""
# <style>
#     .reportview-container .main .block-container {
#         max-width: 800px;
#         padding-top: 2rem;
#         padding-right: 2rem;
#         padding-left: 2rem;
#         padding-bottom: 2rem;
#     }
#     .plotly-graph-div.js-plotly-plot {
#         transition: all 0.8s cubic-bezier(0.25, 0.1, 0.25, 1) !important;
#     }
#     .js-plotly-plot .plot-container .svg-container {
#         transition: all 0.8s cubic-bezier(0.25, 0.1, 0.25, 1) !important;
#     }

# </style>
# """, unsafe_allow_html=True)

# hide_img_fs = '''
# <style>
# button[title="View fullscreen"]{
#     visibility: hidden;}
# </style>
# '''

# st.markdown(hide_img_fs, unsafe_allow_html=True)



# Load the data
df_tire = load_data()

# preload the info data
f_df=get_info_data()
f_df = f_df.loc[f_df['session_type']=='Race',:].copy() # only races have the necessary info
f_df.reset_index(drop=True, inplace=True)


st.markdown("# Team 174:<br>A Deeper Look into the 2024 F1 Season", unsafe_allow_html=True)



# with dropdown_col:
driver_display_list = ['Max Verstappen', 'Logan Sargeant','Lando Norris', 'Pierre Gasly', 
'Sergio Perez', 'Fernando Alonso', 'Charles Leclerc', 'Lance Stroll', 'Kevin Magnussen', 'Yuki Tsunoda', 
'Alexander Albon','Zhou Guanyu', 'Nico Hulkenberg', 'Esteban Ocon','Lewis Hamilton','Carlos Sainz', 'George Russell', 
'Valtteri Bottas','Oscar Piastri']

driver_images = {
    'Max Verstappen': "https://upload.wikimedia.org/wikipedia/commons/thumb/5/52/2024-08-25_Motorsport%2C_Formel_1%2C_Gro%C3%9Fer_Preis_der_Niederlande_2024_STP_3973_by_Stepro_%28medium_crop%29.jpg/500px-2024-08-25_Motorsport%2C_Formel_1%2C_Gro%C3%9Fer_Preis_der_Niederlande_2024_STP_3973_by_Stepro_%28medium_crop%29.jpg",
    'Lando Norris': "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ac/2024-08-25_Motorsport%2C_Formel_1%2C_Gro%C3%9Fer_Preis_der_Niederlande_2024_STP_3975_by_Stepro_%28cropped2%29.jpg/800px-2024-08-25_Motorsport%2C_Formel_1%2C_Gro%C3%9Fer_Preis_der_Niederlande_2024_STP_3975_by_Stepro_%28cropped2%29.jpg",
    'Logan Sargeant': "https://encrypted-tbn3.gstatic.com/licensed-image?q=tbn:ANd9GcTqXtktUmodViriVMDtSgsLa_BaY0uJA-dEp8qTCfzPb8OSIACpPPvFl7p4acI-Tmn9NWkbCzjE3X1iSsg",
    'Pierre Gasly': "https://cdn1.f1oversteer.com/uploads/37/2025/03/GettyImages-2206319951-1140x760.jpg",
    'Sergio Perez': "https://superwatchman.com/wp-content/uploads/2023/03/309105394_422440280023742_3034340603059569439_n.jpg",
    'Fernando Alonso': "https://www.thesun.co.uk/wp-content/uploads/2023/11/2023-sao-paulo-brazil-photo-856765049.jpg?strip=all&w=640",
    'Charles Leclerc': "https://upload.wikimedia.org/wikipedia/commons/7/7b/2024-08-25_Motorsport%2C_Formel_1%2C_Gro%C3%9Fer_Preis_der_Niederlande_2024_STP_3978_by_Stepro_%28cropped2%29.jpg",
    'Lance Stroll': "https://pbs.twimg.com/media/F2TTivIWsAAKHXG?format=jpg&name=large",
    'Kevin Magnussen': "https://fluidideas.s3.eu-west-1.amazonaws.com/haas/s3fs-public/styles/landscape_desktop_1x/public/2023-11/kevin_preview_0.jpg?VersionId=e5i7DjopMZhDimzISYooqumUL9Ikefth",
    'Yuki Tsunoda': "https://img.redbull.com/images/c_fill,g_auto,w_450,h_600/q_auto:low,f_auto/redbullcom/2020/7/12/ek2rehxboc7eka3d99w5/yukitsunodaf2redbullring2",
    'Alexander Albon': "https://cdn-6.motorsport.com/images/amp/2y3MrNm6/s1000/alex-albon-red-bull-dtm-2021-1.jpg",
    'Zhou Guanyu': "https://motorcyclesports.net/wp-content/uploads/2025/03/2025022025-02-26T100911Z_1682194721_UP1EL2Q0S7AEA_RTRMADP_3_MOTOR-F1-TESTING-scaled-1.jpg",
    'Nico Hulkenberg': "https://cdn-1.motorsport.com/images/amp/6n9VEAOY/s1000/nico-hulkenberg-haas-f1-team-1.jpg",
    'Esteban Ocon': "https://shop.esteban-ocon.com/cdn/shop/articles/64f7c5c1d766ed396345974b_Screenshot_202023-02-23_20at_2011.19.27.png?v=1694450141&width=1100",
    'Lewis Hamilton': "https://cdn-4.motorsport.com/images/amp/6zQ51D7Y/s1000/lewis-hamilton-ferrari.jpg",
    'Carlos Sainz': "https://www.racefans.net/wp-content/uploads/2024/03/racefansdotnet-24-03-21-03-35-21-3-240022-scuderia-ferrari-australia-gp-thursday_23e3abab-a8e5-4059-b3b0-780093636aa7.jpg",
    'George Russell': "https://imageio.forbes.com/specials-images/imageserve/675383c944901da3ffa94963/0x0.jpg?format=jpg&crop=1861,1860,x685,y12,safe&height=416&width=416&fit=bounds",
    'Valtteri Bottas': "https://images.ctfassets.net/1fvlg6xqnm65/4r7T359lFdHOh1gZScptfd/5588ed50318b487130feddf3a51b2a8f/VB-Quiz-IMAGE-Mobile.jpg?w=3840&q=75&fm=webp",
    'Oscar Piastri': "https://media.cnn.com/api/v1/images/stellar/prod/gettyimages-2172131554-copy.jpg?c=16x9&q=h_833,w_1480,c_fill"
}

# driver_bio = pd.read_csv('/Users/iangraetzer/Desktop/Georgia_Tech/data_vis_CSE_6040/group_project/driver_bio_2.csv')
driver_bio = pd.read_parquet(driver_bio_2_path)


image_width = 250

# Create three columns for layout
col1, col2, col3 = st.columns([1, 3, 2], gap="small")

# Display the image in the left column
with col1:
    selected_driver_display = st.selectbox("Select Driver", driver_display_list)
    if selected_driver_display in driver_images:
        img_url = driver_images[selected_driver_display]
        try:
            cropped_img = load_and_crop_image(img_url)
            st.image(
                cropped_img,
                caption=f"{selected_driver_display}",
                use_container_width='always'
            )
        except Exception as e:
            st.error(f"Error loading image: {e}")
            # Fallback to display original URL if loading fails
            st.image(img_url, caption=f"{selected_driver_display}", width=250)


    
############################### 
# tire scatter plot in last col
############################### 


with col2:
    st.subheader("How Tire Compounds Affect Lap Speed")

    meeting_names = sorted(df_tire["meeting_name"].unique())
    selected_meeting = st.selectbox("Select Race", meeting_names, key="tab_tire")
    
    # Debug: Check filtered data
    filtered_data = df_tire[(df_tire["meeting_name"] == selected_meeting) & 
                     (df_tire["driver_name"].astype(str) == selected_driver_display)]
    
    if filtered_data.empty:
        st.write("No data available for the selected driver and race.")
    else:
        fig = px.scatter(
            filtered_data, 
            x="Lap Number", 
            y="Lap Time",
            color="compound",
            title=f"{selected_driver_display} Lap Times at {selected_meeting} by Tire Compound",
            labels={"Lap Time": "Lap Time (seconds)", "Lap Number": "Lap Number"}
        )

        # Add black outline/stroke to all markers
        fig.update_traces(marker=dict(size=10, line=dict(width=1, color='#8e8e8e')))

        # Improve layout
        fig.update_layout(
            xaxis=dict(
                tickmode='linear',
                tick0=0,
                dtick=10
            ),
            legend_title="Tire Compound"
        )

        # Add this line to display the plot
        st.plotly_chart(fig)

# Create tabs in the right column
with col3:
    # tab1, tab2, tab3 = st.tabs(["Season Stats","Tire Use Speed", "Basic Info"])
    tab1, tab2  = st.tabs(["Season Stats", "Basic Info"])    
    
  
    with tab1:
        st.subheader(f"{selected_driver_display}'s 2024 Race Results")
        
        # Filter the DataFrame to show only the selected driver's results
        driver_results = driver_bio[driver_bio['full_name'] == selected_driver_display]
        
        # Get unique races from the filtered DataFrame
        races = driver_results['meeting_name'].unique().tolist()
        
        # Create a summary table showing key stats for all races
        summary_data = []
        
        for race in races:
            # Get the row for this race
            race_row = driver_results[driver_results['meeting_name'] == race].iloc[0]
            
            summary_data.append({
                "Race": race,
                "Position": race_row['final_position'],
                "Fastest Lap": race_row['fastest_lap'],
                "Pit Stops": race_row['pit_stops']
            })
        
        # Create DataFrame from summary data - this line was missing
        df = pd.DataFrame(summary_data)
        
        # Display the DataFrame with custom column widths
        st.dataframe(
            df, 
            use_container_width=True, 
            hide_index=True
        )
    
    
    # Content for Tab 2: Career Highlights
    with tab2:
        st.subheader(f"{selected_driver_display}'s Career Highlights")
        if selected_driver_display == 'Max Verstappen':
            st.write("- Youngest F1 driver ever at age 17")
            st.write("- Youngest F1 race winner")
            st.write("- 3-time World Champion (2021, 2022, 2023)")
            st.write("- Most wins in a single season (19 in 2023)")
        elif selected_driver_display == 'Lando Norris':
            st.write("- First F1 win at Miami Grand Prix 2024")
            st.write("- 2018 FIA Formula 2 runner-up")
            st.write("- 2017 FIA Formula 3 European Champion")
            st.write("- Multiple podiums with McLaren")

################################################################################
########## interactive section ########## using selected_driver_display & selected_meeting
################################################################################

on = st.toggle(f"Generate Interactive Track for {selected_driver_display} at {selected_meeting}?")

if on:
    # need to have a button to load for speed's sake
    drv_num, ses_num = f_df.loc[(f_df['meeting_name']==selected_meeting) & 
                                (f_df['full_name'].str.upper() == selected_driver_display.upper()) & 
                                (f_df['year']==2024), ['driver_number','session_key']].values[0]

    # # variable selection dropbox
    # track_var = st.selectbox(
    #     "What variable would you like to see plotted?",
    #     ('speed', 'brake', 'throttle', 'rpm'),
    #     index=None,
    #     placeholder="Select variable to plot...",
    # )

    track_var = 'speed'

    # if on and (track_var != None):
    # pull data
    df = pull_f1_data(ses_num, drv_num)
    
    # curve marking
    dx = df['x'].diff()
    dy = df['y'].diff()
    angles = np.arctan2(dy, dx)
    angular_velocity = np.abs(np.degrees(angles.diff()))
    curve_threshold = 9.5 # sensitivity of curvature
    df['curve'] = 0
    df.loc[angular_velocity > curve_threshold, 'curve'] = 1 # mark the curve    

    # sometimes the curvature calculation is not sensitive enough and leads to discontinous curves
    # let's use an exponentially weighted mean to solve this issue (Zach: ewm is more reactive than rolling averages)
    # we will apply this "filter" twice in opposite directions and then apply thresholds to create boolean arrays
    # pass 1 (backward)
    df.sort_values(by=['date'], ascending=False, inplace=True)
    df['ewm1'] = df['curve'].ewm(span=5, adjust=False).mean()
    
    # pass 2 (forward)
    df.sort_values(by=['date'], ascending=True, inplace=True)
    df['ewm2'] = df['curve'].ewm(span=5, adjust=False).mean()
    # df['ewm0'] = df['ewm1'].ewm(span=5, adjust=False).mean() # i wanted to test a double pass but the additive method works best
    
    # combine results
    df['ewm'] = df['ewm1'] + df['ewm2']
    # df['ewm'] = np.where(df['ewm']>=0.5, 1 , 0) # this does better (in terms of continuous curvature), so we will use this method
    df['curve'] = np.where(df['ewm']>=0.5, 1 , 0)
    # df['ewm0'] = np.where(df['ewm0']>=0.5, 1 , 0)        
    
    # round it to make it more performant
    anim_df = df.copy()
    # start at the start of the first lap
    # anim_df = anim_df.loc[anim_df['lap_number']>0,:].copy() 
    start_time = anim_df.loc[anim_df['lap_number']>0,'date'].min() + dt.timedelta(seconds=-15)
    anim_df = anim_df[anim_df['date'] >= start_time].copy()

    # round to have a smoother animation
    anim_df['date'] = anim_df['date'].dt.round('1s')
    anim_df.drop_duplicates(subset=['date'], inplace=True, ignore_index=True, keep='first')

    # create a plot df to highlight the curves
    plot_df= df.loc[df['curve']==1,:].copy()

    # plot
    # Get unique timestamps and create a mapping
    unique_timestamps = anim_df['date'].unique()
    timestamp_to_index = {ts: idx for idx, ts in enumerate(unique_timestamps)}
    lap_list = [int(x) for x in anim_df['lap_number'].values]

    # global var for duration
    dur = 100
    
    # First, let's normalize the speed for a good color range
    speed_min = anim_df[track_var].min()
    speed_max = anim_df[track_var].max()

    # Create step list using indices
    step_list = [
        {
            'args': [
                [str(i)],  # Use index as frame name
                {
                    'frame': {'duration': 0, 'redraw': False},
                    'mode': 'immediate',
                    'transition': {'duration': 0}
                }
            ],
            'label': unique_timestamps[i].strftime('%H:%M:%S') + f' | Lap { lap_list[i] } ',  # Show time as label
            'method': 'animate'
        } for i in range(len(unique_timestamps))
    ]

    # actual plot
    fig = go.Figure(
        data=[                   
            go.Scatter(
                x=df['x'], 
                y=df['y'],
                mode='markers',
                # line=dict(color='lightgray', width=2),
                marker=dict(color=df['curve'], colorscale=[[0, 'grey'], [1, 'purple']]),
                name='Track (Curve=purple)',
                hoverinfo='skip'
            ),     
            go.Scatter(
                x=[anim_df.loc[0,'x']], 
                y=[anim_df.loc[0,'y']],
                mode='markers',
                hovertemplate='Speed: %{marker.color:,.0f} km/h <extra></extra>',
                name=f'{selected_driver_display}',
                marker=dict(
                    color=[anim_df.loc[0,track_var]],  # Initial speed
                    colorscale='RdYlBu_r',  # Red = Fast, Blue = Slow
                    showscale=True,       # Show the color scale
                    cmin=speed_min,       # Set color scale minimum
                    cmax=speed_max,       # Set color scale maximum
                    size=10,
                    colorbar=dict(
                        title=dict(
                            text='Speed (km/h)',
                            side='top'  # Positioning the title - 'top', 'right', or 'bottom'
                        ),
                        orientation='v',        # Make colorbar vertical
                        x=1.15,                  # Position the colorbar
                        y=0.5,                # Position below the plot
                        len=0.6,                # Length of the colorbar
                        thickness=20,           # Height of the colorbar
                        xanchor='center',       # Center anchor point
                        yanchor='middle',       # Middle anchor point
                    )
                )
            )
        ]
    )

    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "Time:",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": dur, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": step_list
    }

    fig.update_layout(width=800, height=800, # margin=dict(r=100), # Add bottom margin for colorbar
            xaxis=dict(range=[anim_df['x'].min()-500, anim_df['x'].max()+500], autorange=False, zeroline=False),
            yaxis=dict(range=[anim_df['y'].min()-500, anim_df['y'].max()+500], autorange=False, zeroline=False),
            title_text=f"Interactive Track for {selected_driver_display} at {selected_meeting}", title_x=0.5,
            updatemenus =[
                {
                'type':'buttons',
                'buttons':[
                        {
                            'args':[None, {"frame": {"duration": dur, "redraw": False},
                                            "fromcurrent": True, "transition": {"duration": dur*2, 
                                                                                # "easing": "cubic-in-out"
                                                                                }}],
                            'label':"Play",
                            'method':"animate",
                        },
                        {
                            "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                                "mode": "immediate",
                                                "transition": {"duration": 0}}],
                            "label": "Pause",
                            "method": "animate"
                        }        
                    ], # buttons

                # button placement
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": True,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"                
            }
        ], # updatemenus
        sliders=[sliders_dict]
    )


    # Create frames with matching indices
    frames = [
        go.Frame(
            data=[go.Scatter(
                x=[anim_df.loc[anim_df['date'] == ts, 'x'].iloc[0]],
                y=[anim_df.loc[anim_df['date'] == ts, 'y'].iloc[0]],
                hovertemplate='Speed: %{marker.color:,.0f} km/h <extra></extra>',
                mode='markers',
                marker=dict(
                    color=[anim_df.loc[anim_df['date'] == ts, track_var].iloc[0]],
                    colorscale='RdYlBu_r',
                    showscale=True,
                    cmin=speed_min,
                    cmax=speed_max,
                    size=10,
                    colorbar=dict(
                        title=dict(  # Changed from title='Speed (km/h)'
                            text='Speed (km/h)',
                            side='top'  # Changed from titleside='top'
                        ),
                        orientation='v',
                        # Removed titleside='top', 
                        x=1.15,
                        y=0.5,
                        len=0.6,
                        thickness=20,
                        xanchor='center',
                        yanchor='middle'
                    )                
                )            
            )],
            traces=[1],
            name=str(i)  # Use index as frame name
        ) 
        for i, ts in enumerate(unique_timestamps)
    ]

    fig.frames = frames

    # fig.show()
    # Create two columns for layout
    col1, col2 = st.columns([1, 1], gap="small")   

    with col1:     
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        show_cols = ['date','x','y','rpm','speed','n_gear','throttle','brake','lap_number','position','curve']
        st.dataframe(anim_df[show_cols], use_container_width=True, hide_index=True, height=800)
###
#Driving Style section 
###
st.divider()
st.header("What Defines a Driver")

