#pipenv shell
#streamlit hrv  

import streamlit as st
import pandas as pd
import datetime as dt
import numpy as np
from datetime import datetime
from datetime import timedelta
from hrvanalysis import get_frequency_domain_features, remove_outliers, interpolate_nan_values
import freq_psd
import matplotlib.pyplot as plt
import matplotlib.style as style
import openai
import toml 

MOMENTS = False
# import plotly.express as px

# from dateutil.relativedelta import relativedelta # to add days or years

# import plotly.graph_objs as go

# CSS
css = r'''
    <style>
        [data-testid="stForm"] {border: 0px}
    </style>
'''

st.markdown(css, unsafe_allow_html=True)


def load_data(file):
    if file.name.endswith('.csv'):
        df = pd.read_csv(file, index_col=0, parse_dates=True)
    elif file.name.endswith('.parquet'):
        df = pd.read_parquet(file)
        # print(df)
        # if 'timestamp'
    else:
        raise ValueError("Unsupported file type: " + file.name)

    # Ask for the HRV column name

    # if col_name not in df.columns:
    #     default_col_name = 'inter_beat_interval'
    #     col_name = st.text_input("HRV column name", default_col_name)
    #     st.error(f"{col_name} is not a valid HRV column name, aborting.")
    #     st.stop()


        
    # if 'timestamp' not in df.columns:
        # st.error(f"No timestamp found, aborting.")
        # st.write(df.columns)
        # st.stop()
    # else:
    # print(df.index.dtype)
    if not str(df.index.dtype).startswith('datetime64'):
        df = df.set_index('timestamp')

    # st.table(df.head())
    col_name = 'inter_beat_interval'
    return df, col_name

# Processing inter_beat_intervals
def process_ibi(loc):
    candidates=['timestamp','value','bytes']

    # Value to Byte Array
    inter_beat_intervals = pd.read_csv(loc)
    inter_beat_intervals['bytes'] = inter_beat_intervals['value'].apply(lambda x: bytearray.fromhex(x))
    inter_beat_intervals['inter_beat_interval'] = pd.Series(timestamp_conversion(inter_beat_intervals))
    if 'datetime' in inter_beat_intervals:
        inter_beat_intervals.rename(columns={'datetime':'timestamp'},inplace=True)

    inter_beat_intervals['timestamp'] = pd.to_datetime(inter_beat_intervals['timestamp'],unit='s')
    inter_beat_intervals['timestamp'] = inter_beat_intervals['timestamp']
    inter_beat_intervals = inter_beat_intervals.set_index('timestamp')
    inter_beat_intervals = inter_beat_intervals.drop([x for x in candidates if x in inter_beat_intervals.columns], axis=1)
    return inter_beat_intervals

def timestamp_conversion(df):
    # Create Numpy Array to Iterate Over
    heartbeat_frames = np.array(df['value'])
    # Deduce Data From Bytearray
    times = []
    for heartbeat_frame in heartbeat_frames:
        frame_heartbeat_count = int(heartbeat_frame[1:2])
        for heartbeat_index in range(1,frame_heartbeat_count+1):
            index_start = 3+(heartbeat_index-1)*8
            index_end = index_start + 4
            time_bytes = heartbeat_frame[index_start*2:index_end*2]
            time_hex = bytearray.fromhex(time_bytes)
            times.append(int.from_bytes(time_hex, byteorder="little"))

    # Convert to Time array in ms format
    times_ms = []
    for time_index in range(1, len(times)):
        baseline_time = times[time_index-1]
        comparison_time = times[time_index]
        inter_beat_interval = comparison_time - baseline_time
        if inter_beat_interval > 300 and inter_beat_interval < 2500:
            times_ms.append(comparison_time - baseline_time)

    return times_ms

def get_rhr(rr):
    hr = 60000/rr
    rhr = np.min(hr)
    return rhr
def timedomain(rr):
    results = {}

    hr = 60000/rr
    
    results['Mean RR (ms)'] = np.mean(rr)
    results['STD RR/SDNN (ms)'] = np.std(rr)
    results['Mean HR (Kubios\' style) (beats/min)'] = 60000/np.mean(rr)
    results['Mean HR (beats/min)'] = np.mean(hr)
    results['STD HR (beats/min)'] = np.std(hr)
    results['Min HR (beats/min)'] = np.min(hr)
    results['Max HR (beats/min)'] = np.max(hr)
    results['RMSSD (ms)'] = np.sqrt(np.mean(np.square(np.diff(rr))))
    results['NNxx'] = np.sum(np.abs(np.diff(rr)) > 50)*1
    results['pNNxx (%)'] = 100 * np.sum((np.abs(np.diff(rr)) > 50)*1) / len(rr)
    return results

def remove_outliers(df,type='normal',size=30):
    arr = df.to_numpy()
    outlier_counter = 0
    middle = int(size/2)
    if len(arr) > size:
        for i in range(0,len(arr)):
            window = arr[i:i+size]
            mean = np.mean(window)

            # Thresholding 
            if type == 'normal':
                threshold_low = 0.5
                threshold_high = 1.5
            elif type == 'strict':
                threshold_low = 0.75
                threshold_high = 1.25
            elif type == 'soft':
                threshold_low = 0.20
                threshold_high = 1.80
            
            # First <size> values
            if i < size:
                if arr[i] < (threshold_low*mean) or arr[i] > (threshold_high*mean):
                    arr[i] = mean
                    outlier_counter += 1
            # Middle <size values>
            elif i < (len(arr)-size):
                center = arr[i+middle]
                if center < (threshold_low*mean) or center > (threshold_high*mean):
                    arr[i+middle] = mean
                    outlier_counter += 1
            # Final <size> values
            elif i > (len(arr)-size):
                if arr[i] < (threshold_low*mean) or arr[i] > (threshold_high*mean):
                    arr[i] = mean
                    outlier_counter += 1

    else:
        print(f"Window seems empty. Skipping...")
    st.write(f'        {outlier_counter} outliers removed from {df.name}')
    return arr

def analyzer(df,start_time,end_time, openai_key, moments_df):
    st.title('Results')    
    st.write('**Pre-Processing**')
    df_window = df.query("index >= @start_time and index < @end_time")
    if moments_df is not None:
        moments_df = moments_df.query("index >= @start_time and index < @end_time")
    diff = (end_time - start_time) / pd.Timedelta(minutes=1)
    print(diff)
    # This remove outliers from signal
    # rr_intervals_without_outliers = remove_outliers(rr_intervals=rr_intervals,  
    #                                                 low_rri=300, high_rri=2000)
    # # This replace outliers nan values with linear interpolation
    # interpolated_rr_intervals = list(interpolate_nan_values(rr_intervals=rr_intervals_without_outliers,
    #                                                 interpolation_method="linear"))
    df_window['inter_beat_interval'] = remove_outliers(df_window['inter_beat_interval'])
    df_index, df_rri = freq_psd.threshold_filter(df_window['inter_beat_interval'],threshold='strong',local_median_size=5)
    data = {'inter_beat_interval': df_rri}
    df_window = pd.DataFrame(data, index=df_index)
    # df_window['inter_beat_interval'] = freq_psd.moving_median(df_window)
    df_window['inter_beat_interval'] = df_window['inter_beat_interval'].interpolate(method='linear')
    rr_intervals = df_window['inter_beat_interval']

    st.write(df_window)

    # # Create the figure and axes objects
    # fig, ax = plt.subplots()

    # # Plot the rr_intervals series as a line with blue markers
    # ax.plot(rr_intervals.index, rr_intervals.values, '-o', color='blue', markersize=3)

    # # Set the x-axis label
    # ax.set_xlabel('Date and Time')

    # # Set the y-axis label
    # ax.set_ylabel('Inter-Beat Interval (ms)')

    # # Set the title of the plot
    # ax.set_title('Inter-Beat Intervals over Time')
    # fig = px.line(rr_intervals,markers=True)
    st.write('HRV data is unsteady - a single movement can already mess up your HRV data. Use this plot to see whether your data is messy, usually denoted by straight lines')
    st.line_chart(rr_intervals)
    # st.table(rr_intervals.head())
    # st.table(df_window.head())
    timedomain_values = timedomain(rr_intervals)





    # if ibi_df.index.duplicated().sum() == 0:

    # st.write(timedomain_values)
    time_df = pd.DataFrame.from_dict(timedomain_values,orient='index',columns=['value'])
    # st.write(time_df)
    st.write('**Time Domain**')
    st.table(time_df)

    st.write("*RMSSD Trend*")
    rmssd = rr_intervals.rolling('5min',center=True).apply(freq_psd.do_rmssd)
    rmssd_trend = freq_psd.get_trend_line(rmssd)

    rmssd_fig, ax = plt.subplots(figsize=(20,10))
    ax.plot(rmssd.index, rmssd.values, '-',label='rmssd')
    ax.plot(rmssd.index, rmssd_trend.values, '-',label='trend')
    ax.set_ylabel('Value (ms)', color = 'black')
    ax.set_xlabel('Time', color = 'black')
    ax.legend(loc='best')

    if moments_df is not None:
        if not moments_df.empty:
            ymin, ymax = ax.get_ylim()
            for moment in moments_df.index:
                ax.axvline(x=moment, ymin=ymin, ymax=ymax, color='red', alpha=0.8)
                # ax.text(0.5, -0.1, 'MOMENT', transform=ax.transAxes, ha='center', va='center')
                ax.text(moment, ymin-0.1*(ymax-ymin), 'MOMENT', ha='center', va='center', color='red')

                handles, labels = ax.get_legend_handles_labels()
                handles.append(ax.axvline(x=moments_df.index[0], ymin=0, ymax=1, color='red'))
                labels.append('moment')
                ax.legend(handles, labels, loc='best')


    st.pyplot(rmssd_fig)
    rmssd_fig.clf()
    ax.cla()

    st.write("*RHR calculated using HRV*")

    rhr = rr_intervals.rolling('1min',center=True).apply(get_rhr).rolling('5min').mean()
    rhr_trend = freq_psd.get_trend_line(rhr)
    rhr_fig, ax = plt.subplots(figsize=(20,10))
    ax.plot(rhr.index, rhr.values, '-',label='resting heart rate')
    ax.plot(rhr.index, rhr_trend.values, '-',label='trend')
    ax.set_ylabel('Value (bpm)', color = 'black')
    ax.set_xlabel('Time', color = 'black')
    ax.legend(loc='best')


    # if not moments_df.empty:
    #     ymin, ymax = ax.get_ylim()
    #     for moment in moments_df.index:
    #         ax.axvline(x=moment, ymin=ymin, ymax=ymax, color='red', alpha=0.8)
    #         # ax.text(0.5, -0.1, 'MOMENT', transform=ax.transAxes, ha='center', va='center')
    #         ax.text(moment, ymin-0.1*(ymax-ymin), 'MOMENT', ha='center', va='center', color='red')
    #         handles, labels = ax.get_legend_handles_labels()
    #         handles.append(ax.axvline(x=moments_df.index[0], ymin=0, ymax=1, color='red'))
    #         labels.append('moments')
    #         ax.legend(handles, labels, loc='best')

    st.pyplot(rhr_fig)

    ##################################### 

    frequency_values = get_frequency_domain_features(rr_intervals)
    for key in frequency_values:
    # rounding to K using round()
        # if key != 'lf_hf_ratio':
        frequency_values[key] = '{:.2f}'.format(frequency_values[key]).rstrip('0').rstrip('.')


    st.write('**Frequency Domain**')
    freq_df = pd.DataFrame.from_dict(frequency_values,orient='index',columns=['value'])

    # def get_lf_hf(window):
    #     window = list(window)
    #     frequency_domain_features = get_frequency_domain_features(nn_intervals = window)
    #     return frequency_domain_features['lf_hf_ratio']
            
    
    # lf_hf = rr_intervals.rolling('5min',center=True).apply(get_lf_hf)
    # lfhf_trend = freq_psd.get_trend_line(lf_hf)
    # fig, ax = plt.subplots(figsize=(20,10))
    # ax.plot(lf_hf.index, lf_hf.values, '-',label='lf hf')
    # ax.plot(lf_hf.index, lfhf_trend.values, '-',label='trend')
    # ax.set_ylabel('Value (bpm)', color = 'black')
    # ax.set_xlabel('Time', color = 'black')
    # ax.legend(loc='best')
    # st.pyplot(fig)

    st.table(freq_df)
    freq_psd.plot_psd(rr_intervals, method="welch")
    
    # if st.button('start configuration'):
        # st.session_state['step'] = 1
    # if st.session_state['step'] == 1:
        # with st.form('my form'):
        #     st.session_state['number'] = st.number_input('choose a number', 1, 13)
        #     if st.form_submit_button("save configuration"):            
        #         st.session_state['step'] = 2
        #         st.experimental_rerun()  # form should not be shown after clicking 'save' 
    

    # with st.form(key="chatgpt"):
        # _, _, _, col, _, _, _ = st.columns([1]*6+[1])
        # run_chat = st.form_submit_button("Ask ChatGPT about my score", disabled=(freq_df is None))
        # submitted = st.form_submit_button("Start analyzing")
        # if run_chat:

    # st.write(timedomain_values['RMSSD (ms)'], frequency_values['lf'], frequency_values['hf'],frequency_values['lf_hf_ratio'])
    if openai_key:
        if not freq_df.empty and not time_df.empty:

            def show_messages(text):
                messages_str = [
                    f"{_['role']}: {_['content']}" for _ in st.session_state["messages"][1:]
                ]
                text.text_area("Messages", value=str("\n".join(messages_str)), height=400)


            openai.api_key = openai_key
            BASE_PROMPT = [{"role": "system", "content": "You are a helpful assistant."}]

            if "messages" not in st.session_state:
                st.session_state["messages"] = BASE_PROMPT

            st.header("ChatGPT HRV Analysis")
            # st.markdown("""---""")

            # text = st.empty()
            # show_messages(text)
            # ']])
            print(timedomain_values['RMSSD (ms)'])
            rmssd_ms = int(float(timedomain_values['RMSSD (ms)']))
            lf = int(float(frequency_values['lf']))
            hf = int(float(frequency_values['hf']))
            lf_hf = frequency_values['lf_hf_ratio']
            prompt = f"Tell me about my HRV: RMSSD (ms): {rmssd_ms} and my LF, HF, and LF HF Ratio in Hz: , {lf, hf,lf_hf}. The total time of measurement was {diff} minutes. Don't use more than 200 words. Keep it simple and motivating, very positive, and talk about mental balance. Explain how it relates to mental balance. Always display the values."
            # st.write(len(prompt))
            # if st.form_submit_button("Send"):
            with st.spinner("Generating response..."):
                st.session_state["messages"] += [{"role": "user", "content": prompt}]
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo", messages=st.session_state["messages"],
                    temperature=0.3,
                    max_tokens=500,


                )
                message_response = response["choices"][0]["message"]["content"]
                # st.session_state["messages"] += [
                #     {"role": "system", "content": message_response}
                # ]
                # show_messages(text)
                st.write(message_response)

            if st.form_submit_button("Clear"):
                prompt = ""
                # st.session_state["messages"] = BASE_PROMPT
            # show_messages(text)
    # Display the bar chart using st.bar_chart
    # st.bar_chart(lf_hf_df)
    # st.bar_chart(freq_df.iloc['lf','hf']])
    # st.write('Plotting!')
    # st.write(time)


def get_time_range(df, timezone):
    # Convert the datetime index to the selected timezone
    df.index = df.index + timedelta(hours=timezone)
    
    # Get the start and end datetime values as strings
    start_time = df.index.min().strftime('%Y-%m-%d %H:%M:%S %Z%z')
    end_time = df.index.max().strftime('%Y-%m-%d %H:%M:%S %Z%z')
    
    return start_time, end_time

def main():
    df = None
    end_datetime = None
    # st.title("HRV Analyzer")
    st.markdown("<h1 style='text-align: center; color: black;'>HRV Analyzer</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: grey;'>NOWATCH</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: grey;'>Lennart Zegerius</h1>", unsafe_allow_html=True)
    st.write('\n')
    st.markdown("""---""")
    format = st.radio(
    "Uploading CSV or Parquet?",
    ('CSV', 'Parquet'))

    if format == 'Parquet':
        file = st.file_uploader("Upload a parquet file containing interbeat intervals. ",type='parquet')
        if file is not None:
            df, hrv_col = load_data(file)
            # st.table(df.head())
    else:
        file = st.file_uploader("Upload a csv file containing interbeat intervals. ",type='csv')
        
        if file is not None:
            df = process_ibi(file)
            # st.table(df.head())
        # st.write("You didn\'t select comedy.")
    # file = st.file_uploader("Upload a parquet file containing interbeat intervals. ")
    # csv_file = st.file_uploader("Upload hearbeats.csv")
    if df is not None:
        
        
        # st.write(df.index.dtype)
        # dates = pd.to_datetime(df.index)
        # st.write(dates.dtype)

        timezone = 0
        df.index = df.index + timedelta(hours=timezone)
        timezone = st.slider('Timezone (UTC)', min_value=-12, max_value=12, step=1, value=0)
        df.index = df.index + timedelta(hours=timezone)
        st.write("Date: ",df.index.min(), " - ", df.index.max())
        # st.write("End datetime: ",df.index.max())


        Date = df.index.unique().tolist()

        min_value = datetime.strptime(min(Date).strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')  # str to datetime
        max_value = datetime.strptime(max(Date).strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
        value = (min_value, max_value)
        # st.write(df.index.max())
#         st.write(min_value.dtype)
        
        # Model = st.slider(
        #     'Pick a RR-interval window:',
        #     min_value=min_value,
        #     max_value=max_value,
        #     format = "HH:mm",
        #     step = timedelta(minutes=2),
        #     value=value,
        #     )


        # selmin, selmax = Model
        selmind = min_value.strftime('%H:%M')  # datetime to str
        selmaxd = max_value.strftime('%H:%M')
        
        # st.write('Or pick a time here')

        unique_dates = np.unique(df.index.date)
        start_date = st.selectbox(options=unique_dates,label='Start Date:')
        start_time = st.text_input(label='Time:',placeholder=selmind)
        start_datetime_str = f"{start_date} {start_time}"
        if len(start_time) == 5:
            start_datetime = datetime.strptime(start_datetime_str, '%Y-%m-%d %H:%M')
        # st.write(start_date, start_time)

        if start_time:
            end_date = st.selectbox(options=unique_dates,label='End Date:',index=len(unique_dates)-1)
            end_time = st.text_input(label='Time:',placeholder=selmaxd)
            # Concatenate the end date and end time strings
            end_datetime_str = f"{end_date} {end_time}"
            if len(end_time) == 5:
                end_datetime = datetime.strptime(end_datetime_str, '%Y-%m-%d %H:%M')

            # st.write(end_date, end_time)

            if end_time:
                openai_key = st.text_input(label='OpenAI API Key (optional): ',placeholder="...")
                moments = st.file_uploader("Upload Moments (optional): ",type='csv')
                if moments is not None:
                    filename = str(moments.name)

                    if filename.startswith('system'):
                        moments_df = pd.read_csv(moments).drop('value',axis=1)
                        moments_df['timestamp'] = pd.to_datetime(moments_df['timestamp'],unit='s')
                        moments_df = moments_df[moments_df['type'] == 'MOMENT']
                        moments_df = moments_df.set_index('timestamp')
                        moments_df.index = moments_df.index + timedelta(hours=timezone)
                        if not moments_df.empty:
                            MOMENTS = True
                            st.write(moments_df)
                    elif filename.startswith('moment'):                    
                        if moments is not None:
                            moments_df = pd.read_csv(moments).drop('value',axis=1).set_index('timestamp')
                            moments_df = moments_df[moments_df['type'] == 'MOMENT']
                            moments_df.index = pd.to_datetime(moments_df.index) + timedelta(hours=timezone)
                            if not moments_df.empty:
                                MOMENTS = True
                                st.write(moments_df)
                    else:
                        st.write("No moments found")
                        moments_df = None
                else:
                    moments_df = None
        st.markdown("""---""")
        
        st.write('\n')

        st.write('\n')
        if start_time and end_time:
            st.write("Selected window:", start_datetime, " - ", end_datetime)
        # st.write(selmax)
        
        with st.form(key="my_form"):
            _, _, _, col, _, _, _ = st.columns([1]*6+[1])
            submitted = col.form_submit_button("Start analyzing", disabled=(end_datetime is None))
            # submitted = st.form_submit_button("Start analyzing")
            if submitted and start_datetime and end_datetime:
                analyzer(df, start_datetime, end_datetime, openai_key, moments_df)

                # st.write("Running analyzer...")
                

if __name__ == '__main__':
    # Set page title
    # st.set_page_config(page_title="HRV Dashboard")
    # st.set_page_config(layout="wide") 
    # Set app URL
    # app_url = 'https://hrv.lenn.dev'
    # st.write(f'<iframe src="{app_url}" width="100%" height="900" frameborder="0" scrolling="no"></iframe>', unsafe_allow_html=True)
    main()
