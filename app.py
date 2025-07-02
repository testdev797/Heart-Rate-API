import json
import os
import requests
from flask import Flask, request
import cv2
import numpy as np
from scipy.signal import butter, convolve, find_peaks, filtfilt

app = Flask(__name__)


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5*fs
    normal_cutoff = cutoff/nyq
    b, a = butter(order, normal_cutoff, btype='high',
                  analog=False, output='ba')
    return b, a


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5*fs
    normal_cutoff = cutoff/nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False, output='ba')
    return b, a


def filter_all(data, fs, order=5, cutoff_high=8, cutoff_low=25):
    b, a = butter_highpass(cutoff_high, fs, order=order)
    highpassed_signal = filtfilt(b, a, data)
    d, c = butter_lowpass(cutoff_low, fs, order=order)
    bandpassed_signal = filtfilt(d, c, highpassed_signal)
    return bandpassed_signal


def process_signal(y, order_of_bandpass, high, low, sampling_rate, average_filter_sample_length):
    filtered_signal = filter_all(
        y, sampling_rate, order_of_bandpass, high, low)
    squared_signal = filtered_signal**2
    b = (np.ones(average_filter_sample_length))/average_filter_sample_length
    a = np.ones(1)
    averaged_signal = convolve(squared_signal, b)
    averaged_signal = filtfilt(b, a, squared_signal)
    return averaged_signal


def give_bpm(averaged, time_bw_fram):
    # Handle case where there are not enough peaks to calculate BPM
    if len(averaged) < 2:
        return 0.0

    r_min_peak = min(averaged)+(max(averaged)-min(averaged))/16
    r_peaks = find_peaks(averaged, height=r_min_peak)
    
    # Handle case where not enough peaks are found
    if len(r_peaks[0]) < 2:
        print("Not enough peaks found to calculate BPM.")
        return 0.0

    diff_sum = 0
    total_peaks = len(r_peaks[0])
    i = 0

    while i < total_peaks-1:
        diff_sum = diff_sum+r_peaks[0][i+1]-r_peaks[0][i]
        i = i+1

    avg_diff = float(diff_sum/(total_peaks-1))
    avg_time_bw_peaks = float(avg_diff*time_bw_fram)
    
    # Avoid division by zero if peaks are too close
    if avg_time_bw_peaks == 0:
        return 0.0
        
    bpm = float(60.0/avg_time_bw_peaks)
    print("Calculated heart rate "+str(bpm))
    return bpm


@app.route('/api', methods=['GET'])
def get_beats_per_min():
    # --- 1. Get URL and handle missing parameter ---
    try:
        video_url = request.args['query']
    except KeyError:
        return {"error": "Missing 'query' parameter with the video URL."}, 400

    print(f"Received URL: {video_url}")
    
    temp_video_path = "/tmp/temp_video.mp4"

    # --- 2. Download the video from the URL ---
    try:
        response = requests.get(video_url, stream=True)
        # Raise an HTTPError for bad responses (4xx or 5xx)
        response.raise_for_status()
        
        with open(temp_video_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to download video from URL. Reason: {e}"}, 400

    # --- 3. Open the downloaded video file with OpenCV ---
    video_data = cv2.VideoCapture(temp_video_path)

    # --- 4. CRITICAL: Check if the video was opened successfully ---
    if not video_data.isOpened():
        os.remove(temp_video_path) # Clean up the temp file
        return {"error": "OpenCV could not open the video file. It might be corrupted or in an unsupported format."}, 500

    # --- 5. Get video properties and add ZeroDivisionError check ---
    fps = video_data.get(cv2.CAP_PROP_FPS)
    frame_count = int(video_data.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps == 0:
        video_data.release()
        os.remove(temp_video_path)
        return {"error": "Video has zero FPS or could not be read properly. Cannot process."}, 400

    vid_length = frame_count/fps
    time_bw_frame = 1.0/fps

    # --- 6. Process the video frames ---
    R, G, B = [], [], []

    while True:
        ret, frame = video_data.read()
        if not ret:
            break

        # Extract the center 100x100 pixel region
        h, w, _ = frame.shape
        start_y, start_x = int(h/2) - 50, int(w/2) - 50
        end_y, end_x = int(h/2) + 50, int(w/2) + 50
        
        center_region = frame[start_y:end_y, start_x:end_x]
        
        # Calculate average color of the region
        # BGR is the order for OpenCV
        avg_color = np.mean(center_region, axis=(0, 1))
        B.append(avg_color[0])
        G.append(avg_color[1])
        R.append(avg_color[2])

    video_data.release() # Release the video object

    # --- 7. Clean up the temporary file ---
    os.remove(temp_video_path)

    if len(R) < 200:
        return {"error": f"Video is too short. Needs at least 200 frames for processing, but only found {len(R)}."}, 400
        
    # Discarding first few frames and last few
    R = np.array(R[100:-100])
    G = np.array(G[100:-100])
    B = np.array(B[100:-100])

    # --- 8. Signal processing and BPM calculation ---
    r_cutoff_high = 10
    r_cutoff_low = 100
    r_order_of_bandpass = 5
    r_sampling_rate = 8*int(fps+1)
    r_average_filter_sample_length = 7

    r_averaged = process_signal(R, r_order_of_bandpass, r_cutoff_high,
                                r_cutoff_low, r_sampling_rate, r_average_filter_sample_length)
    g_averaged = process_signal(G, r_order_of_bandpass, r_cutoff_high,
                                r_cutoff_low, r_sampling_rate, r_average_filter_sample_length)
    b_averaged = process_signal(B, r_order_of_bandpass, r_cutoff_high,
                                r_cutoff_low, r_sampling_rate, r_average_filter_sample_length)

    bpms = []
    bpms.append(give_bpm(r_averaged, time_bw_frame))
    bpms.append(give_bpm(g_averaged, time_bw_frame))
    bpms.append(give_bpm(b_averaged, time_bw_frame))

    bpm = (bpms[0]+bpms[1]+bpms[2])/3

    result = {
        "r_avg": r_averaged,
        "g_avg": g_averaged,
        "b_avg": b_averaged,
        "r_bpm": bpms[0],
        "g_bpm": bpms[1],
        "b_bpm": bpms[2],
        "avg_bpm": bpm
    }

    json_dump = json.dumps(result, cls=NumpyEncoder)
    return json_dump


if __name__ == "__main__":
    app.run()
