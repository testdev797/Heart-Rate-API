import json
import tempfile
import requests

from flask import Flask, request, jsonify
import cv2
import numpy as np
from scipy.signal import butter, filtfilt, convolve, find_peaks

app = Flask(__name__)

# --- JSON encoder for numpy types ---
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# --- Signal‑processing helper functions ---
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False, output='ba')
    return b, a

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False, output='ba')
    return b, a

def filter_all(data, fs, order=5, cutoff_high=8, cutoff_low=25):
    b, a = butter_highpass(cutoff_high, fs, order=order)
    highpassed = filtfilt(b, a, data)
    d, c = butter_lowpass(cutoff_low, fs, order=order)
    bandpassed = filtfilt(d, c, highpassed)
    return bandpassed

def process_signal(y, order, high, low, fs, avg_len):
    filtered = filter_all(y, fs, order, high, low)
    squared = filtered ** 2
    b = np.ones(avg_len) / avg_len
    averaged = filtfilt(b, [1], squared)
    return averaged

def give_bpm(averaged, time_bw_frame):
    threshold = min(averaged) + (max(averaged) - min(averaged)) / 16
    peaks = find_peaks(averaged, height=threshold)[0]
    if len(peaks) < 2:
        return 0.0
    diffs = np.diff(peaks)
    avg_diff = diffs.mean()
    avg_time = avg_diff * time_bw_frame
    bpm = 60.0 / avg_time
    return float(bpm)

# --- Download helper ---
def download_to_temp(url, timeout=10):
    resp = requests.get(url, stream=True, timeout=timeout)
    if resp.status_code != 200:
        return None
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    for chunk in resp.iter_content(chunk_size=8192):
        tmp.write(chunk)
    tmp.flush()
    return tmp.name

# --- Main API route ---
@app.route('/api', methods=['GET'])
def get_beats_per_min():
    video_base = request.args.get('query')
    token      = request.args.get('token')
    if not video_base or not token:
        return jsonify({"error": "Both 'query' and 'token' parameters are required"}), 400

    complete_url = f"{video_base}&token={token}"
    app.logger.info(f"Downloading video from: {complete_url}")

    # Download video to temp file
    local_path = download_to_temp(complete_url)
    if not local_path:
        return jsonify({"error": "Failed to download video"}), 400

    # Open with OpenCV
    cap = cv2.VideoCapture(local_path)
    if not cap.isOpened():
        return jsonify({"error": "Could not open downloaded video"}), 400

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        return jsonify({"error": "Could not determine video FPS"}), 400

    frame_count   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    time_bw_frame = 1.0 / fps

    # Read frames and collect center-block RGB means
    R = []; G = []; B = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        block = frame[h//2-100:h//2+100, w//2-100:w//2+100]
        mean = block.mean(axis=(0,1))  # B, G, R
        B.append(mean[0]); G.append(mean[1]); R.append(mean[2])

    cap.release()

    # Trim edges
    R = np.array(R[100:-100])
    G = np.array(G[100:-100])
    B = np.array(B[100:-100])

    # Filter & compute BPM for each channel
    order = 5
    high  = 10
    low   = 100
    fs    = 8 * int(fps + 1)
    avg_len = 7

    r_avg = process_signal(R, order, high, low, fs, avg_len)
    g_avg = process_signal(G, order, high, low, fs, avg_len)
    b_avg = process_signal(B, order, high, low, fs, avg_len)

    r_bpm = give_bpm(r_avg, time_bw_frame)
    g_bpm = give_bpm(g_avg, time_bw_frame)
    b_bpm = give_bpm(b_avg, time_bw_frame)
    avg_bpm = float((r_bpm + g_bpm + b_bpm) / 3)

    result = {
        "r_avg":   r_avg,
        "g_avg":   g_avg,
        "b_avg":   b_avg,
        "r_bpm":   r_bpm,
        "g_bpm":   g_bpm,
        "b_bpm":   b_bpm,
        "avg_bpm": avg_bpm
    }

    return json.dumps(result, cls=NumpyEncoder)

if __name__ == "__main__":
    app.run()
