import os

def set_permissions_recursive(directory_path):
    for root, dirs, files in os.walk(directory_path):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            os.chmod(dir_path, 0o777)
        for file_name in files:
            file_path = os.path.join(root, file_name)
            os.chmod(file_path, 0o777)

# Specify the path to the folder you want to set permissions for
folder_path = '/path/to/your/folder'

try:
    set_permissions_recursive(folder_path)
    print(f"Permissions set to 777 for folder and its contents: {folder_path}")
except Exception as e:
    print(f"Error setting permissions: {e}")

sudo yum groupinstall "Development Tools" -y
sudo yum install yasm nasm pkgconfig zlib-devel -y
sudo amazon-linux-extras install epel -y  # for Amazon Linux 2
sudo yum install libX11-devel freetype-devel fontconfig-devel libXfixes -y
cd /usr/local/src
sudo wget https://ffmpeg.org/releases/ffmpeg-snapshot.tar.bz2
sudo tar xjvf ffmpeg-snapshot.tar.bz2
cd ffmpeg
sudo ./configure --prefix=/usr/local --enable-gpl --enable-nonfree --enable-libx264 --enable-libx265 --enable-libvpx --enable-libtheora --enable-libmp3lame --enable-libfdk-aac --enable-libfreetype --enable-libass --enable-libopus --enable-libvorbis --enable-libvpx --enable-sdl2
sudo make
sudo make install
sudo ldconfig


from pydub import AudioSegment

# Load the audio file
audio_file = "path/to/your/large_file.wav"
audio = AudioSegment.from_file(audio_file)

# Define the length of each segment (10 seconds here)
segment_length_ms = 10 * 1000  # 10 seconds in milliseconds

# Calculate the number of segments
num_segments = len(audio) // segment_length_ms + (1 if len(audio) % segment_length_ms else 0)

# Split and export each segment
for i in range(num_segments):
    start_ms = i * segment_length_ms
    end_ms = start_ms + segment_length_ms
    segment = audio[start_ms:end_ms]
    
    # Export segment to a new file
    segment_filename = f"segment_{i+1}.wav"
    segment.export(segment_filename, format="wav")
    print(f"Exported {segment_filename}")



import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_optical_flow(video_path):
    cap = cv2.VideoCapture(video_path)
    
    ret, prev_frame = cap.read()
    if not ret:
        print("Failed to read video")
        return []
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    motion_magnitudes = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mean_magnitude = np.mean(magnitude)
        motion_magnitudes.append(mean_magnitude)

        prev_gray = gray

    cap.release()
    return motion_magnitudes

def identify_slow_motion_sections(motion_magnitudes, threshold_factor=0.5, smoothing_window=5):
    avg_motion = np.mean(motion_magnitudes)
    slow_motion_threshold = avg_motion * threshold_factor
    
    # Apply smoothing
    smoothed_magnitudes = np.convolve(motion_magnitudes, np.ones(smoothing_window)/smoothing_window, mode='same')
    
    slow_motion_sections = []

    for i, magnitude in enumerate(smoothed_magnitudes):
        if magnitude < slow_motion_threshold:
            slow_motion_sections.append(i)

    return slow_motion_sections, smoothed_magnitudes

def plot_motion_magnitudes(motion_magnitudes, smoothed_magnitudes, slow_motion_sections):
    plt.figure(figsize=(10, 5))
    plt.plot(motion_magnitudes, label='Original Motion Magnitude')
    plt.plot(smoothed_magnitudes, label='Smoothed Motion Magnitude', linestyle='--')
    plt.scatter(slow_motion_sections, [smoothed_magnitudes[i] for i in slow_motion_sections], color='red', label='Slow Motion', zorder=2)
    plt.xlabel('Frame Number')
    plt.ylabel('Mean Motion Magnitude')
    plt.title('Motion Magnitude and Slow Motion Sections')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    video_path = "path/to/your/video.mp4"  # Change this to your video file path
    motion_magnitudes = calculate_optical_flow(video_path)
    slow_motion_sections, smoothed_magnitudes = identify_slow_motion_sections(motion_magnitudes)
    plot_motion_magnitudes(motion_magnitudes, smoothed_magnitudes, slow_motion_sections)


import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.io import read_video
from PIL import Image
import numpy as np

# Load pre-trained model
model = models.video.r3d_18(pretrained=True)
model.eval()

# Video transformation
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_features(video_path, segment_duration=1, fps=30):
    video, _, info = read_video(video_path)
    video = video.permute(0, 3, 1, 2)  # Convert to (T, C, H, W)
    frame_count = video.shape[0]
    segment_size = segment_duration * fps
    
    features = []
    for i in range(0, frame_count, segment_size):
        segment = video[i:i+segment_size]
        if segment.shape[0] < segment_size:
            continue
        
        # Convert each frame in the segment to PIL Image, then apply transforms
        segment = torch.stack([transform(Image.fromarray(frame.permute(1, 2, 0).numpy())) for frame in segment])
        segment = segment.unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            feature = model(segment)
            features.append(feature.squeeze().cpu().numpy())
    
    return np.array(features)

# Assuming additional code follows here for classification and plotting...


def classify_segments(features, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
    return kmeans.labels_

def plot_classification(labels):
    plt.figure(figsize=(10, 5))
    plt.plot(labels, marker='o', linestyle='-')
    plt.xlabel('Segment Number')
    plt.ylabel('Cluster Label')
    plt.title('Video Segment Classification')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    video_path = "path/to/your/video.mp4"  # Change this to your video file path
    features = extract_features(video_path)
    labels = classify_segments(features)
    plot_classification(labels)

