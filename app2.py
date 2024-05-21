import aiohttp
import asyncio
from PIL import Image
from io import BytesIO

async def is_image(url, session):
    for attempt in range(3):  # Retry up to 3 times
        try:
            async with session.get(url) as response:
                # Check if the content type indicates an image
                content_type = response.headers.get('content-type', '').lower()
                if content_type.startswith('image'):
                    # Optionally, you can also check if the downloaded content is a valid image
                    image_data = await response.read()
                    Image.open(BytesIO(image_data))  # This line will raise an exception if it's not a valid image
                    return True
                else:
                    return False
        except Exception as e:
            print(f"Error checking image for URL {url}: {e}")
            if attempt < 2:
                print(f"Retrying... (Attempt {attempt + 2})")
                await asyncio.sleep(1)  # Wait for 1 second before retrying
            else:
                print("Max attempts reached. Giving up.")
                return False

async def main(urls):
    async with aiohttp.ClientSession() as session:
        for url in urls:
            result = await is_image(url, session)
            print(f"{url} is an image: {result}")

# Example list of URLs
url_list = [
    'https://example.com/image1.jpg',
    'https://example.com/not_an_image.txt',
    'https://example.com/image2.png',
]

# Run the asynchronous code
loop = asyncio.get_event_loop()
loop.run_until_complete(main(url_list))



import streamlit as st

# URL or local path to your logo
logo_url = 'https://banner2.cleanpng.com/20181122/pye/kisspng-logo-fox-channel-fox-networks-group-latin-america-5bf69ff4ca5908.8354122815428894608288.jpg'

title_text = "My Awesome App"
# Custom CSS to inject for styling the title, removing default padding/margin, and styling the logo
custom_css = """
    <style>
        /* Remove padding and margin from the top of the page */
        .stApp {
            padding-top: 0px;
            margin-top: 0px;
        }

        /* Custom styling for the logo */
        .logo {
            width: 100px;  /* Adjust the size of your logo */
            height: auto;  /* Maintain aspect ratio */
            margin: 20px 0;  /* Space above and below the logo */
            display: block;
            margin-left: auto;
            margin-right: auto;
        }

        /* Custom styling for the title */
        .title {
            color: #ff6347;  /* Title color */
            font-family: 'Garamond', serif;  /* Font family */
            font-size: 48px;  /* Font size */
            font-weight: bold;  /* Font weight */
            text-align: center;  /* Align text to the center */
            margin: 10px 0;  /* Space above and below the title */
        }

        /* Reduce the padding around the markdown container for the title */
        .markdown-text-container {
            padding-top: 0px !important;
            padding-bottom: 0px !important;
        }

        /* Reduce the padding around the main block container */
        .block-container {
            padding-top: 0px;
            padding-left: 0px;
            padding-right: 0px;
            padding-bottom: 0px;
        }
    </style>
"""

# Inject custom CSS with markdown
st.markdown(custom_css, unsafe_allow_html=True)

# Display the logo
st.markdown(f'<img src="{logo_url}" class="logo">', unsafe_allow_html=True)

# Display the title with custom HTML and class for styling
st.markdown(f'<div class="title">{title_text}</div>', unsafe_allow_html=True)

# Continue with the rest of your Streamlit app
# ...


import shutil
import os

# List of filenames to be copied
filenames = [
    '/path/to/source/file1.txt',
    '/path/to/source/file2.jpg',
    '/path/to/source/file3.pdf',
    # Add more files as needed
]

# Destination folder where the files should be copied
destination_folder = '/path/to/destination/folder'

# Ensure the destination folder exists, create if it doesn't
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Copy each file to the destination folder
for filename in filenames:
    # Extract the basename (file name with extension) from the file path
    basename = os.path.basename(filename)
    # Create the destination path by joining the destination folder and basename
    destination_path = os.path.join(destination_folder, basename)
    # Copy the file
    shutil.copy(filename, destination_path)
    print(f'Copied {filename} to {destination_path}')

import streamlit as st
import os

# Assuming your WAV files are stored in a directory named 'wav_files' in the current working directory
base_dir = 'wav_files'

# List of speakers and categories
speakers = ['speaker1', 'speaker2', 'speaker3']
categories = ['category1', 'category2', 'category3']

for speaker in speakers:
    st.header(speaker)
    for category in categories:
        st.subheader(category)
        
        # Construct the path to the category folder
        category_path = os.path.join(base_dir, speaker, category)
        
        # List all WAV files in this category
        try:
            wav_files = [f for f in os.listdir(category_path) if f.endswith('.wav')]
        except FileNotFoundError:
            st.write(f"No files found for {category} in {speaker}.")
            continue
        
        # Display each WAV file with Streamlit's audio widget
        for wav_file in wav_files:
            st.audio(os.path.join(category_path, wav_file))


import streamlit as st
import os

# Assuming you have a directory named 'audio_files' structured as mentioned
BASE_DIR = 'audio_files'
SPEAKERS = ['speaker1', 'speaker2', 'speaker3']
CATEGORIES = ['category1', 'category2', 'category3']

def display_audio_files(speaker, category):
    """Display audio files for a given speaker and category."""
    # Construct the path to the category directory
    category_dir = os.path.join(BASE_DIR, speaker, category)
    
    # Get all .wav files in this directory
    try:
        files = [f for f in os.listdir(category_dir) if f.endswith('.wav')]
    except FileNotFoundError:
        st.error(f"Directory {category_dir} not found.")
        return
    
    # Display the files in 4 columns
    cols = st.columns(4)
    for index, file in enumerate(files):
        with cols[index % 4]:
            st.audio(os.path.join(category_dir, file))

def app():
    """Main function to display the Streamlit app."""
    st.title('Audio File Display')

    for speaker in SPEAKERS:
        st.header(f'Speaker: {speaker}')
        for category in CATEGORIES:
            st.subheader(f'Category: {category}')
            display_audio_files(speaker, category)

import plotly.graph_objects as go
from scipy.io import wavfile
import numpy as np

# Path to your WAV file
wav_file_path = 'path/to/your/audio.wav'

# Load the WAV file
sample_rate, data = wavfile.read(wav_file_path)

# If stereo, just pick one channel
if len(data.shape) > 1:
    data = data[:, 0]

# Calculate the time axis in seconds
time = np.linspace(0, len(data) / sample_rate, num=len(data))

# Create a plotly figure
fig = go.Figure()

# Add the waveform trace to the figure
fig.add_trace(go.Scatter(x=time, y=data, mode='lines', name='Waveform'))

# Update the layout
fig.update_layout(title='Waveform of the Audio File',
                  xaxis_title='Time (s)',
                  yaxis_title='Amplitude',
                  template='plotly_dark')

# Show the plot
fig.show()

import os
from spleeter.separator import Separator
import shutil
from tqdm import tqdm

def separate_vocals(input_folder, output_folder):
    # Create a separator object for 2 stems: vocals and accompaniment
    separator = Separator('spleeter:2stems')
    
    # Prepare a list of MP3 files for processing
    mp3_files = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.mp3'):
                mp3_files.append(os.path.join(root, file))
    
    # Process each file with progress tracking
    for file_path in tqdm(mp3_files, desc="Extracting vocals"):
        input_path = file_path
        output_path = os.path.join(file_path.replace(input_folder, output_folder), os.path.splitext(os.path.basename(file_path))[0])
        
        # Ensure the output directory exists
        os.makedirs(output_path, exist_ok=True)
        
        # Process the file
        separator.separate_to_file(input_path, output_path)
        
        # Move the vocals file to maintain the folder structure and change extension to .wav
        original_vocal_path = os.path.join(output_path, 'vocals.wav')
        target_vocal_path = original_vocal_path.replace(output_folder, output_folder) + ".wav"
        
        # Check if vocal file exists to avoid errors and move it
        if os.path.exists(original_vocal_path):
            shutil.move(original_vocal_path, target_vocal_path)
        
        # Optionally, clean up the accompaniment file and empty directories if needed
        shutil.rmtree(output_path)



import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image, ImageDraw
import numpy as np

def load_model():
    model_id = "google/owlvit-base-patch32"  # Choose appropriate model variant
    model = OwlViTForObjectDetection.from_pretrained(model_id)
    processor = OwlViTProcessor.from_pretrained(model_id)
    return model, processor

def iou(box1, box2):
    """Calculate Intersection Over Union (IOU) between two bounding boxes."""
    x1, y1, x2, y2 = box1
    x1_prime, y1_prime, x2_prime, y2_prime = box2

    xi1 = max(x1, x1_prime)
    yi1 = max(y1, y1_prime)
    xi2 = min(x2, x2_prime)
    yi2 = min(y2, y2_prime)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_prime - x2_prime) * (y2_prime - y1_prime)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area

def non_maximum_suppression(scores, boxes, iou_threshold):
    idxs = np.argsort(scores)
    pick = []

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]

        for pos in range(0, last):
            j = idxs[pos]
            if iou(boxes[i], boxes[j]) > iou_threshold:
                suppress.append(pos)

        idxs = np.delete(idxs, suppress)

    return pick

def predict(image_path, model, processor, threshold=0.9, iou_thresh=0.5):
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]

    # Apply Non-Maximum Suppression
    keep = non_maximum_suppression(results["scores"], results["boxes"], iou_thresh)
    results = {key: [value[i] for i in keep] for key, value in results.items()}
    return results

def display_results(image_path, results):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(b) for b in box.tolist()]
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0], box[1]), f"{label} {score:.2f}", fill="red")

    image.show()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python object_detection.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    model, processor = load_model()
    results = predict(image_path, model, processor)
    display_results(image_path, results)


# Import required libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Load data from Excel (replace 'transactions.xlsx' with your file)
file_path = 'transactions.xlsx'
df = pd.read_excel(file_path)

# Explore the data - identify columns and first few rows
print(df.head())
print(df.info())

# Clean and preprocess the data (customize according to your specific dataset)
# Example: Handle missing values
df.fillna(method='ffill', inplace=True)

# Convert date/time fields if necessary
df['transaction_date'] = pd.to_datetime(df['transaction_date'])

# Example statistical analysis (descriptive statistics)
print(df.describe())

# Define thresholds to mark potential fraudulent activities (custom criteria)
high_value_threshold = df['amount'].quantile(0.95)  # Transactions in top 5% by value

# Flag transactions greater than threshold
df['potential_fraud'] = df['amount'] > high_value_threshold

# Anomaly detection using Isolation Forest
clf = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
clf.fit(df[['amount']])
df['anomaly'] = clf.predict(df[['amount']])
df['anomaly'] = df['anomaly'].map({1: 'normal', -1: 'anomaly'})

# Visualization of anomalies
plt.figure(figsize=(10, 6))
plt.scatter(df.index, df['amount'], c=(df['anomaly'] == 'anomaly'), cmap='coolwarm', label='Anomalous Transactions')
plt.xlabel('Index')
plt.ylabel('Transaction Amount')
plt.legend()
plt.show()

# Document findings
# Filter out only flagged transactions
anomalous_transactions = df[df['anomaly'] == 'anomaly']
print(anomalous_transactions)

# Further investigation can be done by grouping data or cross-checking details

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Excel data into a pandas DataFrame
file_path = "employee_transactions.xlsx"
df = pd.read_excel(file_path)

# Inspect the first few rows and data types
print(df.head())
print(df.info())

# Step 2: Data Cleaning
# Check for missing values and duplicates
missing_values = df.isnull().sum()
duplicates = df.duplicated().sum()
print(f"Missing Values:\n{missing_values}")
print(f"Duplicated Rows: {duplicates}")

# Drop duplicates and handle missing values if necessary
df.drop_duplicates(inplace=True)
df.fillna(value={"transaction_amount": 0}, inplace=True)  # Example for handling missing numeric values

# Convert transaction date to datetime format if not already
df['transaction_date'] = pd.to_datetime(df['transaction_date'])

# Step 3: Exploratory Data Analysis (EDA)
# Describe numeric columns
print(df.describe())

# Aggregation: Total transaction amount per employee
employee_summary = df.groupby('employee_id')['transaction_amount'].sum()
print(employee_summary)

# Distribution of transaction amounts
sns.histplot(df['transaction_amount'], bins=20, kde=True)
plt.title('Distribution of Transaction Amounts')
plt.xlabel('Amount')
plt.ylabel('Frequency')
plt.show()

# Step 4: Identify Anomalies
# Example: Transactions above a certain threshold
threshold = 10000
high_value_transactions = df[df['transaction_amount'] > threshold]
print("High Value Transactions:\n", high_value_transactions)

# Example: Multiple transactions to the same vendor within a short time
df['transaction_date_diff'] = df.groupby('vendor_id')['transaction_date'].diff().dt.days
short_interval = df[(df['transaction_date_diff'] <= 2) & (df['transaction_date_diff'] > 0)]
print("Short Interval Transactions:\n", short_interval)

# Example: Transactions outside regular business hours
df['hour'] = df['transaction_date'].dt.hour
outside_hours = df[(df['hour'] < 8) | (df['hour'] > 18)]
print("Outside Business Hours Transactions:\n", outside_hours)

# Example: Benford's Law analysis
leading_digit = df['transaction_amount'].astype(str).str[0].astype(int)
benford_distribution = leading_digit.value_counts(normalize=True).sort_index()
expected_distribution = np.array([np.log10(1 + 1 / d) for d in range(1, 10)])
plt.bar(range(1, 10), benford_distribution, label='Observed')
plt.plot(range(1, 10), expected_distribution, marker='o', linestyle='--', color='red', label='Expected')
plt.title("Benford's Law Analysis")
plt.xlabel('Leading Digit')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Step 5: Document and Visualize Findings
# Save results of suspicious transactions to a new Excel file
suspicious_transactions = pd.concat([high_value_transactions, short_interval, outside_hours]).drop_duplicates()
suspicious_transactions.to_excel("suspicious_transactions.xlsx", index=False)

# Generate a summary report (in this case, simply printing to the console)
print("Summary Report:")
print(f"High Value Transactions: {len(high_value_transactions)}")
print(f"Short Interval Transactions: {len(short_interval)}")
print(f"Outside Business Hours Transactions: {len(outside_hours)}")


from opensearchpy import OpenSearch, RequestsHttpConnection
from requests.auth import HTTPBasicAuth
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

class OpenSearchClient:
    def __init__(self, host, port, username, password):
        self.client = OpenSearch(
            hosts=[{'host': host, 'port': port}],
            http_auth=HTTPBasicAuth(username, password),
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection
        )

    def fetch_all_documents(self, index_name):
        query = {
            "query": {
                "match_all": {}
            }
        }
        return self.client.search(index=index_name, body=query, scroll='1m', size=1000)

    def scroll_search(self, scroll_id):
        return self.client.scroll(scroll_id=scroll_id, scroll='1m')

    def bulk_delete_documents(self, index_name, doc_ids):
        actions = [{"delete": {"_index": index_name, "_id": doc_id}} for doc_id in doc_ids]
        return self.client.bulk(body=actions)

    def _calculate_hash(self, vector):
        vector_str = json.dumps(vector, sort_keys=True)
        return hashlib.sha256(vector_str.encode('utf-8')).hexdigest()

    def process_batch(self, index_name, hits, seen_hashes):
        to_delete = []
        for hit in hits:
            doc_id = hit['_id']
            vector = hit['_source'].get('vector')

            if vector is None:
                continue

            doc_hash = self._calculate_hash(vector)
            if doc_hash in seen_hashes:
                to_delete.append(doc_id)
            else:
                seen_hashes.add(doc_hash)

        if to_delete:
            self.bulk_delete_documents(index_name, to_delete)
        return len(to_delete)

    def remove_duplicates(self, index_name):
        response = self.fetch_all_documents(index_name)
        scroll_id = response['_scroll_id']
        hits = response['hits']['hits']

        seen_hashes = set()
        total_deleted = 0

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            while hits:
                futures.append(executor.submit(self.process_batch, index_name, hits, seen_hashes))

                response = self.scroll_search(scroll_id)
                scroll_id = response['_scroll_id']
                hits = response['hits']['hits']

            for future in as_completed(futures):
                total_deleted += future.result()

        return total_deleted

if __name__ == "__main__":
    host = 'your-opensearch-endpoint'
    port = 443
    username = 'your-username'
    password = 'your-password'

    client = OpenSearchClient(host, port, username, password)

    index_name = 'vector-index'
    deleted_count = client.remove_duplicates(index_name)
    print(f"Total duplicates removed: {deleted_count}")



if __name__ == "__main__":
    app()
