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

if __name__ == "__main__":
    app()
