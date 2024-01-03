import streamlit as st

# Define custom CSS for the tag cloud
css = """
<style>
    .tag-cloud {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    }
    .tag-pill {
        padding: 6px 12px;
        background-color: #f0f0f0;
        color: #333;
        border: 1px solid #ccc;
        border-radius: 20px;
        cursor: pointer;
        transition: background-color 0.3s, color 0.3s, transform 0.3s;
        font-family: 'Arial', sans-serif;
        font-weight: bold;
    }
    .tag-pill:hover {
        background-color: #007BFF;
        color: white;
        transform: scale(1.05);
    }
</style>
"""

col1, col2 = st.columns([1,1])

tag_cloud_html = css + '<div class="tag-cloud">'

# Display selected names as a tag cloud
if selected_names:
    for name in selected_names:
        tag_cloud_html += f'<span class="tag-pill">{name}</span>'

tag_cloud_html += '</div>'
col1.markdown(tag_cloud_html, unsafe_allow_html=True)

col2.markdown(tag_cloud_html, unsafe_allow_html=True)

# Function to display clickable images and buttons
def display_images(image_data):
    if image_data:
        # Row for images
        image_cols = st.columns(len(image_data))
        # Row for buttons
        button_cols = st.columns(len(image_data))

        for i, (image_url, caption, target_url) in enumerate(image_data):
            button_key = f"button_{i}"

            # Display each image in its column
            image_cols[i].image(image_url, caption=caption, use_column_width=True)

            # Display corresponding button below the image
            if button_cols[i].button(f"Open {caption}", key=button_key):
                st.sidebar.success(f"Image {i+1} clicked! Opening {target_url}")

# Streamlit app
def main():
    st.title("Dynamic Image Gallery App")

    if 'image_data' not in st.session_state:
        st.session_state.image_data = []

    user_input = st.text_input("Enter some text:", "")

    # Always display the images and buttons
    display_images(st.session_state.image_data)

    if user_input:
        # Example placeholder data - replace this with your actual data
        new_image_data = [
            ("https://example.com/new_image1.jpg", "New Caption 1", "https://example.com/new_target1"),
            ("https://example.com/new_image2.jpg", "New Caption 2", "https://example.com/new_target2"),
            ("https://example.com/new_image3.jpg", "New Caption 3", "https://example.com/new_target3"),
            ("https://example.com/new_image4.jpg", "New Caption 4", "https://example.com/new_target4"),
            ("https://example.com/new_image5.jpg", "New Caption 5", "https://example.com/new_target5"),
        ]

        if st.button("Update Image Data"):
            st.session_state.image_data = new_image_data
            st.success("Image Data Updated!")

    if st.button("Refresh Images"):
        st.success("Images refreshed!")


import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def fetch_url(url_index_tuple):
    """
    Function to perform a GET request to the specified URL.
    Returns a tuple of (index, response).
    """
    index, url = url_index_tuple
    try:
        response = requests.get(url)
        return index, response.text
    except requests.RequestException as e:
        return index, str(e)

def concurrent_requests(urls, max_workers=5):
    """
    Makes concurrent API requests to a list of URLs and monitors progress with tqdm.
    Preserves the order of the responses according to the order of URLs.
    
    Parameters:
    urls (list): List of URLs to make requests to.
    max_workers (int): Maximum number of threads to use for making requests.
    """
    results = [None] * len(urls)  # Pre-allocate a list to hold the results in order

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Associate each URL with its index
        future_to_index = {executor.submit(fetch_url, (index, url)): index for index, url in enumerate(urls)}

        for future in tqdm(as_completed(future_to_index), total=len(urls), desc="Fetching URLs"):
            index = future_to_index[future]
            try:
                _, data = future.result()
                results[index] = data
            except Exception as exc:
                print(f'URL at index {index} generated an exception: {exc}')

    return results

# Example usage
urls = ["https://jsonplaceholder.typicode.com/posts/1", "https://jsonplaceholder.typicode.com/posts/2", ...]  # Add more URLs as needed
responses = concurrent_requests(urls)



if __name__ == "__main__":
    main()
