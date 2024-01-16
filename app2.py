import streamlit as st
import random

# Dictionary with keys and lists of names
data_dict = {
    "Group 1": ["Alice", "Bob", "Charlie"],
    "Group 2": ["David", "Eve"],
    "Group 3": ["Frank", "Grace"],
}

# Define custom CSS for the tag clouds
css = """
<style>
    .tag-cloud {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    }
    .tag-pill {
        padding: 6px 12px;
        background-color: #fff;
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

# Create a dictionary to map keys to text colors
text_color_dict = {key: "#{:02x}{:02x}{:02x}".format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for key in data_dict.keys()}

# Display tag clouds for each group with different text colors
for group, names in data_dict.items():
    text_color = text_color_dict[group]
    st.markdown(f'<strong style="color: {text_color};">{group}:</strong>', unsafe_allow_html=True)
    tag_cloud_html = css + '<div class="tag-cloud">'
    for name in names:
        tag_cloud_html += f'<span class="tag-pill" style="color: {text_color};">{name}</span>'
    tag_cloud_html += '</div>'
    st.markdown(tag_cloud_html, unsafe_allow_html=True)
