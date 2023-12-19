import streamlit as st

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

if __name__ == "__main__":
    main()
