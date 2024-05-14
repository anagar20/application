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






import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import os

def load_csv():
    # Manually specify the CSV file path
    file_path = "path/to/your/data.csv"
    df = pd.read_csv(file_path)
    return df

def create_scrollable_frame(parent):
    # Create a canvas with a scrollbar
    canvas = tk.Canvas(parent)
    scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    return scrollable_frame

def create_date_checkboxes(df, frame):
    # Create checkboxes for each unique date
    unique_dates = sorted(df['Date'].unique())
    date_vars = {}
    for date in unique_dates:
        var = tk.BooleanVar()
        chk = ttk.Checkbutton(frame, text=date, variable=var)
        chk.pack(anchor='w')
        date_vars[date] = var
    return date_vars

def create_column_checkboxes(df, frame):
    # Create new checkboxes for each column in the DataFrame
    column_vars = {}
    for column in df.columns:
        var = tk.BooleanVar()
        chk = ttk.Checkbutton(frame, text=column, variable=var)
        chk.pack(anchor='w')
        column_vars[column] = var
    return column_vars

def filter_data(df, date_vars, column_vars, root):
    # Filter data based on selected dates
    selected_dates = [date for date, var in date_vars.items() if var.get()]
    mask = df['Date'].isin(selected_dates)
    filtered_data = df.loc[mask]

    # Filter data based on selected columns
    selected_columns = [col for col, var in column_vars.items() if var.get()]
    filtered_data = filtered_data[selected_columns]

    # Automatically save the filtered DataFrame to a new CSV file
    output_file = "filtered_data.csv"
    filtered_data.to_csv(output_file, index=False)
    messagebox.showinfo("Success", f"Filtered CSV has been saved as {output_file}.")
    root.destroy()

def setup_gui(df):
    root = tk.Tk()
    root.title("CSV Filter and Export Tool")

    # Scrollable Frame for Date Checkboxes
    date_frame = ttk.LabelFrame(root, text="Select Dates")
    date_frame.grid(row=0, column=0, padx=10, pady=10, sticky='ew')
    scrollable_date_frame = create_scrollable_frame(date_frame)

    # Scrollable Frame for Column Checkboxes
    column_frame = ttk.LabelFrame(root, text="Select Columns")
    column_frame.grid(row=1, column=0, padx=10, pady=10, sticky='ew')
    scrollable_column_frame = create_scrollable_frame(column_frame)

    # Load data and create checkboxes
    date_vars = create_date_checkboxes(df, scrollable_date_frame)
    column_vars = create_column_checkboxes(df, scrollable_column_frame)

    # Submit Button for filtering data
    ttk.Button(root, text="Filter Data", command=lambda: filter_data(df, date_vars, column_vars, root)).grid(row=2, column=0, padx=10, pady=10)

    root.mainloop()

if __name__ == "__main__":
    df = load_csv()
    setup_gui(df)



