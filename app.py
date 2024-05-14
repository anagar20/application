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

class CSVFilterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CSV Filter and Export Tool")

        # Load CSV Button
        ttk.Button(self.root, text="Load CSV", command=self.load_csv).grid(row=0, column=0, padx=10, pady=10)

        # Frame for Date Checkboxes
        self.date_frame = ttk.LabelFrame(self.root, text="Select Dates")
        self.date_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky='ew')
        self.date_vars = {}

        # Frame for Column Checkboxes
        self.column_frame = ttk.LabelFrame(self.root, text="Select Columns")
        self.column_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky='ew')
        self.column_vars = {}

        # Submit Button for filtering data
        ttk.Button(self.root, text="Filter Data", command=self.filter_data).grid(row=3, column=0, columnspan=2, padx=10, pady=10)

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return
        self.df = pd.read_csv(file_path)

        # Process dates for checkboxes
        self.show_date_checkboxes()
        self.show_column_checkboxes()

    def show_date_checkboxes(self):
        # Clear previous checkboxes if any
        for widget in self.date_frame.winfo_children():
            widget.destroy()

        # Create checkboxes for each unique date
        unique_dates = sorted(self.df['Date'].unique())
        for date in unique_dates:
            var = tk.BooleanVar()
            chk = ttk.Checkbutton(self.date_frame, text=date, variable=var)
            chk.pack(anchor='w')
            self.date_vars[date] = var

    def show_column_checkboxes(self):
        # Clear previous checkboxes if any
        for widget in self.column_frame.winfo_children():
            widget.destroy()

        # Create new checkboxes for each column in the DataFrame
        for column in self.df.columns:
            var = tk.BooleanVar()
            chk = ttk.Checkbutton(self.column_frame, text=column, variable=var)
            chk.pack(anchor='w')
            self.column_vars[column] = var

    def filter_data(self):
        # Filter data based on selected dates
        selected_dates = [date for date, var in self.date_vars.items() if var.get()]
        mask = self.df['Date'].isin(selected_dates)
        filtered_data = self.df.loc[mask]

        # Filter data based on selected columns
        selected_columns = [col for col, var in self.column_vars.items() if var.get()]
        filtered_data = filtered_data[selected_columns]

        # Save the filtered DataFrame to a new CSV file
        output_file = filedialog.asksaveasfilename(filetypes=[("CSV files", "*.csv")], defaultextension=".csv")
        if output_file:
            filtered_data.to_csv(output_file, index=False)
            messagebox.showinfo("Success", "Filtered CSV has been saved.")

def main():
    root = tk.Tk()
    app = CSVFilterApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()


