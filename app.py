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





from opensearchpy import OpenSearch, exceptions

# Configuration: replace with your server details
host = 'https://your-opensearch-cluster-endpoint'
port = 443  # or another port if you use one
auth = ('username', 'password')  # HTTP basic authentication

# Initialize the OpenSearch client
client = OpenSearch(
    hosts=[{'host': host, 'port': port}],
    http_compress=True,  # enables gzip compression for request bodies
    http_auth=auth,
    use_ssl=True,
    verify_certs=True,
    ssl_assert_hostname=False,
    ssl_show_warn=False,
)

# Verify if index exists
index_name = 'your_index_name'
if not client.indices.exists(index=index_name):
    print("Index named '{}' does not exist.".format(index_name))
else:
    # Define the query
    query = {
        'query': {
            'match': {
                'text_field_name': 'search text'
            }
        }
    }

    # Perform the search
    response = client.search(index=index_name, body=query)
    total_hits = response['hits']['total']['value']
    print("Total records matching the query:", total_hits)

    # If there are hits, delete them
    if total_hits > 0:
        try:
            delete_response = client.delete_by_query(index=index_name, body=query, refresh=True)
            print("Deleted records count:", delete_response['deleted'])
        except exceptions.NotFoundError:
            print("Error: The resource was not found.")
        except Exception as e:
            print("An error occurred:", str(e))
    else:
        print("No records to delete.")

