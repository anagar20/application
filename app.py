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





import logging
from opensearchpy import OpenSearch, exceptions

class OpenSearchIndexManager:
    def __init__(self, host, port, username, password, index_name):
        self.host = host
        self.port = port
        self.auth = (username, password)
        self.index_name = index_name
        self.client = self._create_client()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _create_client(self):
        return OpenSearch(
            hosts=[{'host': self.host, 'port': self.port}],
            http_auth=self.auth,
            use_ssl=True,
            verify_certs=True,
            ssl_assert_hostname=False,
            ssl_show_warn=False,
            http_compress=True
        )

    def create_index(self, index_body):
        if self.client.indices.exists(index=self.index_name):
            self.logger.info(f"Index {self.index_name} already exists.")
            return
        try:
            response = self.client.indices.create(index=self.index_name, body=index_body)
            self.logger.info("Index created successfully.")
            self.logger.debug(f"Response: {response}")
        except exceptions.RequestError as e:
            self.logger.error(f"Request Error: {e.info}")
        except exceptions.ConnectionError as e:
            self.logger.error(f"Connection Error: {e.info}")
        except Exception as e:
            self.logger.error(f"An unexpected error occurred: {str(e)}")

    def delete_index(self):
        try:
            response = self.client.indices.delete(index=self.index_name)
            self.logger.info("Index deleted successfully.")
            self.logger.debug(f"Response: {response}")
        except exceptions.NotFoundError:
            self.logger.error("Index not found.")
        except exceptions.RequestError as e:
            self.logger.error(f"Request Error: {e.info}")
        except Exception as e:
            self.logger.error(f"An unexpected error occurred: {str(e)}")

    def query_index(self, query):
        try:
            response = self.client.search(index=self.index_name, body={"query": query})
            self.logger.info(f"Query executed successfully.")
            return response
        except exceptions.RequestError as e:
            self.logger.error(f"Request Error: {e.info}")
        except exceptions.ConnectionError as e:
            self.logger.error(f"Connection Error: {e.info}")
        except Exception as e:
            self.logger.error(f"An unexpected error occurred: {str(e)}")
            return None

    def delete_documents_by_query(self, query):
        try:
            response = self.client.delete_by_query(index=self.index_name, body={"query": query}, refresh=True)
            self.logger.info(f"Documents deleted successfully. Total deleted: {response['deleted']}")
        except exceptions.RequestError as e:
            self.logger.error(f"Request Error: {e.info}")
        except exceptions.ConnectionError as e:
            self.logger.error(f"Connection Error: {e.info}")
        except Exception as e:
            self.logger.error(f"An unexpected error occurred: {str(e)}")

    def get_opensearch_version(self):
        try:
            info = self.client.info()
            version = info['version']['number']
            self.logger.info(f"OpenSearch version: {version}")
            return version
        except Exception as e:
            self.logger.error(f"Failed to retrieve OpenSearch version: {str(e)}")
            return None

# Example usage of the class:
if __name__ == "__main__":
    index_manager = OpenSearchIndexManager(
        host='https://your-opensearch-cluster-endpoint',
        port=443,
        username='your_username',
        password='your_password',
        index_name='your_index_name'
    )

    # Assuming index is already created, here's how you could query and delete documents:
    query = {"match": {"face_string": "example"}}
    # Query the index
    search_results = index_manager.query_index(query)
    print(search_results)

    # Delete documents matching the query
    index_manager.delete_documents_by_query(query)
