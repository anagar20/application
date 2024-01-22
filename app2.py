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
