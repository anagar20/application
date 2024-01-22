import aiohttp
import asyncio
from aiohttp_retry import RetryClient, ExponentialRetry
from PIL import Image
from io import BytesIO

async def is_image(url, session):
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
        return False

async def main(urls):
    retry_options = ExponentialRetry(attempts=3)
    async with RetryClient(session=aiohttp.ClientSession(), retry_options=retry_options) as session:
        tasks = [is_image(url, session) for url in urls]
        results = await asyncio.gather(*tasks)
        return results

# Example list of URLs
url_list = [
    'https://example.com/image1.jpg',
    'https://example.com/not_an_image.txt',
    'https://example.com/image2.png',
]

# Run the asynchronous code
loop = asyncio.get_event_loop()
results = loop.run_until_complete(main(url_list))

# Display results
for url, result in zip(url_list, results):
    print(f"{url} is an image: {result}")


import aiohttp
import asyncio
from PIL import Image
from io import BytesIO
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt

async def is_image(url, session):
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
        return False

async def main(urls):
    async with aiohttp.ClientSession() as session:
        async def retry_on_exception(retry_state):
            return retry_if_exception_type(Exception)(retry_state)

        for url in urls:
            retrying = AsyncRetrying(
                retry_on_exception=retry_on_exception,
                stop=stop_after_attempt(3)
            )
            result = await retrying.call(is_image, url, session)
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

