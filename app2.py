import asyncio
import time

async def func_a():
    st = time.time()
    print("Function A is running")
    await asyncio.sleep(1)
    print("Function A is done")
    et = time.time()
    print('time taken for A is ', et-st)
    return "Function A"

async def func_b():
    st = time.time()
    print("Function B is running")
    await asyncio.sleep(0.5)
    print("Function B is done")
    et = time.time()
    print('time taken for B is ', et - st)
    return "Function B"

async def main():
    print("Main function is running")
    st = time.time()
    results = await asyncio.gather(func_a(), func_b())
    et = time.time()
    print('time taken for MAIN is ', et - st)
    print("Main function is done")
    #print(results)
    return results


loop = asyncio.get_event_loop()
task = loop.create_task(main())
# Wait for the task to complete (optional)
rs = loop.run_until_complete(task)
print('rs', rs)



import torch

# Assuming you have 3 models: Model1, Model2, Model3

model1 = Model1()  # Initialize model 1
model2 = Model2()  # Initialize model 2
model3 = Model3()  # Initialize model 3

# Load the state dictionaries (if applicable)
model1.load_state_dict(state_dict1)
model2.load_state_dict(state_dict2)
model3.load_state_dict(state_dict3)

# Check if a GPU is available and if so, use it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to print current GPU memory usage
def print_gpu_memory():
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)  # GB
    print(f"GPU Memory Allocated: {allocated:.2f} GB")

# Move the models to GPU and print memory usage
model1.to(device)
print_gpu_memory()

model2.to(device)
print_gpu_memory()

model3.to(device)
print_gpu_memory()

# Now all three models are loaded into the GPU
# You can perform inference or further training as needed

# Clear cache (if needed)
torch.cuda.empty_cache()



import requests
from tenacity import retry, stop_after_attempt, wait_exponential

class APIWrapper:
    def __init__(self, base_url):
        self.base_url = base_url

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, max=10))
    def make_request(self, endpoint, method='get', **kwargs):
        url = self.base_url + endpoint
        response = requests.request(method, url, **kwargs)

        # You can implement your logic to handle response here
        response.raise_for_status()

        return response

# Usage
api = APIWrapper('https://api.example.com/')
try:
    response = api.make_request('/data', params={'param1': 'value'})
    print(response.json())
except requests.exceptions.HTTPError as e:
    print(f"Request failed: {e}")


