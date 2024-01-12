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

import boto3

def delete_folder(bucket_name, folder_path):
    """
    Delete a folder in an S3 bucket

    :param bucket_name: Name of the S3 bucket
    :param folder_path: Path of the folder in the S3 bucket
    """
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)
    folder_path = folder_path if folder_path.endswith('/') else folder_path + '/'

    # List and delete objects
    objects_to_delete = bucket.objects.filter(Prefix=folder_path)
    delete_keys = {'Objects': [{'Key': obj.key} for obj in objects_to_delete]}

    if delete_keys['Objects']:
        bucket.delete_objects(Delete=delete_keys)
        print(f"Deleted folder {folder_path} from bucket {bucket_name}")
    else:
        print(f"No objects found in {folder_path}")

# Example usage
delete_folder('your-s3-bucket', 'path/to/folder')
