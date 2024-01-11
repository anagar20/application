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



import flair
from flair.data import Sentence

# Load the NER tagger
tagger = flair.models.SequenceTagger.load('ner')

# Example text
text = """Google LLC is an American multinational technology company that specializes in Internet-related services and products.
          It was founded in 1998 by Larry Page and Sergey Brin while they were Ph.D. students at Stanford University in California."""

# Preprocessing (if needed)
# For example, removing extra spaces
clean_text = " ".join(text.split())

# Create a Sentence object
sentence = Sentence(clean_text)

# Predict entities
tagger.predict(sentence)

# Print the entities with their types
for entity in sentence.get_spans('ner'):
    print(entity)

# If you want to see more detailed information:
for entity in sentence.get_spans('ner'):
    print(f"Text: {entity.text}, Start: {entity.start_position}, End: {entity.end_position}, Type: {entity.get_label('ner').value}")

