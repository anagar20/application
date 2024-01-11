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



from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging

# Load the NER model
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/ner-model-2020.02.10.tar.gz")

# Example text
text = """Google LLC is an American multinational technology company that specializes in Internet-related services and products.
          It was founded in 1998 by Larry Page and Sergey Brin while they were Ph.D. students at Stanford University in California."""

# Preprocess the text (if necessary)
# Example: Basic whitespace cleaning
clean_text = ' '.join(text.split())

# Use the model to predict entities
predictions = predictor.predict(sentence=clean_text)

# Extract and print entities
for word, tag in zip(predictions['words'], predictions['tags']):
    if tag != 'O':  # 'O' tags are for words that aren't named entities
        print(f"{word}: {tag}")


import stanza

# Download the English model
stanza.download('en')

# Create a Stanza pipeline with the NER processor
nlp = stanza.Pipeline(lang='en', processors='tokenize,ner')

# Example text
text = """Google LLC is an American multinational technology company that specializes in Internet-related services and products.
          It was founded in 1998 by Larry Page and Sergey Brin while they were Ph.D. students at Stanford University in California."""

# Preprocess the text (if necessary)
# Example: Basic whitespace cleaning
clean_text = ' '.join(text.split())

# Process the text
doc = nlp(clean_text)

# Extract and print entities
for ent in doc.ents:
    print(f"{ent.text}: {ent.type}")

# More detailed information
for i, sentence in enumerate(doc.sentences):
    for ent in sentence.ents:
        print(f"Sentence {i+1}, Word: {ent.text}, Type: {ent.type}")
