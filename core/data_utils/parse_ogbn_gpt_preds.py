import os
import json
import re
import tqdm
import pandas as pd

# found here --> https://www.kaggle.com/datasets/dataup1/ogbn-products?resource=download
label_mapping_file = 'dataset/ogbn_products/labelidx2productcategory.csv'
assert os.path.exists(label_mapping_file), "Label mapping file not found, please download it from https://www.kaggle.com/datasets/dataup1/ogbn-products?resource=download and save it to `dataset/ogbn_products/labelidx2productcategory.csv`"
df = pd.read_csv(label_mapping_file)

mapping = {}
for i, row in df.iterrows():
    if type(row['product category']) != str:
        mapping['NAN'] = row['label idx']
    else:
        mapping[row['product category']] = row['label idx']

# Directory containing the JSON files
directory = 'gpt_responses/ogbn-products'

def extract_categories_in_order(response_content) -> list:
    categories_mentioned = []
    # Use regex to find each category in the response content
    for match in re.finditer(r'\b(' + '|'.join(map(re.escape, list(mapping.keys()))) + r')\b', response_content):
        category = match.group(1)
        categories_mentioned.append(mapping[category])
    return categories_mentioned

# List to store the parsed data
data = []

csv_filepath = 'gpt_preds/ogbn-products.csv'
with open(csv_filepath, 'w') as csv_file:
    # Iterate over all files in the directory
    for filename in tqdm.tqdm(range(len(os.listdir(directory)))):
        filepath = os.path.join(directory, f'{filename}.json')
            
        # Open and read the JSON file
        with open(filepath, 'r') as file:
            content = json.load(file)
            
            # Extract the response content
            response_content = content['choices'][0]['message']['content']
            
            # Extract categories mentioned in the response
            predicted_indices = extract_categories_in_order(response_content)
            if len(predicted_indices) == 0:
                # `-1` for no labels predicted
                # labels will shift by one --> https://github.com/XiaoxinHe/TAPE/issues/11#issuecomment-1851251728
                predicted_indices = [-1]
            
            # Append the filename and predicted indices to the data list
            csv_file.write(','.join(map(str, predicted_indices)) + '\n')

print(f"ogbn-products GPT preds saved to {csv_filepath}!")
