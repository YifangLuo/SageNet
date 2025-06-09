from pymongo import MongoClient
from bson import ObjectId
import json
from tqdm import tqdm

client = MongoClient('mongodb://localhost:27017/')
db = client['solve_plus']  # Replace with your actual database name
collection = db['data']  # Replace with your actual collection name

if __name__ == "__main__":
    for volume in tqdm(range(1,38)):
        filename = f'../solve/output/processed_data_{volume}.json'
        try:
            with open(filename, 'r') as f:
                data = json.load(f)

            for doc in data:
                if '_id' in doc:
                    doc['_id'] = ObjectId(doc['_id']['$oid'])

            if data:
                collection.insert_many(data)
                print(f"Imported {len(data)} documents from {filename} into MongoDB")
            else:
                print(f"No documents found in {filename}")
        except FileNotFoundError:
            continue