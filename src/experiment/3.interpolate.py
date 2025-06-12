def interp(max_length, sample, interp_f_name, interp_omega_name):
    current_len = len(sample['log10OmegaGW'])
    insert_num = max_length - current_len

    if insert_num == 0:
        return

    x = sample['f']
    y = sample['log10OmegaGW']
    min_gap = float('inf')
    gap_index = 0

    # only consider former 85% points
    limit = int(len(x) * 0.85) if len(x) > 1 else 1

    # find the pairs with min gap
    for i in range(limit - 1):
        gap = abs(x[i + 1] - x[i])
        if gap < min_gap:
            min_gap = gap
            gap_index = i

    # interpolate
    x0, x1 = x[gap_index], x[gap_index + 1]
    y0, y1 = y[gap_index], y[gap_index + 1]
    dx = (x1 - x0) / (insert_num + 1)
    dy = (y1 - y0) / (insert_num + 1)
    new_x = [x0 + i * dx for i in range(1, insert_num + 1)]
    new_y = [y0 + i * dy for i in range(1, insert_num + 1)]
    f_interp = x[:gap_index + 1] + new_x + x[gap_index + 1:]
    log10OmegaGW_interp = y[:gap_index + 1] + new_y + y[gap_index + 1:]

    return {
        interp_f_name: f_interp,
        interp_omega_name: log10OmegaGW_interp
    }


samples = [
    {'f': [-15, -13, -8, -2, 3], 'log10OmegaGW': [-19, -21, -15, -14, -10]},
    {'f': [-15, -6, -4], 'log10OmegaGW': [-20, -17, -8]},
    {'f': [-15, -10, -1, 5], 'log10OmegaGW': [-20, -17, -8, -6]}
]

if __name__ == "__main__":
    for i, sample in enumerate(samples, 1):
        print(
            f"Sample {i}: {interp(max(len(sample['log10OmegaGW']) for sample in samples), sample, 'f_interp_85', 'log10OmegaGW_interp_85')}")

    # from bson import ObjectId
    # from tqdm import tqdm
    # import pymongo
    #
    # client = pymongo.MongoClient("mongodb://localhost:27017/")
    # db = client["solve_plus"]
    # collection = db["data"]
    # interp_f_name = "f_interp_85"
    # interp_omega_name = "log10OmegaGW_interp_85"
    #
    # for document in tqdm(collection.find({'log10OmegaGW': {'$exists': True}, interp_omega_name: {'$exists': False}})):
    #     set_doc = interp(256, document, interp_f_name, interp_omega_name)
    #     if set_doc is not None:
    #         collection.update_one({'_id': ObjectId(document['_id'])}, {'$set': set_doc}, upsert=True)
    #     else:
    #         collection.update_one({'_id': ObjectId(document['_id'])}, {'$set': {
    #             interp_f_name: document['f'],
    #             interp_omega_name: document['log10OmegaGW']
    #         }}, upsert=True)