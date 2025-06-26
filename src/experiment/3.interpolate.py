def interp(max_length, sample, interp_f_name, interp_omega_name, limit=0.85):
    current_len = len(sample['log10OmegaGW'])
    insert_num = max_length - current_len

    if insert_num == 0:
        return

    # Reverse f and log10OmegaGW to ascending order
    x = sample['f'][::-1]  # Reverse to ascending
    y = sample['log10OmegaGW'][::-1]  # Reverse to ascending
    # x = sample['f']
    # y = sample['log10OmegaGW']

    min_gap = float('inf')
    gap_index = 0

    # only consider former 85% points
    limit = int(len(x) * limit) if len(x) > 1 else 1

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

    # Reverse back to descending order for output consistency
    return {
        interp_f_name: f_interp[::-1],
        interp_omega_name: log10OmegaGW_interp[::-1]
    }
    # return {
    #     interp_f_name: f_interp,
    #     interp_omega_name: log10OmegaGW_interp
    # }


samples = [
    {'f': [-15, -13, -8, -2, 3], 'log10OmegaGW': [-19, -21, -15, -14, -10]},
    {'f': [-15, -6, -4], 'log10OmegaGW': [-20, -17, -8]},
    {'f': [-15, -10, -1, 5], 'log10OmegaGW': [-20, -17, -8, -6]}
]

if __name__ == "__main__":
    # to mongodb
    from bson import ObjectId
    from tqdm import tqdm
    import pymongo

    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client["solve_plus"]
    collection = db["data"]
    percent = 60
    interp_f = f"f_interp_{percent}"
    interp_omega = f"log10OmegaGW_interp_{percent}"

    for document in tqdm(collection.find({'log10OmegaGW': {'$exists': True}})):
        # interp_omega: {'$exists': False}
        set_doc = interp(256, document, interp_f, interp_omega, limit=percent/100)
        if set_doc is not None:
            collection.update_one({'_id': ObjectId(document['_id'])}, {'$set': set_doc}, upsert=True)
        else:
            collection.update_one({'_id': ObjectId(document['_id'])}, {'$set': {
                interp_f: document['f'],
                interp_omega: document['log10OmegaGW']
            }}, upsert=True)
