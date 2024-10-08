import json

def get_label_mapping(file):
    with open(file, 'r') as f:
        cat_to_name = json.load(f)

    cat_to_name = {int(k): v for k, v in cat_to_name.items()}
    return cat_to_name

