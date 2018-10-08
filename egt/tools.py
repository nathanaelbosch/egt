import pickle


def save_data(data, path, hist_freq=10):
    data['history_frequency'] = hist_freq
    data['history'] = data['history'][::(hist_freq//10)]
    with open(path, 'wb') as file:
        pickle.dump(data, file)


def load_data(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    if 'history_frequency' not in data.keys():
        data['history_frequency'] = 1
    return data
