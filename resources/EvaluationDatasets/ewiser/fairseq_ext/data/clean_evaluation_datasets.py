import os
import torch

"__author__ == Siba Mohsen"

TESTING_PATH = "C:/Users/HP/PycharmProjects/RSM4WSD/data/testing_datasets"
full_sent = os.path.join(TESTING_PATH, 'full_sentences/')
names = ["senseval2", "senseval3", "semeval2007", "semeval2013", "semeval2015"] #, "ALL"]
datasets = ["senseval2.pt", "senseval3.pt", "semeval2007.pt", "semeval2013.pt", "semeval2015.pt"] #, "ALL.pt"]
full_sents = ["sent_senseval2.pt", "sent_senseval3.pt", "sent_semeval2007.pt", "sent_semeval2013.pt", "sent_semeval2015.pt"] #, "sent_ALL.pt"]

new_datasets = []
new_fullsentences = []
new_keys_full = []
for i in range(len(names)):
    # initialization
    test_dataset_name = names[i]
    test_dataset = datasets[i]
    test_full_sent = full_sents[i]

    initial_testset = torch.load(os.path.join(TESTING_PATH, test_dataset))

    full_sentences = torch.load(os.path.join(full_sent, test_full_sent))

    # make sure that the testing sets do not contain empty fields
    # indices to be dropped
    drop_idx = []  # those indices are strings (can be used for, e.g., senseval3.pt)
    drop_idx_int = []  # those indices are int (can be use for full sentences)
    for idx, sentence in enumerate(initial_testset):
        if sentence == []:
            drop_idx.append(idx)
            drop_idx_int.append(int(idx))

    for idx in reversed(drop_idx):
        del initial_testset[idx]
        del full_sentences[idx]

    new_datasets.append(initial_testset)
    new_fullsentences.append(full_sentences)
    new_keys_fullsent = new_keys_full.append({k: val for k, val in enumerate(list(full_sentences.values()))})

