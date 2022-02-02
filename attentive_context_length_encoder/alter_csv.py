import csv

train = "../data/test_transformer/train.csv"
validate = "../data/test_transformer/validate.csv"
test = "../data/test_transformer/test.csv"

def change_O(path, name):
    r = csv.reader(open(path))
    lines = list(r)
    for l in lines:
        if l[0][-1] == 'O':
            l[0][-1] = str([0.0, 0.0, 0.0, 0.0, 0.0])

    writer = csv.writer(open("../../data/test_transformer/{}.csv".format(name), 'w'))
    writer.writerows(lines)

# tst = change_O(test, "new_test")