import csv

# To prepare the  PWNGC dataset to the training process, remove all untagged words from the sentences.
# Those words have a spatial parameter "O"

read_file = "C:/Users/HP/PycharmProjects/RSM4WSD/data/training_datasets/pwngc4torchtext.csv"
write_file = "C:/Users/HP/PycharmProjects/RSM4WSD/data/training_datasets/pwngc4regressor.csv"
with open(read_file, "r") as infile:
    lines = infile.readlines()
with open(write_file, "w", newline='') as outfile:
    writer = csv.writer(outfile, delimiter="\t")
    for line in lines:
        print(line.split())
        try:
            if line.split()[-1] != 'O':
                #outfile.write(line)
                writer.writerow(line.split())
        except:
            #if line.split() == []:
            # outfile.write('')
            writer.writerow('')
