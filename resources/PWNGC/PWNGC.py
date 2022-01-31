import ast

"""
Paper (added 500 new annotations): Completing the Princeton Annotated Gloss Corpus Project
https://github.com/own-pt/glosstag
https://github.com/cltl/pwgc
https://wordnetcode.princeton.edu/glosstag.shtml
"""


# Note: extend the PWNGC with the IBM paper of 2019 (not priority now,
# because I need to compare my results to other frameworks)


with open("C:/Users/HP/PycharmProjects/RSM4WSD/data/output/pwngc.txt", "r") as file:
    # pwngc = eval(file.readline())
    # pwngc = file.readline()
    for line in file:
        fields = line.split('\t')
        tokenized_sentence, labels = ast.literal_eval(fields[0]), ast.literal_eval(fields[1])
        print(tokenized_sentence, labels)
