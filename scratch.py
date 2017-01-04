import csv
import os

import pycrfsuite

#f= open("0001.csv")

parentElement = []
labels = []



filedir="/Users/lakshayarya/Desktop/NLP Ass-3/labeled data"

#filedir = os.path.join("/Users/lakshayarya/Desktop/NLP Ass-3/","labeled data")
for root,dirs,files in os.walk(filedir):
    for file in files:
       if file.endswith(".csv"):
           f = open(root+"//"+file, "r");
           csv_f = csv.reader(f)
           flag = 0
           lastValue = 'C'
           next(csv_f)
           for row in csv_f:
               element = []
               if (flag == 0):
                   element.append('0')
               else:
                   element.append('1')
               flag = 1

               if lastValue == row[1]:
                   element.append('0')
               else:
                   element.append('1')
               lastValue = row[1]

               token = ""
               pos = ""
               for word in row[2].split():
                   temp = word.split("/")
                   token = "TOKEN_" + temp[0]
                   element.append(token)
                   pos = "POS_" + temp[1]
                   element.append(pos)




#print(parentElement)
#print(labels)

trainer = pycrfsuite.Trainer(verbose=False)

# for xseq, yseq in zip(parentElement, labels):
#     trainer.append(xseq, yseq)

trainer.append(parentElement, labels)

trainer.set_params({
    'c1': 1.0,   # coefficient for L1 penalty
    'c2': 1e-3,  # coefficient for L2 penalty
    'max_iterations': 50,  # stop earlier

    # include transitions that are possible, but not observed
    'feature.possible_transitions': True
})

trainer.params()

trainer.train('model.crf')

trainer.logparser.last_iteration

tagger = pycrfsuite.Tagger()
tagger.open('model.crf')

pred_labels =  tagger.tag(parentElement)

# a= 0
# e=0
# for i in range(len(pred_labels)):
#     if pred_labels[i] == labels[i]:
#         a+=1
#     else:
#         e+=1
#
# print a,e

# example_sent = test_sents[0]
# print(' '.join(sent2tokens(example_sent)), end='\n\n')
#
# print("Predicted:", ' '.join(tagger.tag(sent2features(example_sent))))
# print("Correct:  ", ' '.join(sent2labels(example_sent)))