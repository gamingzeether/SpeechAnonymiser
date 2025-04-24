import os
import random

processedDir = os.environ.get('PROCESSED_DIR')
dataClientId = os.environ.get('DATA_CLIENT_ID')

def write_labels(file):
  file.write("client_id	path	sentence_id	sentence	sentence_domain	up_votes	down_votes	age	gender	accents	variant	locale	segment\n")

def write_line(file, filename):
  file.write(dataClientId + "\t") #client_id
  file.write(filename + "\t") #path
  file.write("blank" + "\t") #sentence_id
  file.write("blank" + "\t") #sentence
  file.write("blank" + "\t") #sentence_domain
  file.write("blank" + "\t") #up_votes
  file.write("blank" + "\t") #down_votes
  file.write("blank" + "\t") #age
  file.write("blank" + "\t") #gender
  file.write("blank" + "\t") #accents
  file.write("blank" + "\t") #variant
  file.write("blank" + "\t") #locale
  file.write("blank" + "\t") #segment
  file.write("\n")

train = open(processedDir + "/train.tsv", "w")
write_labels(train)
test = open(processedDir + "/test.tsv", "w")
write_labels(test)
dev = open(processedDir + "/dev.tsv", "w")
write_labels(dev)

for audioFile in os.listdir(processedDir + "/clips/"):
  targetFile = random.choices(
    [train, test, dev],
    weights = [10, 1, 1], k = 1)[0]
  write_line(targetFile, audioFile)
