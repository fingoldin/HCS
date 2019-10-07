import json
import os
import sys
import glob
import pickle
import re

message_dir = "messages"
message_file = "message_1.json"
length = 10
max_word_len=20



if len(sys.argv) > 1:
  message_dir = sys.argv[1]

print("Searching directory '" + message_dir + "'...")

message_r = re.compile("^[a-zA-Z0-9\s]*$")

def extract_file_data(data):
  messages = []
  for message in data["messages"]:
    if "content" in message:
      content = message["content"]
      if not message_r.match(content) is None:
        mes = content.lower().split()
        valid=True
        for m in mes:
          if len(m) > max_word_len:
            valid=False
            break
        if valid and len(mes) >= length:
          messages.append(mes)
  
  return messages

invalid_files = 0
invalid_dirs = 0
empty_files = 0

all_threads = []
all_words = []

for thread in glob.glob(os.path.join(message_dir, "*")):
  try:
    fp = open(os.path.join(thread, message_file))

    try:
      data = json.load(fp)

      messages = extract_file_data(data)
      if len(messages):
        all_threads.append(messages)
        for m in messages:
          all_words.extend(m)

        print("Got conversation with " + str(len(messages)) + " messages")
      else:
        empty_files += 1
    except:
      invalid_files += 1
  except IOError:
    invalid_dirs +=1

if not len(all_threads) or not len(all_words):
  print("No data found")

print(str(invalid_dirs) + " invalid directories")
print(str(invalid_files) + " invalid files")
print(str(empty_files) + " empty files")

all_words.sort(key=len)

all_words = list(set(all_words))
all_words.sort()

#print("Found " + str(len(all_words)) + " words: " + str(all_words))

for t in range(len(all_threads)):
  for m in range(len(all_threads[t])):
    for l in range(len(all_threads[t][m])):
      all_threads[t][m][l] = all_words.index(all_threads[t][m][l])

    #all_threads[t][m].append(0) # END

all_messages = []
for t in all_threads:
  all_messages += t

for m in range(len(all_messages)):
  if len(all_messages[m]) < length:
    all_messages[m] += [0] * (length - len(all_messages[m]))
  elif len(all_messages[m]) > length:
    all_messages[m] = all_messages[m][0:length]
    all_messages[m][length-1] = 0

map_words = all_words

pickle.dump({ "map": map_words, "data": all_messages }, open("word_data.pickle", "wb"))
