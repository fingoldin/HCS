## RNN Facebook Message Analyzer
Vassilios Kaxiras

### Goal

I attempted to create a Recurrent Neural Network model to predict phrases said by me and my friends in Facebook messenger. I wanted to create a model that could generate sentences similar to what we would write, starting from a short prompt.

### Model

I used a time-distributed fully connected layer followed by a simple Recurrent Neural Network unit, with a fully connected layer size of 100 and a hidden state of size 150 (about ~200,000 total parameters). I then trained the recurrent unit to output the next word given a series of previous words in a phrase. For data, I gathered all messages with at least 10 words, and without words longer than 20 characters. The model itself accepted and outputted one-hot encodings. Everything was done with vanilla Tensorflow, so I wrote out all the matrix multiplications by hand.

### Files

`data.py` formats the data and stores it as a list of lists in a pickle file.
`model.py` trains the model and forms predictions.

### Dependencies

- Python3
- Tensorflow
- Numpy

### Results

The model outputted many funny and strange phrases, such as
- "hello my name is programming"
- "i love hot haikus"
- "i get junk built"
- "can graduated boys be roommates"

When I gave it "i love" as a prompt, it consistently outputted "i love attention...". Perhaps it's trying to tell me something?
