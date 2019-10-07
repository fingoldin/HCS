## RNN Facebook Message Analyzer

### Goal

I attempted to create a Recurrent Neural Network model to predict phrases said by me and my friends in Facebook messenger. I wanted to create a model that could generate sentences similar to what we would write, starting from a short prompt

### Model

I used a time-distributed fully connected layer followed by simple Recurrent Neural Network unit, with a fully connected layer size of 100 and a hidden state of size 150 (about ~200,000 total parameters). I then trained the recurrent unit to output the next word given a series of previous words in a phrase. For data, I gathered all messages with at least words, and without words longer than 20 characters. The model itself accepted and outputted one-hot encodings. Everything was done with vanilla Tensorflow, so I wrote out all the matrix multiplications by hand.

### Files

`data.py` formats the data and stores it as a list of lists in a pickle file.
`model.py` trains the model and forms predictions

### Dependencies

- Python3
- Tensorflow
- Numpy

### Results

The model outputted many funny and strange phrases, such as
- "hello my name is programming"
- "i love hot haikus"
