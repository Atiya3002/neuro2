import numpy as np

sentence = "I love coding passionately"
words = sentence.split()
vocab = list(set(words))
vocab_size = len(vocab)
word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for i, word in enumerate(vocab)}
input_indices = [word_to_idx[word] for word in words[:-1]]
target_index = word_to_idx[words[-1]]

hidden_size = 10
learning_rate = 0.1
Wxh = np.random.randn(hidden_size, vocab_size) * 0.01
Whh = np.random.randn(hidden_size, hidden_size) * 0.01
Why = np.random.randn(vocab_size, hidden_size) * 0.01
bh = np.zeros((hidden_size, 1))
by = np.zeros((vocab_size, 1))
h_prev = np.zeros((hidden_size, 1))

def forward_pass(inputs, h_prev):
    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = np.copy(h_prev)
    for t in range(len(inputs)):
        xs[t] = np.zeros((vocab_size, 1))
        xs[t][inputs[t]] = 1
        hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh)
        ys[t] = np.dot(Why, hs[t]) + by
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
    return xs, hs, ps

def backward_pass(inputs, xs, hs, ps):
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dh_next = np.zeros_like(hs[0])
    t = len(inputs) - 1
    dy = np.copy(ps[t])
    dy[target_index] -= 1
    dWhy += np.dot(dy, hs[t].T)
    dby += dy
    dh = np.dot(Why.T, dy) + dh_next
    dhraw = (1 - hs[t] * hs[t]) * dh
    dbh += dhraw
    dWxh += np.dot(dhraw, xs[t].T)
    dWhh += np.dot(dhraw, hs[t-1].T)
    dh_next = np.dot(Whh.T, dhraw)
    return dWxh, dWhh, dWhy, dbh, dby

def train(inputs, target, h_prev, num_epochs=100):
    global Wxh, Whh, Why, bh, by
    for epoch in range(num_epochs):
        xs, hs, ps = forward_pass(inputs, h_prev)
        dWxh, dWhh, dWhy, dbh, dby = backward_pass(inputs, xs, hs, ps)
        for param, dparam in zip([Wxh, Whh, Why, bh, by], [dWxh, dWhh, dWhy, dbh, dby]):
            param -= learning_rate * dparam
        if epoch % 10 == 0:
            loss = -np.log(ps[len(inputs)-1][target])
            print(f"Epoch {epoch}, Loss: {loss}")
    _, _, ps = forward_pass(inputs, h_prev)
    predicted_index = np.argmax(ps[len(inputs)-1])
    predicted_word = idx_to_word[predicted_index]
    print(f"\nFinal prediction: '{predicted_word}'")
    print(f"Actual word: '{words[-1]}'")
    print(f"Confidence: {ps[len(inputs)-1][target][0]*100:.2f}%")

train(input_indices, target_index, h_prev)