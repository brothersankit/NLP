#!/usr/bin/env python
# coding: utf-8

# In[3]:


import nltk
from nltk import FreqDist
from nltk.probability import LidstoneProbDist
from tabulate import tabulate


class HMMNER:
    """
    Class representing a Hidden Markov Model (HMM) for Named Entity Recognition (NER).
    """

    def __init__(self, order=1, smoothing=0.1):
        """
        Initializes the model with the specified order (unigram, bigram, or trigram) and smoothing parameter.

        Args:
            order (int, optional): The order of the Markov assumption (1 for bigram, 2 for trigram). Defaults to 1.
            smoothing (float, optional): The smoothing parameter for Lidstone smoothing. Defaults to 0.1.
        """
        self.order = order
        self.states = set()
        self.pi = {}  # Start probability distribution
        self.A = {}  # Transition probability distribution
        self.B = {}  # Emission probability distribution
        self.smoothing = smoothing

    def train(self, dataset):
        """
        Trains the HMM model on the provided dataset.

        Args:
            dataset (list): A list of sentences, where each sentence is a list of tuples (word, tag).
        """

        # Step 1: Find states (unique tags)
        self.states = set(tag for sent in dataset for _, tag in sent)

        # Step 2: Calculate start probability (pi)
        start_tags = [sent[0][1] for sent in dataset]
        self.pi = FreqDist(start_tags)

        # Step 3: Calculate transition probability (A)
        transition_counts = dict()
        for sent in dataset:
            for i in range(len(sent) - self.order):
                context = tuple(sent[j][1] for j in range(i, i + self.order))
                if len(context) == self.order:
                    transition_counts.setdefault(context, FreqDist())[sent[i + self.order][1]] += 1

        self.A = {
    context: LidstoneProbDist(fd, gamma=self.smoothing, bins=max(len(fd), 100))  # Update here
    for context, fd in transition_counts.items()
}


        # Step 4: Calculate emission probability (B)
        emission_counts = {tag: FreqDist() for tag in self.states}
        for sent in dataset:
            for word, tag in sent:
                emission_counts[tag][word] += 1

        self.B = {
            tag: LidstoneProbDist(fd, gamma=self.smoothing, bins=max(len(fd), max(fd.values()) + 1))
            for tag, fd in emission_counts.items()
        }

    def predict(self, sequence):
        """
        Predicts the most likely tag sequence for a given word sequence.

        Args:
            sequence (list): A list of words for which to predict the tags.

        Returns:
            list: A list of predicted tags for each word in the sequence.
        """

        predicted_tags = []
        for i in range(len(sequence)):
            context = tuple(predicted_tags[-self.order:]) if self.order > 0 else ()
            word = sequence[i]

            # Handle empty context
            if context not in self.A:
                predicted_tag = max(
                    self.states,
                    key=lambda tag: self.pi[tag] * self.B[tag].prob(word),
                )
            else:
                predicted_tag = max(
                    self.states,
                    key=lambda tag: self.A[context].prob(tag) * self.B[tag].prob(word),
                )
            predicted_tags.append(predicted_tag)

        return predicted_tags

    def _generate_contexts(self, dataset):
        """
        Generates possible contexts (sequences of tags) based on the model's order.
        """

        contexts = set()
        for sent in dataset:
            for i in range(len(sent) - self.order):
                context = tuple(sent[j][1] for j in range(i, i + self.order))
                if len(context) == self.order:
                    contexts.add(context)
        return contexts


def preprocess_dataset(file_path):
    """
    Preprocesses the dataset by reading a tab-delimited file containing word-tag pairs,
    separating words and tags, and constructing a list of sentences.

    Args:
        file_path (str): The path to the dataset file.

    Returns:
        list: A list of sentences, where each sentence is a list of tuples (word, tag).
    """

    preprocessed_dataset = []
    with open(file_path, 'r') as file:
        current_sentence = []
        for line in file:
            # Skip empty lines
            if not line.strip():
                continue

            word, tag = line.strip().split("\t")  # Split by tab delimiter
            current_sentence.append((word, tag))

        # Add the last sentence if it's not empty
        if current_sentence:
            preprocessed_dataset.append(current_sentence)

    return preprocessed_dataset

    
def evaluate_model(model, dataset):
    """
    Evaluates the performance of the model on the provided dataset.

    Args:
        model (HMMNER): The HMM model to evaluate.
        dataset (list): A list of sentences, where each sentence is a list of tuples (word, tag).

    Returns:
        dict: A dictionary containing precision, recall, and F1 scores for each tag and the overall performance.
    """

    true_positives = {tag: 0 for tag in model.states}
    false_positives = {tag: 0 for tag in model.states}
    false_negatives = {tag: 0 for tag in model.states}

    for sent in dataset:
        predicted_tags = model.predict([word for word, _ in sent])
        gold_tags = [tag for _, tag in sent]

        for i in range(len(sent)):
            predicted_tag = predicted_tags[i]
            gold_tag = gold_tags[i]

            if predicted_tag == gold_tag:
                true_positives[gold_tag] += 1
            else:
                false_positives[predicted_tag] += 1
                false_negatives[gold_tag] += 1

    precision = {
        tag: (true_positives[tag] / (true_positives[tag] + false_positives[tag]))
        if (true_positives[tag] + false_positives[tag]) > 0 else 0
        for tag in model.states
    }
    recall = {
        tag: (true_positives[tag] / (true_positives[tag] + false_negatives[tag]))
        if (true_positives[tag] + false_negatives[tag]) > 0 else 0
        for tag in model.states
    }
    f1_score = {
        tag: (2 * precision[tag] * recall[tag]) / (precision[tag] + recall[tag])
        if (precision[tag] + recall[tag]) > 0 else 0
        for tag in model.states
    }

    overall_precision = sum(true_positives.values()) / (
        sum(true_positives.values()) + sum(false_positives.values())
    )
    overall_recall = sum(true_positives.values()) / (
        sum(true_positives.values()) + sum(false_negatives.values())
    )
    overall_f1 = (2 * overall_precision * overall_recall) / (
        overall_precision + overall_recall
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "overall_precision": overall_precision,
        "overall_recall": overall_recall,
        "overall_f1": overall_f1,
    }


# Load and preprocess the dataset
dataset_file_path = 'F:/IIT 2nd Semester/ASSIGNMENT/NLP/Ass02/NER-Dataset-Train.txt'
preprocessed_dataset = preprocess_dataset(dataset_file_path)

# Initialize HMM models for unigram, bigram, and trigram
hmm_unigram = HMMNER(order=1)
hmm_bigram = HMMNER(order=2)
hmm_trigram = HMMNER(order=3)

# Train the models
hmm_unigram.train(preprocessed_dataset)
hmm_bigram.train(preprocessed_dataset)
hmm_trigram.train(preprocessed_dataset)

# Test the models and evaluate performance (You can replace the test sequence with your own)

test_sequence = ['Intervention','Singapore','nintendo','sam']
predicted_tags_unigram = hmm_unigram.predict(test_sequence)
predicted_tags_bigram = hmm_bigram.predict(test_sequence)
predicted_tags_trigram = hmm_trigram.predict(test_sequence)

evaluation_unigram = evaluate_model(hmm_unigram, preprocessed_dataset)
evaluation_bigram = evaluate_model(hmm_bigram, preprocessed_dataset)
evaluation_trigram = evaluate_model(hmm_trigram, preprocessed_dataset)

print("Predicted tags (Unigram):", predicted_tags_unigram)
print("Predicted tags (Bigram):",  predicted_tags_bigram)
print("Predicted tags (Trigram):", predicted_tags_trigram)


# Define the results for each model
results = [
    ("Unigram", hmm_unigram, predicted_tags_unigram, evaluation_unigram),
    ("Bigram", hmm_bigram, predicted_tags_bigram, evaluation_bigram),
    ("Trigram", hmm_trigram, predicted_tags_trigram, evaluation_trigram)
]

# Watermark
watermark = "Ankit Anand 2311MC04"

# Print results in table format
for model_name, model, predicted_tags, evaluation in results:
    print(f"Results for {model_name} Model:")
    print("\nTransition Probability Table:")
    transition_table = [[context, tag, model.A[context].prob(tag)] for context in model.A.keys() for tag in model.A[context].samples()]
    print(tabulate(transition_table, headers=["Context", "Tag", "Probability"]))
    
    print("\nEmission Probability Table:")
    emission_table = [[tag, word, model.B[tag].prob(word)] for tag in model.B.keys() for word in model.B[tag].samples()]
    print(tabulate(emission_table, headers=["Tag", "Word", "Probability"]))
    
    print("\nPrecision, Recall, and F1-Score:")
    print(tabulate({
        "Tag": list(model.states),
        "Precision": [evaluation['precision'][tag] for tag in model.states],
        "Recall": [evaluation['recall'][tag] for tag in model.states],
        "F1-Score": [evaluation['f1_score'][tag] for tag in model.states]
    }, headers="keys"))
    
    print("\nOverall Performance:")
    print(tabulate({
        "Overall Precision": [evaluation['overall_precision']],
        "Overall Recall": [evaluation['overall_recall']],
        "Overall F1-Score": [evaluation['overall_f1']]
    }, headers="keys"))
    
    print("\n" + watermark.center(80, "-") + "\n")


# In[4]:


import numpy as np

# Define the states
states = ['O', 'B', 'I']

# Define the observations
observations = [
    '@LewisDixon', 'Trust', 'me', '!', 'im', 'gonna', 'be', 'bringing', 'out', 'music',
    'like', 'theres', 'no', 'tomorrow', ',', 'Be', 'doing', 'pure', 'blog', 'videos',
    '&amp;', 'freestyle', '#Moesh', '@joshHnumber1fan', 'its', 'okay', 'then', '..',
    'make', 'it', 'when', 'works', ':D', 'Asprin', 'check', 'cup', 'of', 'tea',
    'pillow', 'warm', 'sleeping', 'bag', 'fanfiction', 'on', 'the', 'laptop', '.',
    'Time', 'to', 'settle', 'down', 'and', 'relax', '@angelportugues', 'LMAO',
    'When', 'is', 'tht', 'one', 'day', '?:', 'P', 'The', 'Basic', 'Step', 'Before',
    'You', 'Even', 'Start', 'Thinking', 'Of', 'Making', 'Your', '...:', 'Keyword',
    'research', 'a', 'well', 'known', 'subject', 'yet', 'so', '...',
    'http://bit.ly/9XQgSr', 'today', 'just', 'does', "n't", 'feel', 'Friday', 'RT',
    '@Slijterijmeisje', ':', 'Kreeg', 'net', 'een', 'bruikbare', 'tip', 'van',
    'iemand', 'die', 'vorige', 'week', 'was', 'begonnen', 'met', 'whiskydieet',
    'hij', 'nu', 'al', '3', 'dagen', 'kwijt', 'Get', 'What', 'Give', '~', '40',
    'days', 'in', 'top', '100', 'Release', 'Date', 'September', '21', '2010Buy',
    'new', '$', '18.98', 'http://amzn.to/9Cfkpc', 'Last', 'stop', 'thank',
    'goddddd', '(@', 'H-E-B', 'Plus', ')', 'http://4sq.com/7RDhgd', 'friday', 'but',
    'instead', 'partying', 'or', 'homework', "I'm", 'going', 'for', 'bit', 'sleep',
    'XD', "i'm", 'off', 'bed', "i'll", 'go', 'meet', '@ElineEpica', 'there', 'her',
    'sweeeet16', '#partyyy', '@GOBLUE_FUCKosu', 'our', '10th', 'grade', 'year', 'smh',
    'I', 'think', 'coolest', 'things', 'do', 'would', 'hang', 'with', 'remeber', 'we',
    'were', 'sitting', 'by', 'water', 'you', 'put', 'your', 'arm', 'around', 'first',
    'Intervention','Singapore','nintendo','sam',
    'time', 'made', 'rebel', 'careless', 'mans', 'careful', 'daughter', '@daxx_d24',
    'best', 'feeling', 'world', 'twice', 'this', 'Got', 'more', 'way', 'lmao', 'i',
    'cant', '@DEVEY2G', 'liked', 'u', 'had', 'that', 'sure', 'cood', 'wit', 'my',
    'bro', 'possibly', 'Always', 'kinda', 'awkward', 'get', 'chat', 'group', 'wrong',
    'Sometimes', "don't", 'want', 'tank', 'know', "you're", 'dissin', "'", 'him',
    'Oops', 'MY', 'LIFE', 'She', 'who', 'am', 'fucker', 'before', 'she', 'fixed', 'And',
    'did', 'Wtf', 'have', 'stupid', '3hr', 'shift', 'tonight', '!!', 'omg', 'stuck',
    'work', 'last', 'weekend', 'before', 'college', 'start', ':(', 'Dont', 'tweet',
    'Andreas', 'aunt', 'Follow', '@kid_Geniuz', '(', 'rap', 'artist', ')', 'cool', 'shit',
    'http://www.reverbnation.com/kidgeniuz', 'Wusup', 'everybody', 'follow', 'me', 'and',
    'keep', 'with', 'updates', 'new', 'tracks', 'more', 'music', 'videos', 'Dont',
    'ever', 'play', 'slippery', 'floor', 'chucks', 'on', 'ya', 'gon', 'na', 'go', 'down',
    'Sittin', 'round', 'house', 'boys', 'hit', 'beach', 'Make', 'up', 'mind', 'Make', 'up',
    'mind', '!', 'Friday', 'nights', 'are', 'music', 'vids', '&', 'fruit', 'http://tinyurl.com/37nt43s'
]

# Define the transition probabilities
transition_prob = {
    'O': {'O': 0.5, 'B': 0.4, 'I': 0.1},
    'B': {'O': 0.3, 'B': 0.1, 'I': 0.6},
    'I': {'O': 0.2, 'B': 0.7, 'I': 0.1}
}

# Define the emission probabilities
emission_prob = {
    'O': {},
    'B': {},
    'I': {}
}

# Count occurrences of each observation in each state
for state in states:
    for obs in observations:
        if obs not in emission_prob[state]:
            emission_prob[state][obs] = 1
        else:
            emission_prob[state][obs] += 1

# Normalize emission probabilities
for state in emission_prob:
    total_obs = sum(emission_prob[state].values())
    for obs in emission_prob[state]:
        emission_prob[state][obs] /= total_obs

# Define the Viterbi algorithm function
def viterbi(obs, states, start_prob, trans_prob, emit_prob):
    V = [{}]
    path = {}

    # Initialize base cases (t == 0)
    for state in states:
        V[0][state] = start_prob[state] * emit_prob[state].get(obs[0], 0)
        path[state] = [state]

    # Run Viterbi algorithm for t > 0
    for t in range(1, len(obs)):
        V.append({})
        new_path = {}

        for state in states:
            (prob, prev_state) = max(
                (V[t - 1][prev_state] * trans_prob[prev_state].get(state, 0) * emit_prob[state].get(obs[t], 0), prev_state)
                for prev_state in states
            )
            V[t][state] = prob
            new_path[state] = path[prev_state] + [state]

        path = new_path

    # Find the most probable final state
    (prob, state) = max((V[len(obs) - 1][final_state], final_state) for final_state in states)

    return (prob, path[state])

# Define the initial probabilities
start_prob = {'O': 0.6, 'B': 0.2, 'I': 0.2}

# Apply the Viterbi algorithm
prob, best_sequence = viterbi(observations, states, start_prob, transition_prob, emission_prob)
example_observation = ['Intervention','Singapore','nintendo','sam']

# Apply the Viterbi algorithm to the example observation
example_prob, example_best_sequence = viterbi(example_observation, states, start_prob, transition_prob, emission_prob)

# Print the best sequence for the example observation
print("Example Best Sequence:", example_best_sequence)
print("Example Probability:", example_prob)


# In[ ]:




