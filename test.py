#from datasets import load_dataset
import numpy as np
import csv
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
import tensorflow as tf
import time

special_chars = ['„', '”', "!", "\"", "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/", 
    ":", ";", "<", "=", ">", "?", "@", "[", "\\", "]", "^", "_", "`", "{", "|", "}", "~",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    " ", "\t", "\n", "\r", "\f", "\v"
]
def process():
    # Load the dataset (e.g., Europarl)
    dataset = load_dataset("europarl_bilingual", "de-en")

    # Define the 100-word vocabulary
    most_used_english_words = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "I", 
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at", 
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she", 
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what", 
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me", 
    "when", "make", "can", "like", "time", "no", "just", "him", "know", "take", 
    "people", "into", "year", "your", "good", "some", "could", "them", "see", "other", 
    "than", "then", "now", "look", "only", "come", "its", "over", "think", "also", 
    "back", "after", "use", "two", "how", "our", "work", "first", "well", "way", 
    "even", "new", "want", "because", "any", "these", "give", "day", "most", "us", 
    "is", "are", "was", "were", "been", "being", "have", "has", "had", "having", 
    "do", "does", "did", "doing", "say", "says", "said", "saying", "get", "gets", 
    "got", "getting", "make", "makes", "made", "making", "go", "goes", "went", "going", 
    "know", "knows", "knew", "knowing", "take", "takes", "took", "taking", "see", "sees", 
    "saw", "seeing", "come", "comes", "came", "coming", "think", "thinks", "thought", "thinking", 
    "look", "looks", "looked", "looking", "want", "wants", "wanted", "wanting", "give", "gives", 
    "gave", "giving", "use", "uses", "used", "using", "find", "finds", "found", "finding", 
    "tell", "tells", "told", "telling", "ask", "asks", "asked", "asking", "work", "works", 
    "worked", "working", "seem", "seems", "seemed", "seeming", "feel", "feels", "felt", "feeling", 
    "try", "tries", "tried", "trying", "leave", "leaves", "left", "leaving", "call", "calls", 
    "called", "calling", "need", "needs", "needed", "needing", "become", "becomes", "became", "becoming", 
    "put", "puts", "put", "putting", "mean", "means", "meant", "meaning", "keep", "keeps", 
    "kept", "keeping", "let", "lets", "let", "letting", "begin", "begins", "began", "beginning", 
    "seem", "seems", "seemed", "seeming", "help", "helps", "helped", "helping", "talk", "talks", 
    "talked", "talking", "turn", "turns", "turned", "turning", "start", "starts", "started", "starting", 
    "show", "shows", "showed", "showing", "hear", "hears", "heard", "hearing", "play", "plays", 
    "played", "playing", "run", "runs", "ran", "running", "move", "moves", "moved", "moving", 
    "like", "likes", "liked", "liking", "live", "lives", "lived", "living", "believe", "believes", 
    "believed", "believing", "hold", "holds", "held", "holding", "bring", "brings", "brought", "bringing", 
    "happen", "happens", "happened", "happening", "write", "writes", "wrote", "writing", "provide", "provides", 
    "provided", "providing", "sit", "sits", "sat", "sitting", "stand", "stands", "stood", "standing", 
    "lose", "loses", "lost", "losing", "pay", "pays", "paid", "paying", "meet", "meets", 
    "met", "meeting", "include", "includes", "included", "including", "continue", "continues", "continued", "continuing", 
    "set", "sets", "set", "setting", "learn", "learns", "learned", "learning", "change", "changes", 
    "changed", "changing", "lead", "leads", "led", "leading", "understand", "understands", "understood", "understanding", 
    "watch", "watches", "watched", "watching", "follow", "follows", "followed", "following", "stop", "stops", 
    "stopped", "stopping", "create", "creates", "created", "creating", "speak", "speaks", "spoke", "speaking", 
    "read", "reads", "read", "reading", "allow", "allows", "allowed", "allowing", "add", "adds", 
    "added", "adding", "spend", "spends", "spent", "spending", "grow", "grows", "grew", "growing", 
    "open", "opens", "opened", "opening", "walk", "walks", "walked", "walking", "win", "wins", 
    "won", "winning", "offer", "offers", "offered", "offering", "remember", "remembers", "remembered", "remembering", 
    "love", "loves", "loved", "loving", "consider", "considers", "considered", "considering", "appear", "appears", 
    "appeared", "appearing", "buy", "buys", "bought", "buying", "wait", "waits", "waited", "waiting", 
    "serve", "serves", "served", "serving", "die", "dies", "died", "dying", "send", "sends", 
    "sent", "sending", "build", "builds", "built", "building", "stay", "stays", "stayed", "staying", 
    "fall", "falls", "fell", "falling", "cut", "cuts", "cut", "cutting", "reach", "reaches", 
    "reached", "reaching", "kill", "kills", "killed", "killing", "remain", "remains", "remained", "remaining", 
    "suggest", "suggests", "suggested", "suggesting", "raise", "raises", "raised", "raising", "pass", "passes", 
    "passed", "passing", "sell", "sells", "sold", "selling", "require", "requires", "required", "requiring", 
    "report", "reports", "reported", "reporting", "decide", "decides", "decided", "deciding", "pull", "pulls", 
    "pulled", "pulling", "return", "returns", "returned", "returning", "explain", "explains", "explained", "explaining", 
    "hope", "hopes", "hoped", "hoping", "develop", "develops", "developed", "developing", "carry", "carries", 
    "carried", "carrying", "break", "breaks", "broke", "breaking", "receive", "receives", "received", "receiving", 
    "agree", "agrees", "agreed", "agreeing", "support", "supports", "supported", "supporting", "hit", "hits", 
    "hit", "hitting", "produce", "produces", "produced", "producing", "eat", "eats", "ate", "eating", 
    "cover", "covers", "covered", "covering", "catch", "catches", "caught", "catching", "draw", "draws", 
    "drew", "drawing", "choose", "chooses", "chose", "choosing", "wear", "wears", "wore", "wearing", 
    "fight", "fights", "fought", "fighting", "throw", "throws", "threw", "throwing", "fill", "fills", 
    "filled", "filling", "drop", "drops", "dropped", "dropping", "push", "pushes", "pushed", "pushing", 
    "close", "closes", "closed", "closing", "drive", "drives", "drove", "driving", "reduce", "reduces", 
    "reduced", "reducing", "imagine", "imagines", "imagined", "imagining", "wonder", "wonders", "wondered", "wondering", 
    "notice", "notices", "noticed", "noticing", "shut", "shuts", "shut", "shutting", "form", "forms", 
    "formed", "forming", "lay", "lays", "laid", "laying", "avoid", "avoids", "avoided", "avoiding", 
    "accept", "accepts", "accepted", "accepting", "prepare", "prepares", "prepared", "preparing", "describe", "describes", 
    "described", "describing", "improve", "improves", "improved", "improving", "realize", "realizes", "realized", "realizing", 
    "refer", "refers", "referred", "referring", "manage", "manages", "managed", "managing", "thank", "thanks", 
    "thanked", "thanking", "control", "controls", "controlled", "controlling", "prevent", "prevents", "prevented", "preventing", 
    "express", "expresses", "expressed", "expressing", "compare", "compares", "compared", "comparing", "determine", "determines", 
    "determined", "determining", "apply", "applies", "applied", "applying", "argue", "argues", "argued", "arguing", 
    "organize", "organizes", "organized", "organizing", "establish", "establishes", "established", "establishing", "recognize", "recognizes", 
    "recognized", "recognizing", "mention", "mentions", "mentioned", "mentioning", "introduce", "introduces", "introduced", "introducing", 
    "imply", "implies", "implied", "implying", "reflect", "reflects", "reflected", "reflecting", "replace", "replaces", 
    "replaced", "replacing", "focus", "focuses", "focused", "focusing", "define", "defines", "defined", "defining", 
    "state", "states", "stated", "stating", "identify", "identifies", "identified", "identifying", "exist", "exists", 
    "existed", "existing", "occur", "occurs", "occurred", "occurring", "depend", "depends", "depended", "depending", 
    "respond", "responds", "responded", "responding", "claim", "claims", "claimed", "claiming", "maintain", "maintains", 
    "maintained", "maintaining", "indicate", "indicates", "indicated", "indicating", "publish", "publishes", "published", "publishing", 
    "assume", "assumes", "assumed", "assuming", "obtain", "obtains", "obtained", "obtaining", "achieve", "achieves", 
    "achieved", "achieving", "seek", "seeks", "sought", "seeking", "select", "selects", "selected", "selecting", 
    "participate", "participates", "participated", "participating", "contribute", "contributes", "contributed", "contributing", "investigate", "investigates", 
    "investigated", "investigating", "demonstrate", "demonstrates", "demonstrated", "demonstrating", "emphasize", "emphasizes", "emphasized", "emphasizing", 
    "promote", "promotes", "promoted", "promoting", "assess", "assesses", "assessed", "assessing", "enhance", "enhances", 
    "enhanced", "enhancing", "examine", "examines", "examined", "examining", "illustrate", "illustrates", "illustrated", "illustrating", 
    "implement", "implements", "implemented", "implementing", "justify", "justifies", "justified", "justifying", "clarify", "clarifies", 
    "clarified", "clarifying", "confirm", "confirms", "confirmed", "confirming", "evaluate", "evaluates", "evaluated", "evaluating", 
    "interpret", "interprets", "interpreted", "interpreting", "predict", "predicts", "predicted", "predicting", "resolve", "resolves", 
    "resolved", "resolving", "specify", "specifies", "specified", "specifying", "transform", "transforms", "transformed", "transforming", 
    "adapt", "adapts", "adapted", "adapting", "allocate", "allocates", "allocated", "allocating", "anticipate", "anticipates", 
    "anticipated", "anticipating", "assure", "assures", "assured", "assuring", "attain", "attains", "attained", "attaining", 
    "comply", "complies", "complied", "complying", "conclude", "concludes", "concluded", "concluding", "conduct", "conducts", 
    "conducted", "conducting", "consist", "consists", "consisted", "consisting", "construct", "constructs", "constructed", "constructing", 
    "consult", "consults", "consulted", "consulting", "coordinate", "coordinates", "coordinated", "coordinating", "derive", "derives", 
    "derived", "deriving", "detect", "detects", "detected", "detecting", "devote", "devotes", "devoted", "devoting", 
    "diminish", "diminishes", "diminished", "diminishing", "eliminate", "eliminates", "eliminated", "eliminating", "emerge", "emerges", 
    "emerged", "emerging", "ensure", "ensures", "ensured", "ensuring", "exceed", "exceeds", "exceeded", "exceeding", 
    "exclude", "excludes", "excluded", "excluding", "facilitate", "facilitates", "facilitated", "facilitating", "generate", "generates", 
    "generated", "generating", "guarantee", "guarantees", "guaranteed", "guaranteeing", "highlight", "highlights", "highlighted", "highlighting", 
    "impose", "imposes", "imposed", "imposing", "incorporate", "incorporates", "incorporated", "incorporating", "induce", "induces", 
    "induced", "inducing", "initiate", "initiates", "initiated", "initiating", "integrate", "integrates", "integrated", "integrating", 
    "intervene", "intervenes", "intervened", "intervening", "isolate", "isolates", "isolated", "isolating", "minimize", "minimizes", 
    "minimized", "minimizing", "monitor", "monitors", "monitored", "monitoring", "negotiate", "negotiates", "negotiated", "negotiating", 
    "offset", "offsets", "offset", "offsetting", "perceive", "perceives", "perceived", "perceiving", "preserve", "preserves", 
    "preserved", "preserving", "proceed", "proceeds", "proceeded", "proceeding", "pursue", "pursues", "pursued", "pursuing", 
    "reinforce", "reinforces", "reinforced", "reinforcing", "restore", "restores", "restored", "restoring", "retain", "retains", 
    "retained", "retaining", "reverse", "reverses", "reversed", "reversing", "simulate", "simulates", "simulated", "simulating", 
    "submit", "submits", "submitted", "submitting", "sustain", "sustains", "sustained", "sustaining", "terminate", "terminates", 
    "terminated", "terminating", "transform", "transforms", "transformed", "transforming", "undergo", "undergoes", "underwent", "undergoing", 
    "utilize", "utilizes", "utilized", "utilizing", "verify", "verifies", "verified", "verifying", "withdraw", "withdraws", 
    "withdrew", "withdrawing"
]

    vocabulary = set(most_used_english_words)
    # Filter the dataset
    filtered_data = []
    for example in dataset["train"]:
        english_sentence = example["translation"]["en"]
        german_sentence = example["translation"]["de"]

        # Check if all words in the English sentence are in the vocabulary
        sentence = english_sentence.split()
        k = 0
        for i in sentence:
            if i.lower() in vocabulary:
                k += 1
        if k / len(sentence) > 0.9:
            filtered_data.append((english_sentence, german_sentence))

    # Print the filtered dataset
    with open('./data.txt', 'w') as f:
        for pair in filtered_data:
            f.write(pair[0][:-1] + ':' + pair[1][:-1] + '\n')
    print(len(filtered_data))
    for en, de in filtered_data[:10]:
        print(f"English: {en}")
        print(f"German: {de}")
        print()

all_tokens = {}
mxlen = 0

def data_preparation_wmt(source_path, x_path, y_path):
    global all_tokens
    global mxlen

    data_raw = []
    with open(source_path, "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for i, row in enumerate(reader):
            if len(row) == 2:
                data_raw.append(row)
            if i == 30000:
                break

    data_train = []
    for src, tgt in data_raw:
        src_tokens = tokenize_text(src)
        tgt_tokens = tokenize_text(tgt)
        mxlen = max(mxlen, len(src_tokens) + len(tgt_tokens) + 2)
        data_train.append((src_tokens, tgt_tokens))
        update_vocab(src_tokens + tgt_tokens)

    save_vocab(all_tokens)

    all_tokens_list = sorted(all_tokens.keys())
    token_to_idx = {token: idx for idx, token in enumerate(all_tokens_list)}
    data_train = [
        (
            [[1, token_to_idx[token] / (len(all_tokens_list) + 1)] for token in src],
            [[1, token_to_idx[token] / (len(all_tokens_list) + 1)] for token in tgt],
        )
        for src, tgt in data_train
    ]

    X, Y = prepare_sequences_variable_length(data_train, len(all_tokens_list))

    np.save(x_path, np.array(X, dtype=object), allow_pickle=True)
    np.save(y_path, np.array(Y, dtype=object), allow_pickle=True)

def tokenize_text(text):
    tokens = []
    current_token = ""
    for char in text:
        if char in special_chars:
            if current_token:
                tokens.append(current_token)
            tokens.append(char)
            current_token = ""
        else:
            current_token += char.lower()
    if current_token:
        tokens.append(current_token)
    return tokens

def update_vocab(tokens):
    for token in tokens:
        all_tokens[token] = 1

def save_vocab(vocab):
    with open("all_tokens.txt", "w") as f:
        for token in vocab:
            f.write(token + "\n")

def prepare_sequences_variable_length(data_train, vocab_size):
    X = []
    Y = []
    for src, tgt in data_train:
        for j in range(len(src)):
            X_seq = tgt + src[:j]
            X_seq = [[k / mxlen, val[1]] for k, val in enumerate(X_seq)]
            X.append(np.array(X_seq, dtype="float32"))
            Y_seq = round(src[j][1] * (vocab_size + 1))
            Y.append(np.array(Y_seq, dtype="float32"))
    return X, Y

#data_preparation_wmt('/home/user/Desktop/datasets/wmt14_translate_de-en_train.csv',
#                     '/home/user/Desktop/datasets/translation_en_de_train_wmt_x.npy',
#                     '/home/user/Desktop/datasets/translation_en_de_train_wmt_y.npy')

'''tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
trainer = trainers.BpeTrainer(vocab_size=32000, special_tokens=special_chars)
tokenizer.train(files=['/home/user/Desktop/datasets/wmt14_translate_de-en_train.csv'], trainer=trainer)
tokenizer.save("tokenizer.json")'''
'''
import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from tqdm import tqdm
import os


# Step 1: Load the full WMT24 English-German translation dataset
print("Loading dataset...")
dataset = load_dataset("wmt19", "de-en")

# Step 2: Train a tokenizer with a vocabulary limited to 10,000 tokens
def train_tokenizer(dataset, vocab_size=10000, save_path="./tokenizer.json"):
    print("Training tokenizer...")
    # Initialize a BPE tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # Define a trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    )

    # Prepare the text data for training
    texts = []
    for split in dataset:
        for example in dataset[split]:
            example = example['translation']
            texts.append(example["en"])
            texts.append(example["de"])

    # Train the tokenizer
    tokenizer.train_from_iterator(texts, trainer=trainer)

    # Save the tokenizer
    tokenizer.save(save_path)
    print(f"Tokenizer saved to {save_path}")
    return tokenizer

# Train and save the tokenizer
tokenizer = train_tokenizer(dataset)

# Step 3: Partition the dataset into chunks of 10,000 sentence pairs
def partition_dataset(dataset, chunk_size=10000):
    print("Partitioning dataset...")
    chunks = []
    current_chunk = []
    for split in dataset:
        for example in dataset[split]:
            current_chunk.append((example["en"], example["de"]))
            if len(current_chunk) == chunk_size:
                chunks.append(current_chunk)
                current_chunk = []
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

chunks = partition_dataset(dataset)

# Step 4: Tokenize each sentence pair and write to numpy arrays
def tokenize_and_save_chunks(chunks, tokenizer, filepath):
    print("Tokenizing and saving chunks...")
    os.makedirs(filepath, exist_ok=True)
    all_tokens = tokenizer.get_vocab_size()

    for chunk_number, chunk in enumerate(tqdm(chunks)):
        X = np.empty((len(chunk), 2), dtype=object)
        Y = np.empty((len(chunk), 2), dtype=object)

        for i, (en_text, de_text) in enumerate(chunk):
            # Tokenize English text
            en_tokens = tokenizer.encode(en_text).ids
            en_token_number = en_tokens[0] if en_tokens else 0
            X[i] = [1, en_token_number / (all_tokens + 1)]

            # Tokenize German text
            de_tokens = tokenizer.encode(de_text).ids
            de_token_number = de_tokens[0] if de_tokens else 0
            Y[i] = [1, de_token_number / (all_tokens + 1)]

        # Save X and Y to files
        np.save(os.path.join(filepath, f"{chunk_number}X.npy"), X)
        np.save(os.path.join(filepath, f"{chunk_number}Y.npy"), Y)

# Save chunks to files
#filepath = "./chunks/"
#tokenize_and_save_chunks(chunks, tokenizer, filepath)

#print("Processing complete!")'''

def f():
    global mxlen
    samples_selected = 1000
    X = np.load('/home/user/Desktop/datasets/translation_en_de_train_wmt_x.npy', allow_pickle=True)[:samples_selected]
    #X = np.load('/home/user/Desktop/datasets/translation_en_de_train_common_seq2seq_x.npy')
    print(len(X))
    #Y = np.load('/home/user/Desktop/datasets/translation_en_de_train_common_seq2seq_y.npy')
    Y = np.load('/home/user/Desktop/datasets/translation_en_de_train_wmt_y.npy', allow_pickle=True)[:samples_selected]
    print(X.shape, Y.shape, mxlen)
    learning_rate = 0.001
    hidden_size = 100
    total_input = 2
    batch_size = 256
    X = X.tolist()
    batches_x = [X[i:i + batch_size] for i in range(0, len(X), batch_size)]
    lastlen = len(batches_x[-1])
    for i in range(batch_size - lastlen):
        batches_x[-1].append(batches_x[-1][-1])
    for i in range(len(batches_x)):
        batches_x[i] = np.array(batches_x[i], dtype=object)
    Y = Y.tolist()
    batches_y = [Y[i:i + batch_size] for i in range(0, len(X), batch_size)]
    lastlen = len(batches_y[-1])
    for i in range(batch_size - lastlen):
        batches_y[-1].append(batches_y[-1][-1])
    for i in range(len(batches_y)):
        batches_y[i] = np.array(batches_y[i], dtype=object)
    filepath = './custom_models_2/test_model'
    loss_graph_x = []
    loss_graph_y = []
    accuracy_graph_x = []
    accuracy_graph_y = []
    avg_loss = [0 for i in range(100)]
    avg_accuracy = [0 for i in range(100)]
    threshold_loss = 0
    mxlen = int(open('./mxlen.txt', 'r').readline())
    sequence_order_padding = []
    for i in range(mxlen):
        sequence_order_padding.append([i / mxlen])
    sequence_order_padding = tf.convert_to_tensor(sequence_order_padding, dtype='float32')
    sequence_mxlen_padding = []
    for i in range(mxlen):
        sequence_mxlen_padding.append([i / mxlen, len(all_tokens) / (len(all_tokens) + 1)])
    sequence_mxlen_padding = tf.convert_to_tensor(sequence_mxlen_padding, dtype='float32')
    print(sequence_mxlen_padding.shape)
    def pad_arrays(X, P):
        padded_X = []
        for i in range(len(X)):
            padded_X.append(tf.concat((X[i], P[len(X[i]):]), axis=0))
        return tf.convert_to_tensor(padded_X, dtype='float32')
    
    def make_categorical(Y):
        categorical = []
        for i in range(len(Y)):
            categorical.append(tf.cast([k == Y[i] for k in range(len(all_tokens) + 1)], dtype='float32'))
        return tf.convert_to_tensor(categorical, dtype='float32')

    def trainint_loop(epoch):
        start_time = time.time()
        loss = 0
        accuracy = 0
        for i in range(len(batches_x)):
            X = batches_x[i]
            X = pad_arrays(X, sequence_mxlen_padding)
            Y = batches_y[i]
            Y = make_categorical(Y)
        print(time.time() - start_time)
    trainint_loop(1)

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

stop_words_full = set(stopwords.words('english'))

stop_words = []

for word in stop_words_full:
    text = tokenize_text(word)
    for token in text:
        stop_words.append(token)

stop_words = sorted(list(set(stop_words)))

with open('./stop_words_imdb.txt', 'w') as f:
    for token in stop_words:
        f.write(token + '\n')

def preprocess_mxlen_tokens():
    global mxlen
    global all_tokens
    data_raw = []
    with open('/home/user/Desktop/datasets/IMDB Dataset.csv', "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for i, row in enumerate(reader):
            if len(row) == 2:
                data_raw.append(row)
            if i == 100000:
                break
    for row in data_raw:
        tokenized_full = tokenize_text(row[0])
        tokenized = []
        for token in tokenized_full:
            if not (token in stop_words):
                tokenized.append(token)
        if len(tokenized) >= 300:
            continue
        mxlen = max(mxlen, len(tokenized) + 10)
        for token in tokenized:
            all_tokens[token] = 1
    with open('./all_tokens_imdb.txt', 'w') as f:
        for token in all_tokens:
            f.write(token + '\n')
    with open('./mxlen_imdb.txt', 'w') as f:
        f.write(str(mxlen) + '\n')

def preprocess_data_imdb(index):
    global mxlen
    global all_tokens
    data_raw = []
    data_train = []
    with open('/home/user/Desktop/datasets/IMDB Dataset.csv', "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for i, row in enumerate(reader):
            if len(row) == 2 and i >= 10000 * (index - 1 - int(index == 6)) and i < 10000 * index:
                data_raw.append(row)
    for row in data_raw:
        tokenized_full = tokenize_text(row[0])
        tokenized = []
        for token in tokenized_full:
            if not (token in stop_words):
                tokenized.append(token)
        if len(tokenized) < 300:
            data_train.append([tokenized, int(row[1] == 'positive')])
    mxlen = int(open('./mxlen_imdb.txt', 'r').readlines()[0])
    all_tokens_list = open('./all_tokens_imdb.txt', 'r').readlines()
    all_tokens_list = [token[:-1] for token in all_tokens_list]
    all_tokens_list = sorted(all_tokens_list)
    for i in range(len(all_tokens_list)):
        all_tokens[all_tokens_list[i]] = i
    X = []
    Y = []
    for data in data_train:
        X.append([])
        for token in data[0]:
            X[-1].append([1, all_tokens[token] / (len(all_tokens) + 1)])
        nowlen = len(X[-1])
        for i in range(mxlen - nowlen):
            X[-1].append([1, len(all_tokens) / (len(all_tokens) + 1)])
        for i in range(mxlen):
            X[-1][i][0] = i / mxlen
        Y.append([1 - data[1], data[1]])
    X = np.array(X, dtype='float32')
    Y = np.array(Y, dtype='float32')
    np.save(f'/home/user/Desktop/datasets/imdb_50k_x_{index - int(index == 6)}.npy', X)
    np.save(f'/home/user/Desktop/datasets/imdb_50k_y_{index - int(index == 6)}.npy', Y)
    print(index)

#preprocess_mxlen_tokens()
#preprocess_data_imdb(1)
#preprocess_data_imdb(2)
#preprocess_data_imdb(3)
#preprocess_data_imdb(4)
#preprocess_data_imdb(6)
    
def preprocess_training_batches(index):
    mxlen = int(open('./mxlen_imdb.txt', 'r').readlines()[0])
    all_tokens_list = open('./all_tokens_imdb.txt', 'r').readlines()
    all_tokens_list = [token[:-1] for token in all_tokens_list]
    all_tokens_list = sorted(all_tokens_list)
    for i in range(len(all_tokens_list)):
        all_tokens[all_tokens_list[i]] = i
    X = np.load(f'/home/user/Desktop/datasets/imdb_50k_x_{index}.npy', allow_pickle=True)
    print(len(X))
    Y = np.load(f'/home/user/Desktop/datasets/imdb_50k_y_{index}.npy', allow_pickle=True)
    batch_size = 300
    X = X.tolist()
    batches_x = [X[i:i + batch_size] for i in range(0, len(X), batch_size)]
    lastlen = len(batches_x[-1])
    for i in range(batch_size - lastlen):
        batches_x[-1].append(batches_x[-1][-1])
    for i in range(len(batches_x)):
        batches_x[i] = tf.convert_to_tensor(batches_x[i], dtype='float32')
    Y = Y.tolist()
    batches_y = [Y[i:i + batch_size] for i in range(0, len(X), batch_size)]
    lastlen = len(batches_y[-1])
    for i in range(batch_size - lastlen):
        batches_y[-1].append(batches_y[-1][-1])
    for i in range(len(batches_y)):
        batches_y[i] = tf.convert_to_tensor(batches_y[i], dtype='float32')
    for i in range(len(batches_x)):
        np.save(f'/home/user/Desktop/batches/training_batches_x_{68 + i}', batches_x[i])
        np.save(f'/home/user/Desktop/batches/training_batches_y_{68 + i}', batches_y[i])
    return len(batches_x)

#preprocess_training_batches(1)
#preprocess_training_batches(2)
#preprocess_training_batches(3)
#preprocess_training_batches(4)
#preprocess_training_batches(5)
