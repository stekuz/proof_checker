from datasets import load_dataset
import numpy as np
import csv

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

Y = np.load('/home/user/Desktop/datasets/translation_en_de_train_wmt_y.npy', allow_pickle=True)
print(len(Y))