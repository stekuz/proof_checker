from object_classes import *
import json
import random
import requests

#run server with:
# $ cd ~/Desktop/stanford-corenlp-4.5.7
# $ java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
#print(requests.post('http://[::]:9000/?properties={"annotators":"parse","outputFormat":"json"}', data = {'data':'A strong man had hit the ball.'}).json()['sentences'][0]['parse'])

def set_generator():
    output_json = open('./class_labeled/set.json','w')
    words_for_naming_set = ['set','multiset','collection','list','array','number of']
    symbols_for_opening_set = ['\{','[','(']
    symbols_for_closing_set = ['\}',']',')']
    defining_words = ['=','==',':=','is','defined as','as','of']
    middle_words = ['\mid','|','where','with']
    belonging_words = ['\in','from','belongs to','belong to','in']
    separators = [';',',',' ']
    latin_alphabet = 'abcdefghijklmnopqrstuvwxyz'
    latin_alphabet += latin_alphabet.upper()
    latin_alphabet += '0123456789\\_-'
    number_of_samples = 100
    res = {}
    for i in range(number_of_samples):
        set_object = Set('','','')
        name = ''
        namelen = random.randint(1,20)
        for j in range(namelen):
            name += latin_alphabet[random.randint(0,len(latin_alphabet)-1)]
        set_object.set_name = name
        naming_word_before = random.randint(0,1)
        if naming_word_before:
            name = words_for_naming_set[random.randint(0,len(words_for_naming_set)-1)] + ' ' + name
        else:
            name = name + ' ' + words_for_naming_set[random.randint(0,len(words_for_naming_set)-1)]
        name += ' ' + defining_words[random.randint(0,len(defining_words)-1)]
        symbol_for_opening = random.randint(0,len(symbols_for_opening_set)-1)
        name += ' ' + symbols_for_opening_set[symbol_for_opening]
        space_after_opening= random.randint(0,1)
        if space_after_opening:
            name += ' '
        set_generators_number = random.randint(1,10)
        for j in range(set_generators_number):
            generator_name = ''
            generator_name_len = random.randint(1,10)
            for jj in range(generator_name_len):
                generator_name += latin_alphabet[random.randint(0,len(latin_alphabet)-1)]
            generator_type = ''
            generator_type_len = random.randint(1,20)
            for jj in range(generator_type_len):
                generator_type += latin_alphabet[random.randint(0,len(latin_alphabet)-1)]
            set_object.objects_type += generator_type + ' '
            set_object.content += generator_name + ' '
            name += separators[random.randint(0,len(separators)-1)]
            name += generator_name
            name += belonging_words[random.randint(0,len(belonging_words)-1)]
        space_before_middle = random.randint(0,1)
        if space_before_middle:
            name += ' '
        name += middle_words[random.randint(0,len(middle_words)-1)]
        space_after_middle = random.randint(0,1)
        if space_after_middle:
            name += ' '
        namelen = random.randint(3,20)
        for j in range(namelen):
            name += latin_alphabet[random.randint(0,len(latin_alphabet)-1)]
        space_before_closing = random.randint(0,1)
        if space_before_closing:
            name += ' '
        name += symbols_for_closing_set[symbol_for_opening]
        res[name] = set_object.__dict__
    json.dump(res,output_json)
        
def atom_generator():
    output_json = open('./class_labeled/atom.json','w')
    defining_words = ['=','==',':=',' is ',' defined as ',' as ',' of ']
    belonging_words = ['\in ',' from ',' belongs to ',' belong to ',' in ']
    latin_alphabet = 'abcdefghijklmnopqrstuvwxyz'
    latin_alphabet += latin_alphabet.upper()
    latin_alphabet += '0123456789\\_-'
    number_of_samples = 100
    res = {}
    for i in range(number_of_samples):
        atom_object = Atom('','')
        name = ''
        namelen = random.randint(1,20)
        for j in range(namelen):
            name += latin_alphabet[random.randint(0,len(latin_alphabet)-1)]
        atom_object.atom_name = name
        if_belongs = random.randint(0,1)
        if if_belongs:
            name += belonging_words[random.randint(0,len(belonging_words)-1)]
        else:
            name += defining_words[random.randint(0,len(defining_words)-1)]
        content = ''
        contentlen = random.randint(1,100)
        for j in range(contentlen):
            content += latin_alphabet[random.randint(0,len(latin_alphabet)-1)]
        atom_object.content = content
        name += content
        res[name] = atom_object.__dict__
    json.dump(res,output_json)

def random_tree_generator(n):
    parents = [0]
    number_of_children = [0]*n
    for i in range(1,n):
        parent = random.randint(0,i-1)
        additional_child = random.randint(0,10)
        while number_of_children[parent] >= 2 and additional_child > 2:
            parent = random.randint(0,i-1)
            additional_child = random.randint(0,10)
        parents.append(parent)
        number_of_children[parent] += 1
    return parents

def get_leaves(tree_parents):
    n = len(tree_parents)
    have_children = [0]*n
    for i in range(n):
        have_children[tree_parents[i]] = 1
    leaves = []
    for i in range(n):
        if have_children[i] == 0:
            leaves.append(i)
    return leaves

def action_generator():
    output_json = open('./class_labeled/action.json','w')
    parts_of_speech_tags = ['CC','CD','DT','EX','FW','IN','JJ','JJR','JJS','LS','MD','NN','NNS','NNP','NNPS','PDT','POS','PRP','PRP$','RB','RBR','RBS','RP','SYM','TO','UH','VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP','WP$','WRB']
    latin_alphabet = 'abcdefghijklmnopqrstuvwxyz'
    latin_alphabet += latin_alphabet.upper()
    latin_alphabet += '0123456789\\_-'
    number_of_samples = 100
    res = {}
    for i in range(number_of_samples):
        tree_parents = random_tree_generator(random.randint(5,50))
        tree_children = []
        n = len(tree_parents)
        for j in range(n):
            tree_children.append([])
        for j in range(1,n):
            tree_children[tree_parents[j]].append(j)
        leaves = get_leaves(tree_parents)
        tree_labels = ['']*n
        for leave in leaves:
            tree_labels[leave] = parts_of_speech_tags[random.randint(0,len(parts_of_speech_tags)-1)]
        not_visited_children = [0]*n
        for j in range(1,n):
            not_visited_children[tree_parents[j]] += 1
        def lift(v):
            if v == 0:
                tree_labels[v] = 'S'
                return
            if not_visited_children[v]:
                return
            for child in tree_children[v]:
                if 'N' in tree_labels[child]:
                    tree_labels[v] = 'NP'
                    break
                elif 'V' in tree_labels[child]:
                    tree_labels[v] = 'VP'
                    break
                elif 'J' in tree_labels[child]:
                    tree_labels[v] = 'JP'
                    break
                elif 'P' in tree_labels[child]:
                    tree_labels[v] = 'PP'
                    break
                elif 'R' in tree_labels[child]:
                    tree_labels[v] = 'RP'
                    break
                else:
                    continue
            if tree_labels[v] == '':
                tree_labels[v] = tree_labels[tree_children[v][0]]
            not_visited_children[tree_parents[v]] -= 1
            lift(tree_parents[v])
        for leave in leaves:
            not_visited_children[tree_parents[leave]] -= 1
            lift(tree_parents[leave])
        words = ['']*len(leaves)
        for j in range(len(words)):
            word = ''
            wordlen = random.randint(1,15)
            for jj in range(wordlen):
                word += latin_alphabet[random.randint(0,len(latin_alphabet)-1)]
            words[j] = word
        text = ''
        for word in words:
            text += word + ' '
        action_object = Action('','','')
        last_noun = -1
        actor_name_leave = -1
        for j in range(len(leaves)):
            if 'N' in tree_labels[leaves[j]]:
                if last_noun == -1:
                    last_noun = j
                elif action_object.actor_name == '':
                    for k in range(last_noun+1,j):
                        if 'R' in tree_labels[leaves[k]]:
                            action_object.actor_name = words[j]
                            actor_name_leave = j
                            break
                        elif 'IN' == tree_labels[leaves[k]]:
                            action_object.actor_name = words[last_noun]
                            actor_name_leave = last_noun
                            break
            if 'V' in tree_labels[leaves[j]]:
                action_object.output = words[j]
        for j in range(len(leaves)):
            if 'N' in tree_labels[leaves[j]]:
                if actor_name_leave != j:
                    action_object.input_list += words[j] + ' '
        res[text] = action_object.__dict__
    json.dump(res,output_json)

action_generator()