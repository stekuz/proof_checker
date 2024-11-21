from object_classes import *
import json
import random
import requests

#run server with:$ java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
print(requests.post('http://[::]:9000/?properties={"annotators":"tokenize,pos","outputFormat":"json"}', data = {'data':'John hit the ball.'}).text)

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