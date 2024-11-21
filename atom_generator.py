from object_classes import *
import json
import random

def generate_numbers():
    example_atom = Atom('x', 'number', 10)#x=10
    example_text = '$x=10$'

    numbers_json = open('./numbers.json','w')
    numbers = {}
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    for i in range(10000):
        namelen = random.randint(1,5)
        name = ''
        for j in range(namelen):
            name += alphabet[random.randint(0,25)]
        name += '_' + str(random.randint(0,100))
        value = random.randint(1,100)
        atom = Atom(name, 'number', value)
        text = '$' + name + '=' + str(value) + '$'
        numbers[text] = {
            'name': atom.atom_name,
            'type': atom.type_name,
            'value': atom.content
        }
    json.dump(numbers, numbers_json)

generate_numbers()