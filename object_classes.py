class Action:
    actor_name = ''
    input_list = ''
    output = ''
    
    def __init__(self, actor_name, input_list, output):
        self.actor_name = actor_name
        self.input_list = input_list
        self.output = output

class Set:
    set_name = ''
    objects_type = ''
    content = ''

    def __init__(self, set_name, objects_type, content=''):
        self.set_name = set_name
        self.objects_type = objects_type
        self.content = content

    def insert(self, element):
        self.content.add(element)
    
    def erase(self, element):
        self.content.remove(element)

class Atom:
    atom_name = ''
    content = ''

    def __init__(self, atom_name, content):
        self.atom_name = atom_name
        self.content = content

functions = {}

'''
Пример:
Пусть $n$ -- нечетное натуральное число.
Пусть $S$ - мультимножество, изначально состоящее из чисел $\{1,2,...,2n\}$.
$A$ выполняет следующую операцию над этим множеством до того момента, пока во множестве больше, чем один элемент:
берет различные элементы $a,b\in S$, удаляет каждое из них из $S$, и добавляет $|a-b|$ в $S$.
Докажите, что в $S$ останется нечетное число.

В данном примере есть следующие объекты:

positive_integers -- класса Atom со свойствами:
    atom_name: 'positive_integers'
    type_name: 'set'
    content: 'axiomatic:positive_integers'

positive_integers_square -- класса Atom со свойствами:
    atom_name: 'positive_integers_square'
    type_name: 'set'
    content: 'axiomatic:cartesian_product(positive_integers,positive_integers)'

n -- класса Atom со свойствами:
    atom_name: 'n'
    type_name: 'number'
    content: 'axiomatic:odd'

comp -- класса Function со свойствами:
    function_name: 'comp'
    input_set_name: 'positive_integers_square'
    output_set_name: 'bool'
    realization: (x,y): return x<y

S -- класса Set со свойствами:
    set_name: 'S'
    object_type: 'number'
    content: ['positive_integers', comp(x,2*n)==1]

S_square -- класса Set со свойствами:
    set_name: 'S_square'
    object_type: 'number'
    content: 'axiomatic:cartesian_product(S,S)'

operation_function -- класса Function со свойствами:
    function_name: 'operation_function'
    input_set_name: 'S_square'
    output_set_name: 'positive_integers'
    realization: 
        (a,b): 
            S.erase(a)
            S.erase(b)
            S.insert(abs(a-b))

operation -- класса Action со свойствами:
    actor_name: 'operation'
    input_list: [a,b]
    function_name: 'operation_function'
'''
