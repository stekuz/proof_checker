class Action:
    actor_name = ''
    input_list = []
    function_name = ''
    
    def __init__(self, actor_name, input_list, function_name):
        self.actor_name = actor_name
        self.input_list = input_list
        self.function_name = function_name
    
    def result(self):
        return functions[self.function_name](self.input_list)

class Set:
    set_name = ''
    objects_type = ''
    content = {}

    def __init__(self, set_name, objects_type, content={}):
        self.set_name = set_name
        self.objects_type = objects_type
        self.content = content.copy()

    def insert(self, element):
        self.content.add(element)
    
    def erase(self, element):
        self.content.remove(element)

class Function:
    function_name = ''
    input_set_name = ''
    output_set_name = ''
    def realization(self):
        return 0

    def __init__(self, function_name, input_set_name, output_set_name, realization):
        self.function_name = function_name
        self.input_set_name = input_set_name
        self.output_set_name = output_set_name
        self.realization = realization

class Atom:
    atom_name = ''
    type_name = ''
    content = ''

    def __init__(self, atom_name, type_name, content):
        self.atom_name = atom_name
        self.type_name = type_name
        self.content = content

functions = {}
