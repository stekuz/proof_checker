def atom_to_graph(key,value):
    def json_record_to_text(key,value):
        text = ''
        text += key
        text += '\\\\\\'
        for key2 in value:
            text += key2
            text += '\\'
            text += value[key2]
            text += '\\\\'
        return text 
    separators = [' ',',','.',';',':','\\','\\\\','\\\\\\']
    text = json_record_to_text(key,value)
    adj_matrix = []
    n = len(text)
    for i in range(n):
        adj_matrix.append([0]*n)
    word_now = ''
    for i in range(n-1):
        if text[i+1] not in separators:
            adj_matrix[i][i+1] = 1
            word_now += text[i]
        else:
            index = i
            while True:
                index = text[index+1:].find(word_now)
                if index == -1:
                    break
                adj_matrix[i][index] = 1
    return (text,adj_matrix)
