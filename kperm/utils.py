def write_list_of_tuples(filename, data):
    with open(filename, 'w') as f:
        for d in data:
            line = ' '.join([str(x) for x in d]) + '\n'
            f.write(line)
