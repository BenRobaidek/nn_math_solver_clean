import json
import numpy as np

def main():
    jsonToTsv('./draw-train.txt','./draw.json', './draw-train.tsv')
    jsonToTsv('./draw-dev.txt','./draw.json', './draw-dev.tsv')
    jsonToTsv('./draw-test.txt','./draw.json', './draw-test.tsv')

def jsonToTsv(indices_path, json_path, output_path):
    json_indices = np.genfromtxt(indices_path).astype(int)
    data = json.loads(open(json_path).read())
    output = open(output_path, 'w')
    for d in data:
        #print(d['iIndex'] in indices)
        if d['iIndex'] in json_indices:
            print(d['sQuestion'])

            # Preprocess Question
            tokens = np.array(d['sQuestion'].split())
            for a in d['Alignment']:
                indices = np.array([])
                indices = np.append(indices, np.where(tokens == '.')) # add . indicies
                indices = np.append(indices, np.where(tokens == '?')) ## add ? indicies
                indices += 1
                indices = np.append(indices, [0])
                indices.sort()
                tokens[int(indices[a['SentenceId']] + a['TokenId'])] = '[' + a['coeff'] + ']'
            for token in tokens:
                output.write(token + ' ')
            output.write('\t')

            # Preprocess Equations
            result = ''
            for eq in d['Template']:
                symbols = eq.split()
                for i,symbol in enumerate(symbols):
                    if symbol not in ['+', '-', '*', '/', '(', ')', '='] and not isFloat(symbol):
                        symbols[i] = '[' + symbol + ']'
                for symbol in symbols:
                    result += str(symbol) + ' '
                result += ' ; '
            result = result[:-3]
            output.write(result + '\n')

def isFloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

if __name__ == '__main__':
    main()
