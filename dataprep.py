import csv,json,os,re 

def create_dataset(path):
    text  =''
    with open(path,'r') as f:
        reader = csv.reader(f,delimiter = "\n")
        next(reader, None)
        for row in reader:
            row = ''.join(row)
            row = re.sub(r'[^a-zA-Z0-9:,.]', ' ', row)
            row = re.sub(r'\s+',' ',row)
            row = row.split(',')
            s = ''
            s += row[1] +':\n'
            s += ''.join(row[2:-1]) + '\n'
            
            s= re.sub(r'[.]+',' ',s)
            text += s
            
    return text

def prepText(text,path):
    file = open(os.path.join(path,'dataset.txt'),'w+')
    file.write(text)
        
def readTxt(path):
    file = open(path,'rb').read().decode(encoding='utf_8')
    return file
            
def readTxtwithn(path,n):
    file = open(path,'r')
    text = file.readline([n])
    return text

def loadHyperParameters(path):
    with open(path,'r') as f :
        param = json.load(f)
    return param
    
base = os.getcwd()
base = os.path.join(base,"LoTR-Script-Generator")
datadir = os.path.join(base,"data")
csvDir = os.path.join(datadir,"lotr_scripts.csv")

