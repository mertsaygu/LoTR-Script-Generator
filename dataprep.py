import csv,json,os,re 

def create_dataset(path):
    # Preprocessing of 'lotr_scripts.csv' Returns a string of formatted dataset
    text  =''
    with open(path,'r') as f:
        reader = csv.reader(f,delimiter = "\n")
        next(reader, None)
        for row in reader:
            row = ''.join(row) 
            row = re.sub(r'[^a-zA-Z0-9:,.]', ' ', row)   # Replacing every char with empty space except lowercase letters, upper case letters, numbers, comma, period and colon 
            row = re.sub(r'\s+',' ',row)   # Replacing multiple white spaces with single white space
            row = row.split(',')   # Index still exists in the row. To get rid of it, split the row by comma
            s = ''
            s += row[1] +':\n' #  First index is where the Character name is. Then add colon and go to next line 
            s += ''.join(row[2:-1]) + '\n'  #  Add dialog.
            
            s= re.sub(r'[.]+',' ',s)
            text += s
            
    return text  

def prepText(text,path):
    # Creates a file to given path and writes given text in it 
    file = open(os.path.join(path,'dataset.txt'),'w+')
    file.write(text)
        
def readTxt(path):
    # Reads the file from given path and returns the file
    # NOTE: never used
    file = open(path,'rb').read().decode(encoding='utf_8')
    return file
            
def readTxtwithn(path,n):
    # Reads n lines of the text and returns the text 
    # NOTE: never used
    file = open(path,'r')
    text = file.readline([n])
    return text

def loadHyperParameters(path):
    # Returns a JSON object 
    with open(path,'r') as f :
        param = json.load(f)
    return param
    
base = os.getcwd()
base = os.path.join(base,"LoTR-Script-Generator")
datadir = os.path.join(base,"data")
csvDir = os.path.join(datadir,"lotr_scripts.csv")

