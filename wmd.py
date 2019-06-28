from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import numpy as np
from fpdf import FPDF
from random import randint
# import pandas as pd

# df = pd.read_excel('HWest Responses_Venter.xlsx')
# feedbacks = df['Feedback']

# temp = open('./test_data.txt', 'w', encoding='utf-8')

# for feedback in feedbacks:
#     if type(feedback) == str:
#         print(feedback)
#         temp.writelines(feedback.split('\n')[0] + '\n')

# temp.close()

#wordmodelfile = 'E:/Me/IITB/Work/CIVIS/ML Approaches/word embeddings and similarity matrix/GoogleNews-vectors-negative300.bin'

wordmodelfile='/Users/Anushri-MacBook/GoogleNews-vectors-negative300.bin.gz'
wordmodel = KeyedVectors.load_word2vec_format(wordmodelfile, binary = True, limit=200000)

keywords = {
    'hw': ['bedbugs', 'cctv', 'pipeline', 'Open spaces', 'gutter', 'garbage',
                    'rats', 'mice', 'robbery', 'theft', 'passage', 'galli', 'lane',
                    'light', 'bathrooms not clean', 'toilets not clean', 'playarea', 'mosquito', 'fogging','water'],
}

hw_comments = open('test_data.txt', 'r', encoding='utf-8').readlines()

'''
#stemming keywords
ps = PorterStemmer()
print("Stemming of keywords...\n")
keyword_i=0
for keyword in keywords['hw']:
    keyword = keyword.strip()
    tokenized_word=word_tokenize(keyword)
    num=0
    for w in tokenized_word:
        if num<1:
            stemmed_word = ps.stem(w)
        else:
            stemmed_word+=' '+ps.stem(w)
        num+=1
    keywords['hw'][keyword_i] = stemmed_word
    keyword_i+=1
print("Keywords are:\n",keywords['hw'],'\n')
'''

#creating color dictionary for keywords
clr_dict={}
for keyword in keywords['hw']:
    clr_dict[keyword]=[randint(100,255),randint(100,255),randint(100,255)]

#####################hw test_data################################

distances = []

print('Initializing distances...')
pdf = FPDF()
pdf.add_page()
pdf.set_xy(0, 0)
pdf.set_font('arial', 'B', 12.0)
num = 0
for comment in hw_comments:
    num+=1
    comments = list(filter(None, comment.lower().split('.')))
    no_of_comments = len(comments)
    heading = "Comment "+str(num)
    pg_no = pdf.page_no()
    # checking for page no. in advance based on future formatting(line break) that will be done 
    # to ensure all sentences of comment are in the same page for each heading.
    if(pdf.ln(2*(no_of_comments+1))):
        if(pdf.page_no()!=pg_no):
            pdf.add_page()
    pdf.cell(h=5.0, align='L', w=0, txt=heading,  border=0, fill=False,ln=2)
    #print(pdf.page_no())
    for single_comment in comments:
        if len(single_comment) == 1:
            continue
        print(single_comment)

        #can't apply stemming as google negative pretrained vectors are used.
        '''
        single_comment = single_comment.strip()
        tokenized_word = word_tokenize(single_comment)
        num=0
        for w in tokenized_word:
            if num<1:
                stemmed_word = ps.stem(w)
            else:
                stemmed_word+=' '+ps.stem(w)
            num+=1
        print("Stemmed words: ",stemmed_word,'\n')
        '''
        cwords = single_comment.split()
        cwords = set(cwords)
        for word in cwords.copy():
            if word in stopwords.words('english'):
                cwords.remove(word)
        
        cwords = list(cwords)
        

        wmd = []
        for keyword in keywords['hw']:
            swords = keyword.split()
            swords = set(swords)
            for word in swords.copy():
                if word in stopwords.words('english'):
                    swords.remove(word)
            swords = list(swords)
            distance = wordmodel.wmdistance(cwords, swords)
            wmd.append(distance)

        distances.append(wmd)
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n")
        print("Array is: \n",np.array(wmd),'\n')
        minIndex = np.array(wmd).argmin()
        category = keywords['hw'][minIndex]
        r = int(clr_dict[category][0])
        g = int(clr_dict[category][1])
        b = int(clr_dict[category][2])
        single_comment = (single_comment+" : "+category).lstrip()
        #single_comment = single_comment+" : "+category
        pdf.set_fill_color(r,g,b)
        effective_page_width = pdf.w - 2*pdf.l_margin
        pdf.cell(h=5.0, align='L', w=effective_page_width, txt=single_comment, border=0, fill=True, ln=2)
        print(keywords['hw'][minIndex], wmd[minIndex]) 
        print('******************************\n\n')
    #pdf.ln(2)
    pdf.cell(h=5.0, align='L', w=effective_page_width, txt="   ",  border=0, fill=False,ln=2)
print('Done.')
pdf.output('beautified_response_added_keyword.pdf', 'F')

# print('\n\nPrinting Analysis...\n')
# for comment in env_comments:
#     comments = list(filter(None, comment.lower().split('.')))
#     for wmds,single_comment in zip(distances,comments):
#         print(single_comment)
#         print('-------------')
#         print(wmds)
#         wmds = np.array(wmds)
#         minIndex = wmds.argmin()
#         print(keywords['environment'][minIndex])
#         print(wmds[minIndex])
#         print('*************************************************')