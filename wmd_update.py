from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import numpy as np
from fpdf import FPDF
from random import randint
from pyemd import emd
from gensim.similarities import WmdSimilarity
import os
import time
import networkx
import numpy as np
from gensim.models import KeyedVectors
from networkx.algorithms.components.connected import connected_components
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
import json

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

# wordmodelfile = "C:/Users/anush/Desktop/Venter_CMS-master/Venter/ML_model/Civis/MAX.bin"
# wordmodel = KeyedVectors.load_word2vec_format(wordmodelfile, binary = True, limit=200000)

# keywords = {
#     'hw': ['bedbugs', 'cctv', 'pipeline', 'Open spaces', 'gutter', 'garbage',
#                     'rats', 'mice', 'robbery', 'theft', 'passage', 'galli', 'lane',
#                     'light', 'bathrooms not clean', 'toilets not clean', 'playarea', 'mosquito', 'fogging','water'],
# }

# hw_comments = open('test_data.txt', 'r', encoding='utf-8').readlines()

# #creating color dictionary for keywords
# clr_dict={}
# for keyword in keywords['hw']:
#     clr_dict[keyword]=[randint(100,255),randint(100,255),randint(100,255)]

# #####################hw test_data################################

# distances = []

# print('Initializing distances...')
# pdf = FPDF()
# pdf.add_page()
# pdf.set_xy(0, 0)
# pdf.set_font('arial', 'B', 12.0)
# num = 0
# for comment in hw_comments:
#     num+=1
#     comments = list(filter(None, comment.lower().split('.')))
#     no_of_comments = len(comments)
#     heading = "Comment "+str(num)
#     pg_no = pdf.page_no()
#     # checking for page no. in advance based on future formatting(line break) that will be done 
#     # to ensure all sentences of comment are in the same page for each heading.
#     if(pdf.ln(2*(no_of_comments+1))):
#         if(pdf.page_no()!=pg_no):
#             pdf.add_page()
#     pdf.cell(h=5.0, align='L', w=0, txt=heading,  border=0, fill=False,ln=2)
#     #print(pdf.page_no())
#     for single_comment in comments:
#         if len(single_comment) == 1:
#             continue
#         print(single_comment)
        
#         cwords = single_comment.split()
#         cwords = set(cwords)
#         for word in cwords.copy():
#             if word in stopwords.words('english'):
#                 cwords.remove(word)
        
#         cwords = list(cwords)
        

#         wmd = []
#         # for domain in keywords:
#         #     for domain_keyword in keywords[domain]:
#         for keyword in keywords['hw']:
#             swords = keyword.split()
#             swords = set(swords)
#             for word in swords.copy():
#                 if word in stopwords.words('english'):
#                     swords.remove(word)
#             swords = list(swords)
#             distance = wordmodel.wmdistance(cwords, swords)
#             print("distance is: \n",distance)
#             wmd.append(distance)

#         distances.append(wmd)
#         print("%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n")
#         print("Array is: \n",np.array(wmd),'\n')
#         minIndex = np.array(wmd).argmin()
#         category = keywords['hw'][minIndex]
#         r = int(clr_dict[category][0])
#         g = int(clr_dict[category][1])
#         b = int(clr_dict[category][2])
#         single_comment = (single_comment+" : "+category).lstrip()
#         #single_comment = single_comment+" : "+category
#         pdf.set_fill_color(r,g,b)
#         effective_page_width = pdf.w - 2*pdf.l_margin
#         pdf.cell(h=5.0, align='L', w=effective_page_width, txt=single_comment, border=0, fill=True, ln=2)
#         print(keywords['hw'][minIndex], wmd[minIndex]) 
#         print('******************************\n\n')
#     #pdf.ln(2)
#     pdf.cell(h=5.0, align='L', w=effective_page_width, txt="   ",  border=0, fill=False,ln=2)
# print('Done.')
# pdf.output('beautified_response_added_keyword.pdf', 'F')

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

# results = {}
# for domain in keyword.keys():
#     results[domain]={}
#     for category in keywords.values():
#         results[domain][category]=[]
    
#     results[domain][keywords[domain][minIndex]].append(temp)


def similarityIndex(s1, s2, wordmodel):
    
    #To compare the two sentences for their similarity using the gensim wordmodel 
    #and return a similarity index
    
    if s1 == s2:
        return 1.0

    s1words = s1.split()
    s2words = s2.split()

    s1words = set(s1words)
    for word in s1words.copy():
        if word in stopwords.words('english'):
            s1words.remove(word)
    
    s2words = set(s2words)
    for word in s2words.copy():
        if word in stopwords.words('english'):
            s2words.remove(word)

    s1words = list(s1words)
    s2words = list(s2words)

    s1set = set(s1words)
    s2set = set(s2words)

    vocab = wordmodel.vocab
    
    if len(s1set & s2set) == 0:
        return 0.0
    for word in s1set.copy():
        if (word not in vocab):
            s1words.remove(word)
    for word in s2set.copy():
        if (word not in vocab):
            s2words.remove(word)
    
    return wordmodel.n_similarity(s1words, s2words)


def wmd_similarity(single_comment,single_keyword,wordmodel):
    
    cwords = single_comment.split()
    cwords = set(cwords)
    for word in cwords.copy():
        if word in stopwords.words('english'):
            cwords.remove(word)
        
    cwords = list(cwords)

    swords = single_keyword.split()
    swords = set(swords)
    for word in swords.copy():
        if word in stopwords.words('english'):
            swords.remove(word)

    swords = list(swords)
    
    return wordmodel.wmdistance(cwords, swords)

def categorizer():
    
    #driver function,
    #returns model output mapped on the input corpora as a dict object
    
    stats = open('stats.txt', 'w', encoding='utf-8')

    st = time.time()
    
    wordmodelfile = "C:/Users/anush/Desktop/Venter_CMS-master/Venter/ML_model/Civis/MAX.bin"
    wordmodel = KeyedVectors.load_word2vec_format(wordmodelfile, binary = True, limit=200000)

    keywords = {
    'test_data': ['bedbugs', 'cctv', 'pipeline', 'Open spaces', 'gutter', 'garbage',
                    'rats', 'mice', 'robbery', 'theft', 'passage', 'galli', 'lane',
                    'light', 'bathrooms not clean', 'toilets not clean', 'playarea', 'mosquito', 'fogging','water'],
    }
    #wordmodelfile = os.path.join(BASE_DIR, 'Venter/ML_model/Civis/MAX.bin')
    wordmodel = KeyedVectors.load_word2vec_format(wordmodelfile, binary=True, limit=200000)
    et = time.time()
    s = 'Word embedding loaded in %f secs.' % (et-st)
    print(s)
    stats.write(s + '\n')

    #filepaths
    #responsePath = os.path('./comments/')
    responsePath=('./comments/')
    responseDomains = os.listdir('./comments/')
    #responseDomains.sort()
    
    #dictionary for populating the json output
    results = {}
    for responseDomain in zip(responseDomains):
        #instantiating the key for the domain
        responseDomain=str(responseDomain)
        domain=responseDomain[2:-7]
        responseDomain=responseDomain[2:-3]
        #domain = responseDomain[:-4]
        print("ResponseDomain is: ",responseDomain)
        print("Domain is: ",domain)
        results[domain] = {}

        print('Categorizing %s domain...' % domain)

        temp = open(os.path.join(responsePath, responseDomain), 'r', encoding='utf-8-sig')
        responses = temp.readlines()
        rows=0
        for response in responses:
            response = list(filter(None, response.lower().split('.'))) 
            num=0
            if '\n' in response:
                num+=1
            rows+=(len(response)-num)

        categories=keywords[domain]
        columns = len(categories)

        #categories = category
        #saving the scores in a similarity matrix
        #initializing the matrix with -1 to catch dump/false entries
        st = time.time()
        similarity_matrix = [[-1 for c in range(columns)] for r in range(rows)]
        et = time.time()
        s = 'Similarity matrix initialized in %f secs.' % (et-st)
        print(s)
        stats.write(s + '\n')

        row = 0
        st = time.time()
        for response in responses:
            response = list(filter(None, response.lower().split('.'))) 
            print("Row: ",row)
            for single_response in response:
                print("Current sentence is: ",single_response)
                if len(single_response) == 1:
                    continue
                #print(single_response)
                if single_response=='\n':
                    continue
                else:
                    column = 0
                    for category in categories:
                        print("Current category is: ",category)
                        similarity_matrix[row][column] = wmd_similarity(single_response, category, wordmodel)
                        column += 1
            row += 1
        et = time.time()
        s = 'Similarity matrix populated in %f secs. ' % (et-st)
        print(s)
        stats.write(s + '\n')

        print('Initializing json output...')
        for catName in categories:
            results[domain][catName] = []

        print('Populating category files...')
        for score_row, response in zip(similarity_matrix, responses):
            #max_sim_index = len(categories)-1
            response = list(filter(None, response.lower().split('.'))) 
            for single_response in response:
                if single_response!='\n':
                    print("Current score row: \n",np.array(score_row))
                    min_sim_index=len(categories)-1
                #if np.array(score_row).sum() > 0:
                    min_sim_index = np.array(score_row).argmin()
                    temp = {}
                    temp['response'] = single_response
                    temp['score'] = float((np.array(score_row).min()))
            # else:
                    #temp = response
                    results[domain][categories[min_sim_index]].append(temp)
        print('Completed.\n')

        #sorting domain wise categorised responses based on scores
        for domain in results:
            for category in results[domain]:                                                                                                                                      
                temp = results[domain][category]
                if len(temp)==0 or category=='Novel':
                    continue
                #print(temp)
                results[domain][category] = sorted(temp, key=lambda k: k['score'], reverse=True)
        #newlist = sorted(list_to_be_sorted, key=lambda k: k['name']) --> to sort list of dictionaries

        print('***********************************************************') 

        with open('out_new_2.json', 'w') as temp:
            json.dump(results, temp)
    
    return results



if __name__=="__main__":
    results = categorizer()
    print(len(results['test_data']))