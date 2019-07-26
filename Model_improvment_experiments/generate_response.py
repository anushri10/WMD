from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import pandas as pd 
import numpy as np

raw_data = pd.read_excel('C:/Users/anush/Desktop/IIT_B/SentenceSimilarity-master/HWest Responses_Venter.xlsx',skiprows=1)
response = raw_data['Feedback']
response=response.dropna()
print("Type is: ",type(response))
print("Null values are: ", response.isnull().sum())
tokenized_text=[]
current_text=[]
file2= open('all_sent_HWest_check2.txt','w')
for item in response:
    tokenized_text+=sent_tokenize(item)
    current_text=sent_tokenize(item)
    for sent in current_text:
        file2.write(sent)
    file2.write('\n')
file2.close()
file1= open('all_sent_HWest_Check.txt','w')
for sent in tokenized_text:
    file1.write(sent)
    file1.write('\n')
file1.close()

#print(tokenized_text)
# response.to_csv('response.csv',index=False)

# response = pd.read_csv('response.csv')
# tokenized_text=[]
# num=1
# for index, row in response.iterrows():
# 	#if len(tokenized_text)<40:
#     data = row[0]
#     tokenized_text+=sent_tokenize(data)
#     #print(index)
#         #df = pd.DataFrame(tokenized_text)
#         #name ='comment'+str(num)+'.txt'
#         #path = './comments/'+name
#         #df.to_csv(path,header=None, index=None, sep=' ', mode='a')
#     #num+=1
# my_df = pd.DataFrame(tokenized_text)
# name ='sentence.csv'
# path = './comments/'+name
# my_df.to_csv(path, index=False, header=False)    

#my_df.to_csv('all_sent_HWest.txt',header=None, index=None, sep=' ', mode='a')