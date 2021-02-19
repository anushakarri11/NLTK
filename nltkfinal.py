from nltk.tokenize import sent_tokenize, word_tokenize
import re, math
import nltk
from nltk.tokenize import PunktSentenceTokenizer, word_tokenize
from collections import Counter
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))
dataset = ["physics", "maths", "english_composition", "social", "science", "chemistry", "german_language", "french_language", "biology", "thermodynamics", "A", "B", "tennis", "hockey"]
ps = PorterStemmer()
WORD = re.compile(r'\w+')
########################################
def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
         return 0.0
    else:
         return float(numerator) / denominator

def text_to_vector(text):
     words = WORD.findall(text)
     return Counter(words)
l = 'null'
intersection = 0
list2 = []
output1 = 0
ab = 0
count = 0
temp = []
list3 = []
list4 = []
list5 = []
list6 = []
number = []
numbers = []
######################################
list1 = []
list2 = []


f = open('ques1.txt','r')
s = f.read()
text = s

list1 = word_tokenize(s)
for k in list1:
    if (k != "Description"):
        list2.append(k)
        if (k == 'Questions'):
            ab = 1
            break
list2.remove(k)
text1 = " ".join(map(str, list2))
########################
def project(text2, text3):
    options = ""
    option = []
    options = re.sub('[a-zA-Z:_()=?]', '', text3)
    for i in word_tokenize(options):
        option.append(float(i))
    s = text
    list3 = sent_tokenize(s)
    text2 = text2
    text3 = text3
    print(text2)
    print(text3)
    f.close()
    list5 = sent_tokenize(text1)
    output = re.sub('[a-zA-Z:_()=]', '', text1)
    #print(output)
    number = []
    s = sent_tokenize(output)
    for i in sent_tokenize(output):
        j = re.sub('[ .]', '', i)
        number.append(float(j))
    #print (number)
    maximum = max(number)
    sentence_index = []
    #sent_tokenize(text1)
    for i in sent_tokenize(text1):     
        sentence_index.append(i)
        #print(i)
        ps = PorterStemmer()
        #print(ps.stem(i))
        words = word_tokenize(i)
        stopped_sentence = []
        for w in words:
            if w not in stop_words:
                stopped_sentence.append(w)
                i = stopped_sentence
        #print(i)
    ####################################################################################################################################################################
    q = []
    Other_than_q = []
    s = text2;
    #f = open("../ques.txt",'r')
    #s = f.read();
    w = word_tokenize(s)
    tagged = []
    for i in w:
        words = nltk.word_tokenize(i)
        tagged.append(nltk.pos_tag(words))
    L = tagged
    for i in L:
        for j in dataset:
            vector1 = text_to_vector(i[0][0])
            vector2 = text_to_vector(j)
            cosine = get_cosine(vector1, vector2)
            #print(cosine)
            if (cosine > 0.0):
                q.append(j)
                #print (q)
    #print(q)
    max_size = len(q)

    count = 0
    #print(max_size)
    for k in sent_tokenize(text1):
        for h in word_tokenize(k):
            if (h == 'only'):
                count = count+1
    #################################################################################################################################################################           
    if (count == 2):
        if(max_size == 2):
            i = q[0]
            j = q[1]
            #print (i, j)
            for k in sent_tokenize(text1):
                   vector1 = text_to_vector(k)
                   vector2 = text_to_vector('only '+i)
                   cosine = get_cosine(vector1, vector2)
                   #print(cosine)
                   if (cosine > 0.5):
                       numofi = re.sub('[a-zA-Z _()=]', '', k)
                       num_of_i = float(numofi)
                       #print('only '+ i+' =',num_of_i)
                
            for k in sent_tokenize(text1):
                    vector1 = text_to_vector(k)
                    vector2 = text_to_vector('only '+j)
                    cosine = get_cosine(vector1, vector2)
                    #print(cosine)
                    if (cosine > 0.5):
                        numofj = re.sub('[a-zA-Z _()=]', '', k)
                        num_of_j = float(numofj)
                        #print('only '+j+' = ',num_of_j)
            intersection = maximum-num_of_i-num_of_j            
            union = num_of_i+num_of_j+intersection
            #print(i+' = ', num_of_i+intersection)
            #print(j+' = ', num_of_j+intersection)
            
            k = i+' and '+j
            l = i+' or '+j
            m = 'only '+i
            n = 'only '+j

            vector1 = text_to_vector(k)
            vector2 = text_to_vector(text2)
            cosine1 = get_cosine(vector1, vector2)

            vector1 = text_to_vector(l)
            vector2 = text_to_vector(text2)
            cosine2 = get_cosine(vector1, vector2)

            vector1 = text_to_vector(m)
            vector2 = text_to_vector(text2)
            cosine3 = get_cosine(vector1, vector2)
            
            vector1 = text_to_vector(n)
            vector2 = text_to_vector(text2)
            cosine4 = get_cosine(vector1, vector2)
            
            #print(cosine)
            if (cosine1 > 0.5):
                count = 0
                result = 4
                print (i+' and '+j+' = ',intersection)
                for h in option:
                    if (h == intersection):
                        result = count
                        break
                    else:
                        if(h != intersection):
                            count = count+1
                result = count            
                if(result == 0):
                    print ('A')
                elif(result == 1):
                    print('B')
                elif(result == 2):
                    print('C')
                elif(result == 3):
                    print ('D')
                elif(result == 4):
                    print('None of the above')
            elif(cosine2 > 0.35):
                count = 0
                print (i+' or '+j+' = ',union)
                for h in option:
                    if (h == union):
                        result = count
                        break
                    else:
                        if(h != union):
                            count = count+1
                result = count
                if(result == 0):
                    print ('A')
                elif(result == 1):
                    print('B')
                elif(result == 2):
                    print('C')
                elif(result == 3):
                    print ('D')
        ######################################################################################################################################
        elif(max_size == 1):
            i = q[0]
            temp = []
            for k in sentence_index:
                vector1 = text_to_vector(k)
                for j in dataset:
                    for l in q:
                        if (j != l):
                            vector2 = text_to_vector(j)
                            cosine = get_cosine(vector1, vector2)
                            #print ('Cosine: ',cosine)
                            if (cosine > 0.0):
                                temp.append(j)
                                #print(temp)
            for k in temp:
                if k not in Other_than_q:
                    Other_than_q.append(k)
            j = Other_than_q[0]
            for k in sent_tokenize(text1):
                   vector1 = text_to_vector(k)
                   vector2 = text_to_vector('only '+i)
                   cosine = get_cosine(vector1, vector2)
                   #print(cosine)
                   if (cosine > 0.5):
                       numofi = re.sub('[a-zA-Z _()=]', '', k)
                       num_of_i = float(numofi)
                       #print('only '+ i+' =',num_of_i)
            #print(num_of_i)
            for k in sent_tokenize(text1):
                    vector1 = text_to_vector(k)
                    vector2 = text_to_vector('only '+j)
                    cosine = get_cosine(vector1, vector2)
                    #print(cosine)
                    if (cosine > 0.5):
                        numofj = re.sub('[a-zA-Z _()=]', '', k)
                        num_of_j = float(numofj)
                        #print('only '+j+' = ',num_of_j)
            for w in word_tokenize(text2):
                if (w == i):
                    for h in word_tokenize(text2):
                        if(h == 'only'):
                            output = num_of_i
                            break
                        else:
                            output = maximum - num_of_j

                elif (w == j):
                    for h in word_tokenize(text2):
                        if(h == 'only'):
                            output = num_of_j
                            break
                        else:
                            output = maximum - num_of_i

            print (i+' : ',output)
            count = 0
            result = 4
            for h in option:
                if (h == output):
                    result = count
                    break
                else:
                    if(h != output):
                        count = count+1
            result = count
            if(result == 0):
                print ('A')
            elif(result == 1):
                print('B')
            elif(result == 2):
                print('C')
            elif(result == 3):
                print ('D')
            elif(result == 4):
                print('None of the above')
    #################################################################################################################################################################
    else:
        if(max_size == 1):
                temp = []
                for i in sentence_index:
                    vector1 = text_to_vector(i)
                    for j in dataset:
                        for k in q:
                            if (j != k):
                                vector2 = text_to_vector(j)
                                cosine = get_cosine(vector1, vector2)
                                #print ('Cosine: ',cosine)
                                if (cosine > 0.0):
                                    temp.append(j)
                                    #print(temp)
                for i in temp:
                    if i not in Other_than_q:
                        Other_than_q.append(i)
                        #print (Other_than_q)
                temp = 0
                for i in dataset:
                    for j in list3:
                        if i == j:
                            print (i)
                for i in sentence_index:
                    for k in Other_than_q:
                        vector1 = text_to_vector(i)
                        vector2 = text_to_vector(k)
                        cosine = get_cosine(vector1, vector2)
                        if (cosine > 0.0):
                            numofk = re.sub('[a-zA-Z _()=]', '', i)
                            num_of_OQ = float(numofk)
                            #print ( k, ':', num_of_OQ)
                            temp = 1
                    if(temp == 1):
                        break
                temp = 0
                result = 4
                for i in q:
                    for j in Other_than_q:
                        conj = i+' and '+j
                        disj = i+' or '+j
                for i in sentence_index:
                    for k in q:
                        vector1 = text_to_vector(i)
                        vector2 = text_to_vector(k)
                        cosine = get_cosine(vector1, vector2)
                        if (cosine > 0.0):
                            numofk = re.sub('[a-zA-Z _()=]', '', i)
                            num_of_Q = float(numofk)
                            #print ( k, ':', num_of_Q)
                            temp = 1
                    if(temp == 1):
                        break
                for i in sent_tokenize(text1):
                    vector1 = text_to_vector(i)
                    vector2 = text_to_vector(conj)
                    cosine1 = get_cosine(vector1, vector2)
                    #print(i, cosine1)
                    if(cosine1 > 0.5):
                        for h in word_tokenize(conj):
                            #print(h)
                            if(h == 'and'):
                                intersection = re.sub('[a-zA-Z _()=]', '', i)
                                intersection = float(intersection)
                                #print(h, intersection)
                    else:
                        intersection = num_of_Q+num_of_OQ-maximum
                if(cosine1 < 0.7):
                    for i in sent_tokenize(text1):
                        vector1 = text_to_vector(i)
                        vector2 = text_to_vector(disj)
                        cosine2 = get_cosine(vector1, vector2)
                        #print(i, cosine2)
                        if(cosine2 > 0.7):
                            union = re.sub('[a-zA-Z _()=]', '', i)
                            union = float(union)
                            intersection = num_of_Q+num_of_OQ-union
                union = num_of_Q+num_of_OQ-intersection
                ##########################################################################################
                #print ("Total = ",maximum)
                vector1 = text_to_vector(text2)
                vector2 = text_to_vector('total')
                cosine = get_cosine(vector1, vector2)
                #print ('Cosine :' ,j,cosine)
                result = 4
                if (cosine > 0.0):
                    count = 0
                    for h in option:
                        if (h == maximum):
                            result = count
                        else:
                            if(h != maximum):
                                count = count+1
                    result = count            
                    if(result == 0):
                        print ('A')
                    elif(result == 1):
                        print('B')
                    elif(result == 2):
                        print('C')
                    elif(result == 3):
                        print ('D')
                    elif(result == 4):
                       print ('None of the above options')
                temp = 1
                for i in q:
                    for j in Other_than_q:
                        #print ( i, 'and', j, ':', intersection)
                        vector1 = text_to_vector(text2)
                        vector2 = text_to_vector(i+' and '+j)
                        cosine = get_cosine(vector1, vector2)
                        #print ('Cosine :' ,j,cosine)
                    for j in word_tokenize(text2):
                        if (j == 'only'):
                            temp = temp*0
                        else:
                            temp = temp*1
                for k in q:
                    if(temp == 0):
                         print ('only '+ k, ':', num_of_Q-intersection)
                         count = 0
                         for h in option:
                            if (h == num_of_Q-intersection):
                                result = count
                          #      print (result)
                            else:
                                if(h != num_of_Q-intersection):
                                    count = count+1
                         #result = count
                         #print(result)
                         if(result == 0):
                             print ('A')
                         elif(result == 1):
                             print('B')
                         elif(result == 2):
                             print('C')
                         elif(result == 3):
                             print ('D')
                         elif(result == 4):
                             print('None of the above options')
                    else:
                         print ( k, ':', num_of_Q)
                         count = 0
                         for h in option:
                            if (h == num_of_Q):
                                result = count
                                #print(result)
                            else:
                                if(h != num_of_Q):
                                    count = count+1
                         #result = count
                         if(result == 0):
                            print('A')
                         elif(result == 1):
                            print('B')
                         elif(result == 2):
                            print('C')
                         elif(result == 3):
                            print ('D')
                         elif(result == 4):
                            print ('None of the above options')
        ###########################################################################################################################################
        if(max_size == 2):
            i = q[0]
            j = q[1]
            #print (i, j)
            
            def intersection1():    
                    output1 = 0.0
                    num_of_q1 = 0
                    num_of_q2 = 0
                    i = q[0]
                    j = q[1]
                    k = i+" and "+j
                    #print(k)
                    for k in sent_tokenize(text1):
                        vector1 = text_to_vector(k)
                        vector2 = text_to_vector(i)
                        cosine = get_cosine(vector1, vector2)
                        #print(cosine)
                        if (cosine > 0.0):
                            numofq1 = re.sub('[a-zA-Z _()=]', '', k)
                            num_of_q1 = float(numofq1)
                            #print(i, ':', num_of_q1)
                    for k in sent_tokenize(text1):
                        vector1 = text_to_vector(k)
                        vector2 = text_to_vector(j)
                        cosine = get_cosine(vector1, vector2)
                        #print(cosine)
                        if (cosine > 0.0):
                            numofq2 = re.sub('[a-zA-Z _()=]', '', k)
                            num_of_q2 = float(numofq2)
                            #print(j,' : ',num_of_q2)
                    for k in sent_tokenize(text1):
                        vector1 = text_to_vector(k)
                        vector2 = text_to_vector(l)
                        cosine2 = get_cosine(vector1, vector2)
                        #print(cosine2)
                    if(cosine2 > 0.3):
                        union = re.sub('[a-zA-Z _()=]', '', k)
                        union = float(union)
                    else:
                        union = maximum
                    intersection = num_of_q1+num_of_q2-union
                    print(i+' and '+j+' : ',intersection)
                    result = 4
                    count = 0
                    for h in option:
                        if (h == intersection):
                            result = count
                            #print(result)
                        else:
                            if(h != intersection):
                                count = count+1
                         #result = count
                    if(result == 0):
                       print('A')
                    elif(result == 1):
                       print('B')
                    elif(result == 2):
                       print('C')
                    elif(result == 3):
                       print ('D')
                    elif(result == 4):
                       print ('None of the above options')
                    #exit()
            def union1():
                    i = q[0]
                    j = q[1]
                    k = i+" or "+j
                    for k in sent_tokenize(text1):
                        vector1 = text_to_vector(k)
                        vector2 = text_to_vector(i)
                        cosine = get_cosine(vector1, vector2)
                        #print(cosine)
                        if (cosine > 0.0):
                            numofq1 = re.sub('[a-zA-Z _()=]', '', k)
                            num_of_q1 = float(numofq1)
                    for k in sent_tokenize(text1):
                        vector1 = text_to_vector(k)
                        vector2 = text_to_vector(j)
                        cosine = get_cosine(vector1, vector2)
                        #print(cosine)
                        if (cosine > 0.0):
                            numofq2 = re.sub('[a-zA-Z _()=]', '', k)
                            num_of_q2 = float(numofq2)
                    union = num_of_q1+num_of_q2-(num_of_q1+num_of_q2-maximum)
                    print(l,' :',union)
                    result = 4
                    count = 0
                    for h in option:
                        if (h == union):
                            result = count
                            #print(result)
                        else:
                            if(h != union):
                                count = count+1
                         #result = count
                    if(result == 0):
                       print('A')
                    elif(result == 1):
                        print('B')
                    elif(result == 2):
                       print('C')
                    elif(result == 3):
                       print ('D')
                    elif(result == 4):
                       print ('None of the above options')
                    #exit()
            result = 4
            k = i+' and '+j
            l = i+' or '+j
            vector1 = text_to_vector(text2)
            vector2 = text_to_vector(k)
            cosine1 = get_cosine(vector1, vector2)
            vector2 = text_to_vector(l)
            cosine2 = get_cosine(vector1, vector2)    
            #print(cosine1, cosine2)
            if(cosine1 > 0.5):
                for i in sent_tokenize(text1):
                    vector1 = text_to_vector(i)
                vector2 = text_to_vector(k)
                cosine1 = get_cosine(vector1, vector2)
                #print(i, cosine1)
                if(cosine1 > 0.5):
                    intersection = re.sub('[a-zA-Z _()=]', '', i)
                    intersection = float(intersection)
                    print(k,' : ',intersection)
                    for h in option:
                        if (h == intersection):
                            result = count
                            #print(result)
                        else:
                            if(h != intersection):
                                count = count+1
                         #result = count
                    if(result == 0):
                       print('A')
                    elif(result == 1):
                        print('B')
                    elif(result == 2):
                       print('C')
                    elif(result == 3):
                       print ('D')
                    elif(result == 4):
                       print ('None of the above options')
                    #exit()
                else:
                    intersection1()

            if(cosine2 > 0.5):
                for i in sent_tokenize(text1):
                    vector1 = text_to_vector(i)
                vector2 = text_to_vector(l)
                cosine2 = get_cosine(vector1, vector2)
                #print(i, cosine2)
                if(cosine2 > 0.5):
                    union = re.sub('[a-zA-Z _()=]', '', i)
                    union = float(union)
                    print(l,' : ',union)
                    for h in option:
                        if (h == union):
                            result = count
                            #print(result)
                        else:
                            if(h != union):
                                count = count+1
                         #result = count
                    if(result == 0):
                       print('A')
                    elif(result == 1):
                        print('B')
                    elif(result == 2):
                       print('C')
                    elif(result == 3):
                       print ('D')
                    elif(result == 4):
                       print ('None of the above options')
                    #exit()
                else:
                    union1()
 ###################################################################################################################################################################                   
q=0
tet2 = ""
tet3= ""
tet = ""
for i in sent_tokenize(s):
    temp_var = i
    if (q==2):
        tet3,tet = temp_var.split('*')
        project(tet2,tet3)
        print('_______________________________________________________________________')
        tet2=tet
        
    if(q==1):
        tet2 = i
        q+=1
    if("Questions" in i):
        q=1
        print(" OUTPUT ")
        print("")
