from django.shortcuts import render
from django.http.response import HttpResponseRedirect, HttpResponse
from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage
from myapp.models import Board, BoardComment
import datetime
from ipware.ip import get_client_ip
import json
import matplotlib.pyplot as plt
import os
import seaborn as sns
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.graphics.factorplots import interaction_plot
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from pandas.plotting import table 
from sklearn.tree._classes import DecisionTreeRegressor
from sklearn.ensemble._forest import RandomForestRegressor
from xgboost.sklearn import XGBRegressor
plt.rc('font', family='malgun gothic') # 한글 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False 
# Create your views here.
def IndexFunc(request):
    return render(request, 'index.html')

def AboutFunc(request):
    return render(request, 'about.html')

def ContentFunc(request):
    return render(request, 'content.html')

def BoardFunc(request):
    search_value = request.GET.get('value')
    print(search_value)
    if search_value == None:
        datas = Board.objects.all().order_by('-board_id')
    else:
        datas = Board.objects.all().filter(title__contains = search_value).order_by('board_id')
        
    paginator = Paginator(datas, 5) #django에 있는 페이징
    try:
        page = request.GET.get('page')
    except:
        page = 1
    
    try :
        data = paginator.page(page)
    except PageNotAnInteger:
        data = paginator.page(1)
    except EmptyPage:
        data = paginator.page(paginator.num_pages())
    
    #개별 페이지 표시용 작업
    allpage = range(paginator.num_pages + 1)
    #print('allpage:',allpage)
        
    return render(request, 'board.html', {'board_datas':data, 'allpage':allpage, 'value':search_value})

def BoardWriteFunc(request):
    return render(request, 'board_write.html')

def BoardSaveFunc(request):
    if request.method=='POST':
        Board(
            title = request.POST.get('board_write_title'),
            content = request.POST.get('message'),
            name = request.POST.get('board_write_name'),
            pw = request.POST.get('board_write_passwd'),
            ip = get_client_ip(request)[0],
            date = datetime.datetime.today(),
        ).save() # ORM에서의 추가
        return HttpResponseRedirect('board')
        
    else:
        return HttpResponseRedirect('board_write')
    
def BoardViewFunc(request):
    datas = Board.objects.all().get(pk=request.GET.get('board_view_no'))
    viewsData = Board.objects.get(pk=request.GET.get('board_view_no'))
    Board.objects.filter(pk=request.GET.get('board_view_no')).update(
        views = viewsData.views+1
    )
    datas2 = BoardComment.objects.all().filter(board = request.GET.get('board_view_no'))
    
    return render(request, 'board_view.html', {'board_view_datas':datas, 'board_comment_datas':datas2})

def BoardPasswdCheckFunc(request):
    datas =  Board.objects.get(pk = request.POST.get('board_no'))
    if datas.pw == request.POST.get('board_passwd'):
        result = {'result':"success"}
    else:
        result = {'result':"false"}
    return HttpResponse(json.dumps(result), content_type='application/json')

def BoardUpdateFunc(request):
    datas = Board.objects.get(pk=request.GET.get('board_no'))
    return render(request, 'board_update.html', {'datas':datas})

def BoardUpdateSaveFunc(request):
    datas = Board.objects.all().get(pk=request.POST.get('board_no'))
    if request.method=='POST':
        Board.objects.filter(pk=request.POST.get('board_no')).update(
            content = request.POST.get('message'),
        )
        return HttpResponseRedirect('board_view?board_view_no={}'.format(request.POST.get('board_no')))
    else:
        return HttpResponseRedirect('board_update')

def BoardDeleteFunc(request):
    Board.objects.get(pk = request.GET.get('board_no')).delete()
    return HttpResponseRedirect('board')

def CommentPasswdCheckFunc(request):
    datas =  BoardComment.objects.get(pk = request.POST.get('comment_no'))
    if datas.pw == request.POST.get('comment_passwd2'):
        result = {'result':"success"}
    else:
        result = {'result':"false"}
    return HttpResponse(json.dumps(result), content_type='application/json')

def CommentDeleteFunc(request):
    BoardComment.objects.get(pk = request.GET.get('comment_no')).delete()
    return HttpResponseRedirect('board_view?board_view_no={}'.format(request.GET.get('board_no')))

def CommentSaveFunc(request):
    requestpk = Board.objects.get(pk = request.POST.get('board_no2'))
    print(requestpk.board_id)
    if request.method=='POST':
        BoardComment(
            board = requestpk,
            content = request.POST.get('comment'),
            name = request.POST.get('comment_name'),
            ip = get_client_ip(request)[0],
            pw = request.POST.get('comment_passwd'),
            date = datetime.datetime.today(),
        ).save() # ORM에서의 추가
        return HttpResponseRedirect('board_view?board_view_no={}'.format(request.POST.get('board_no2')))
         
    else:
        return HttpResponseRedirect('board')
#-----------------------------이대희






import pickle
from django.contrib.staticfiles import finders
def ShowWordCloudFunc(request):
    with open(finders.find('datafiles/pickles/france_titledata.pickle'), 'rb') as f:
        data = pickle.load(f)
        frtt = []
        cnt = 0
        for k, v in data.items():
            cnt += 1
            if cnt > 10:
                break
            frtt.append((k, v))
        
    with open(finders.find('datafiles/pickles/france_tagsdata.pickle'), 'rb') as f:
        data = pickle.load(f)
        frtg = []
        cnt = 0
        for k, v in data.items():
            cnt += 1
            if cnt > 10:
                break
            frtg.append((k, v))
     
    with open(finders.find('datafiles/pickles/canada_titledata.pickle'), 'rb') as f:
        data = pickle.load(f)
        catt = []
        cnt = 0
        for k, v in data.items():
            cnt += 1
            if cnt > 10:
                break
            catt.append((k, v))
         
    with open(finders.find('datafiles/pickles/canada_tagsdata.pickle'), 'rb') as f:
        data = pickle.load(f)
        catg = []
        cnt = 0
        for k, v in data.items():
            cnt += 1
            if cnt > 10:
                break
            catg.append((k, v))
 
    with open(finders.find('datafiles/pickles/germany_titledata.pickle'), 'rb') as f:
        data = pickle.load(f)
        gett = []
        cnt = 0
        for k, v in data.items():
            cnt += 1
            if cnt > 10:
                break
            gett.append((k, v))
 
    with open(finders.find('datafiles/pickles/germany_tagsdata.pickle'), 'rb') as f:
        data = pickle.load(f)
        getg = []
        cnt = 0
        for k, v in data.items():
            cnt += 1
            if cnt > 10:
                break
            getg.append((k, v))
 
    with open(finders.find('datafiles/pickles/korea_titledata.pickle'), 'rb') as f:
        data = pickle.load(f)
        krtt = []
        cnt = 0
        for k, v in data.items():
            cnt += 1
            if cnt > 10:
                break
            krtt.append((k, v))
         
    with open(finders.find('datafiles/pickles/korea_tagsdata.pickle'), 'rb') as f:
        data = pickle.load(f)
        krtg = []
        cnt = 0
        for k, v in data.items():
            cnt += 1
            if cnt > 10:
                break
            krtg.append((k, v))
 
    with open(finders.find('datafiles/pickles/russia_titledata.pickle'), 'rb') as f:
        data = pickle.load(f)
        rutt = []
        cnt = 0
        for k, v in data.items():
            cnt += 1
            if cnt > 10:
                break
            rutt.append((k, v))
 
    with open(finders.find('datafiles/pickles/russia_tagsdata.pickle'), 'rb') as f:
        data = pickle.load(f)
        rutg = []
        cnt = 0
        for k, v in data.items():
            cnt += 1
            if cnt > 10:
                break
            rutg.append((k, v))
 
    with open(finders.find('datafiles/pickles/uk_titledata.pickle'), 'rb') as f:
        data = pickle.load(f)
        uktt = []
        cnt = 0
        for k, v in data.items():
            cnt += 1
            if cnt > 10:
                break
            uktt.append((k, v))
 
    with open(finders.find('datafiles/pickles/uk_tagsdata.pickle'), 'rb') as f:
        data = pickle.load(f)
        uktg = []
        cnt = 0
        for k, v in data.items():
            cnt += 1
            if cnt > 10:
                break
            uktg.append((k, v))
 
    with open(finders.find('datafiles/pickles/usa_titledata.pickle'), 'rb') as f:
        data = pickle.load(f)
        ustt = []
        cnt = 0
        for k, v in data.items():
            cnt += 1
            if cnt > 10:
                break
            ustt.append((k, v))
 
    with open(finders.find('datafiles/pickles/usa_tagsdata.pickle'), 'rb') as f:
        data = pickle.load(f)
        ustg = []
        cnt = 0
        for k, v in data.items():
            cnt += 1
            if cnt > 10:
                break
            ustg.append((k, v))
            
    with open(finders.find('datafiles/pickles/korean_from_others_titledata.pickle'), 'rb') as f:
        data = pickle.load(f)
        korea = []
        cnt = 0
        for k, v in data.items():
            cnt += 1
            if cnt > 11:
                break
            korea.append((k, v))
            
    return render(request, 'wordclouds_result.html', {'frtt':frtt, 'frtg':frtg, 'catt':catt, 'catg':catg, 'gett':gett, 'getg':getg, 'krtt':krtt, 'krtg':krtg, 'rutt':rutt, 'rutg':rutg, 'uktt':uktt, 'uktg':uktg, 'ustt':ustt, 'ustg':ustg, 'korea':korea})


''' ===== 파일이 인코딩 문제로 불러와지지 않아서 새로 파일을 만들어서 저장하기 ====='''
'''CSV 파일 재저장, django가 아니라 파이썬 프로젝트로 실행하는 것을 추천
import csv
trending_date = []
title = []
channel_title = []
category_id = []
publish_time = []
tags = []
views = []
likes = []
dislikes = []
comment_count = []
description = []
with open('./trending/RUvideos.csv', 'r', encoding='utf-8', errors='ignore') as f: # 파일 경로 주의
    reader = csv.reader(f)
    next(reader)
    linenumber = 1
    try:
        for row in reader:
            print(linenumber)
            print(row)
            linenumber += 1
            trending_date.append(row[1])
            title.append(row[2])
            channel_title.append(row[3])
            category_id.append(row[4])
            publish_time.append(row[5])
            tags.append(row[6])
            views.append(row[7])
            likes.append(row[8])
            dislikes.append(row[9])
            comment_count.append(row[10])
            description.append(row[15])
             
    except Exception as e:
        print('Error line {}: {}'.format(linenumber, e))
 
 
df = pd.DataFrame(list(zip(trending_date,title,channel_title,category_id,publish_time,tags,views,likes,dislikes,comment_count,description)), columns=['trending_date','title','channel_title','category_id','publish_time','tags','views','likes','dislikes','comment_count','description'])
df.to_csv('RUtrending.csv', index=False, sep=',', na_rep='NaN', encoding='utf-8')
====================='''

''' ===== 한국 이외의 나라 데이터에서 한글만 뽑아와서 워드클라우드 만들기 ====='''
''' # pip install wordcloud 과 pip install nltk 필요
from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image
import numpy as np
import re
from konlpy.tag import Okt
from collections import Counter
import pickle

def ExtractTitleAll(df): # 영상 제목만 뽑기
    titles = ''
    for i in range(len(df['title'])): 
        titles = titles + ' ' + df['title'][i]
    return titles

def ExtractKoreanWord(strdata): # 한글만 추출
    temp = re.findall(r'[가-힣]+', strdata)

    for i in temp: # 한글자 제외
        if len(i) < 2:
            temp.remove(i)
    temp2 = ' '.join(temp)
    
    nlp = Okt()
    nouns = nlp.nouns(temp2) # 명사로 나누고
    result = []
    for i in nouns: 
        if len(i) > 1: #명사중 길이가 2이상인 단어만 결과에 넣기
            result.append(i)
    return result

def save_img(data, types, nation):
    #단어빈도수 계산
    counter = Counter(data)
    tag = counter.most_common(600)
    dictTag = dict(tag)

    # pickle로 빈도수 딕셔너리 저장
    #with open('{}_{}data.pickle'.format(nation, types), 'wb') as f:
        #pickle.dump(dictTag, f, pickle.HIGHEST_PROTOCOL)
        
    mask = np.array(Image.open(finders.find("images/mask/mask-title-All-korean.jpg".format(nation)))) # 워드클라우드 모양과 색깔을 결정하는 마스크 이미지
    color = ImageColorGenerator(mask)

    wc = WordCloud(font_path='c:/Windows/Fonts/malgunbd.ttf', 
                   width=1000, 
                   height=1000, 
                   background_color='white', 
                   max_font_size=300, 
                   max_words=300, 
                   mode='RGBA',
                   mask=mask,
                   color_func=color,
                )
    wc.generate_from_frequencies(dictTag)
    wc.to_file('C:/Work/py_sou/pyweb/myapp/static/images/wc_images/이미지파일저장이름.png') # 파일이 저장될 절대 경로. 데이터 타입에 따라서 각자 따로 폴더, 파일명 지정하기
    
# # 파이썬 프로젝트에서 실행할 경우
# if __name__ == '__main__':
# #     dfca = pd.read_csv('./trending/CAvideos.csv')
# #     dfde = pd.read_csv('./trending/DEvideos.csv')
# #     dffr = pd.read_csv('./trending/FRvideos.csv')
# #     dfgb = pd.read_csv('./trending/GBvideos.csv')
# #     dfin = pd.read_csv('./trending/INvideos.csv')
# #     dfus = pd.read_csv('./trending/USvideos.csv')
# #     dfjp = pd.read_csv('JPtrending.csv')
# #     dfmx = pd.read_csv('MXtrending.csv')
# #     dfru = pd.read_csv('RUtrending.csv')
# #     print('== load data ==')
# #     
# #     t_ca = ExtractTitleAll(dfca)
# #     t_de = ExtractTitleAll(dfde)
# #     t_fr = ExtractTitleAll(dffr)
# #     t_gb = ExtractTitleAll(dfgb)
# #     t_in = ExtractTitleAll(dfin)
# #     t_us = ExtractTitleAll(dfus)
# #     t_jp = ExtractTitleAll(dfjp)
# #     t_mx = ExtractTitleAll(dfmx)
# #     t_ru = ExtractTitleAll(dfru)
# #     print('== 추출 end ==')
# # 
# #     titles = t_ca + t_de + t_fr + t_gb + t_in + t_us + t_jp + t_mx + t_ru
# #     words = ExtractKoreanWord(titles)
# #     print('== Title 한글 end ==')
#     
#     # 저장한 딕셔너리 데이터를 이용해서 워드클라우드 생성
#     # 시간 줄이기 위해서
#     with open('korean_from_others_titledata.pickle', 'rb') as file:
#         data = pickle.load(file)
#     
#     # 워드클라우드 생성 함수
#     save_img(data, 'title', 'korean_from_others')
#     print('== end == ')

def test(request):
#     dfca = pd.read_csv('./trending/CAvideos.csv')
#     dfde = pd.read_csv('./trending/DEvideos.csv')
#     dffr = pd.read_csv('./trending/FRvideos.csv')
#     dfgb = pd.read_csv('./trending/GBvideos.csv')
#     dfin = pd.read_csv('./trending/INvideos.csv')
#     dfus = pd.read_csv('./trending/USvideos.csv')
#     dfjp = pd.read_csv('JPtrending.csv')
#     dfmx = pd.read_csv('MXtrending.csv')
#     dfru = pd.read_csv('RUtrending.csv')
#     print('== load data ==')
#     
#     t_ca = ExtractTitleAll(dfca)
#     t_de = ExtractTitleAll(dfde)
#     t_fr = ExtractTitleAll(dffr)
#     t_gb = ExtractTitleAll(dfgb)
#     t_in = ExtractTitleAll(dfin)
#     t_us = ExtractTitleAll(dfus)
#     t_jp = ExtractTitleAll(dfjp)
#     t_mx = ExtractTitleAll(dfmx)
#     t_ru = ExtractTitleAll(dfru)
#     print('== 추출 end ==')
# 
#     titles = t_ca + t_de + t_fr + t_gb + t_in + t_us + t_jp + t_mx + t_ru
#     words = ExtractKoreanWord(titles)
#     print('== Title 한글 end ==')

    # 저장한 딕셔너리 데이터를 이용해서 워드클라우드 생성
    # 시간 줄이기 위해서
    with open(finders.find('datafiles/pickles/korean_from_others_titledata.pickle'), 'rb') as file:
        data = pickle.load(file)
    
    # 워드클라우드 생성 함수
    save_img(data, 'title', 'korean_from_others')
    print('== end == ')
    
    return render(request, 'test.html')
====================='''

''' ===== 워드 클라우드 작성 ===== '''
'''
import pandas as pd
from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image
import numpy as np
import re
from collections import Counter
from nltk.corpus import stopwords
import pickle

# 데이터 양이 많을 경우 그대로 프린트를 하면 컴퓨터가 멈출 수 있으므로 주의
def ExtractTagAll(df): # tag 만 뽑기
    tags = ''
    for i in range(len(df['tags'])): 
        tags = tags + ' ' + df['tags'][i]
    return tags

def ExtractTitleAll(df): # 영상 제목만 뽑기
    titles = ''
    for i in range(len(df['title'])): 
        titles = titles + ' ' + df['title'][i]
    return titles


# nltk가 지원하는 언어
# ['arabic', 'danish', 'dutch', 'english', 'finnish', 'french', 'german', 'hungarian', 'italian', 'kazakh', 'norwegian', 'portuguese', 'romanian', 'russian', 'spanish', 'swedish', 'turkish']
def CleanWord(strdata): # 영상 제목의 특수문자와 숫자를 제외하고 띄워쓰기로 데이터를 나누기
    strdata = strdata.lower()
    strdata = re.sub(r'(\W+|\d+)', ' ', strdata)
    stop = stopwords.words('english')  # 만들 언어에 따라서 값 바꿔주기
    stop.append('video') # 기본 stopword 말고 더 추가 하고 싶은 단어를 넣음
    stop.append('official')
    stop.append('trailer')
    stop.append('episode')
    stop.append('vs')
    stop.append('full')
    stop.append('feat')
    stop.append('ft')
    stop.append('the')
    stop = set(stop)
    #print(stop) 
    
    temp = strdata.split()
    #print(temp)
    result = [i for i in temp if i not in stop]
    result2 = [i for i in result if len(i) > 2]
    
    return result2

def CleanTag(strdata): 
    strdata = strdata.lower()
    strdata = re.sub(r'("+)', ' ', strdata)
    print(strdata[:10])
    stop = stopwords.words('english')
    stop.append('video')
    stop.append('official')
    stop.append('trailer')
    stop.append('episode')
    stop.append('vs')
    stop.append('full')
    stop.append('feat')
    stop.append('ft')
    stop.append('the')
    stop = set(stop)
    #print(stop)
    
    # 태그는 "" 만 지우고 | 를 기준으로 단어 나누기
    temp = strdata.split('|')
    for i in range(len(temp)):
        temp[i] = temp[i].strip() 
    print(temp[:10])
    result = [i for i in temp if i not in stop]
    
    return result

def save_img(data, types, nation): # 이미지 저장 경로와 폰트를 제외하고 위의 이미지 저장함수와 동일
    counter = Counter(data) # 빈도수로 단어 뽑기
    tag = counter.most_common(600)
    dictTag = dict(tag)

    # 피클로 저장
    #with open('{}_{}data.pickle'.format(nation, types), 'wb') as f:
    #    pickle.dump(dictTag, f, pickle.HIGHEST_PROTOCOL)

    mask = np.array(Image.open(finders.find("images/mask/{}01.jpg".format(nation)))) 
    color = ImageColorGenerator(mask)
    
    wc = WordCloud(font_path='c:/Windows/Fonts/Arial.ttf',  # 한글이 아닌 경우 폰트
                   width=1000, 
                   height=1000, 
                   background_color='white', 
                   max_font_size=300, 
                   max_words=300, 
                   mask=mask,
                   mode='RGBA',
                   color_func=color
                )
    wc.generate_from_frequencies(dictTag)
    wc.to_file('C:/Work/py_sou/pyweb/myapp/static/images/wc_images/이미지파일저장이름.png') 

# 파이썬 프로젝트에서 실행할 경우
# if __name__ == '__main__':
#     #df = pd.read_csv('./trending/DEvideos.csv')
#     df = pd.read_csv('RUtrending.csv')
#     print('== load data ==')
#     
#     titles = ExtractTitleAll(df)
#     print('..')
#     tags = ExtractTagAll(df)
#     print('== 추출 end ==')
# 
#     words = CleanWord(titles)
#     print('..')
#     tags = CleanTag(tags)
#     print('== clean word end ==')
#     #print(words[:100])
#     
#     save_img(words, 'title', 'russia') # 불러온 데이터의 종류에 따라서 title과 russia 값을 바꿔주기
#     print('..')
#     save_img(tags, 'tags', 'russia')
#     print('== end == ')

def test(request):
    df = pd.read_csv(finders.find('datafiles/USvideos.csv'))
    #df = pd.read_csv('./trending/DEvideos.csv')
    #df = pd.read_csv('RUtrending.csv')
    print('== load data ==')
    
    titles = ExtractTitleAll(df[:100]) # 전체 데이터를 사용하면 시간이 오래걸리기 때문에 일부만 적은것
    print('..')
    tags = ExtractTagAll(df[:100])
    print('== 추출 end ==')

    words = CleanWord(titles)
    print('..')
    tags = CleanTag(tags)
    print('== clean word end ==')
    #print(words[:100])
    
    save_img(words, 'title', 'usa') # 불러온 데이터의 종류에 따라서 title과 russia 값을 바꿔주기
    print('..')
    save_img(tags, 'tags', 'russia')
    print('== end == ')
    
    return render(request, 'test.html')
====================='''
'''========================끝============================'''

def TitleGenPageFunc(request):
    return render(request, 'titleGen.html')

def TitleGenFunc(request):
    new_keyword = request.GET['keyword']
    print(new_keyword)

    text = ExtractTitle()
    sequences, tok, vocab = TokenizeTitle(text)
    x, y, m = makeDatas(sequences)
    #makeModel(x, y, m, vocab) # 모델생성, 데이터 양이 많아질수록 시간이 오래 걸림
    bestmodel = tf.keras.models.load_model('C:/Work/py_sou/pyweb/myapp/static/datafiles/models/best_model.hdf5')
    #model = tf.keras.models.load_model('C:/Work/py_sou/pyweb/myapp/static/datafiles/models/gen_title_model.hdf5')
    eval = bestmodel.evaluate(x, y)
    results = sentence_generation(bestmodel, tok, new_keyword, 10, m)
    #results = sentence_generation(model, tok, new_keyword, 10, m)
    print('{} : '.format(new_keyword), results)
    
    context = {'resultTitle':results, 'loss':eval[0], 'acc':eval[1]}
    # context = {'resultTitle':results}
    
    print(context, type(context)) #dict
    print(json.dumps(context), type(json.dumps(context)))
    
    return HttpResponse(json.dumps(context), content_type='application/json')



import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

max_len = 0
vocab_size = 0
def ExtractTitle(): # 영상 제목 가져오기
    df = pd.read_csv('C:/Work/py_sou/pyweb/myapp/static/datafiles/KRtrending.csv')
    titles = ''
    cnt = 0

    for i in range(len(df['title'])): 
        if df['views'][i] > 100000: # 조회수 10만 이상인 제목만 읽기
            cnt += 1
            titles = df['title'][i] + '\n' + titles
            if cnt > 5000: # train한 데이터 그대로 예측에 사용
                break
    print(len(titles), type(titles))

    return titles

def TokenizeTitle(text): # 가져온 영상 제목을 토큰화 하기
    tok = Tokenizer()
    tok.fit_on_texts([text])
    
    global vocab_size
    vocab_size = len(tok.word_index) + 1
    print('단어집합의 크기 : %d'%vocab_size) # 11

    # train data
    sequences = list()
    for line in text.split(sep='\n'): # 라인단위로 먼저 자르고(문장 토큰화)
        encoded = tok.texts_to_sequences([line])[0]
        for i in range(1, len(encoded)):
            sequ = encoded[: i+1]
            sequences.append(sequ) # 토큰이 어떤 토큰 다음으로 나올지 학습시키기 위해
    print('샘플 수 : %d'%len(sequences)) # 10
    
    return sequences, tok, vocab_size

def makeDatas(sequences): # 토큰화한 데이터에서 feature, label 구하기
    global max_len
    max_len = max(len(i) for i in sequences)
    sequences = pad_sequences(sequences, maxlen=max_len, padding='pre')
    
    # 각 샘플의 마지막 요소값을 레이블로 사용하기 위해 분리
    sequences = np.array(sequences)
    x = sequences[:, :-1]  # feature
    y = sequences[:, -1]   # label or class

    # label : onehot encoding
    y = to_categorical(y, num_classes=vocab_size)
    
    return x, y, max_len

def makeModel(x, y, max_len, vocab_size): # 모델 생성. 시간이 오래걸림
    model = Sequential()
    model.add(Embedding(vocab_size, 32, input_length=(max_len-1)))
    model.add(LSTM(32))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=vocab_size, activation='softmax'))
    model.summary()

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    earlystop = EarlyStopping(monitor='loss', patience=10, mode='min')
    mcheck = ModelCheckpoint(filepath='C:/Work/py_sou/pyweb/myapp/static/datafiles/models/best_model.hdf5', monitor='loss', save_best_only=True)
    model.fit(x, y, epochs=1000, verbose=2, batch_size=64, callbacks=[earlystop, mcheck])
    print(model.evaluate(x, y))
    
    # 모델 저장
    model.save('C:/Work/py_sou/pyweb/myapp/static/datafiles/models/gen_title_model.hdf5')
    del model

def sentence_generation(model, t, current_word, n, max_len): # 모델이 정확하게 예측하는지 함수
    init_word = current_word
    sentence = ''
    for _ in range(n):
        encoded = t.texts_to_sequences([current_word])[0]  # 현재 단어에 대한 정수 인코딩  
        encoded = pad_sequences([encoded], maxlen=(max_len - 1), padding='pre')
        result = np.argmax( model.predict(encoded) )
        for word, index in t.word_index.items():
            if index == result: # 만약 예측한 단어의 인덱스와 동일한 단어가 있으면
                break
        current_word = current_word + ' ' + word # 현재단어 + 예측단어
        sentence = sentence + ' ' + word
        
    sentence = init_word + sentence
    return sentence

# 파이썬 프로젝트로 실행시
# if __name__ == '__main__':
#     text = ExtractTitle()
#     sequences, tok, vocab = TokenizeTitle(text)
#     x, y, m = makeDatas(sequences)
#     #makeModel(x, y, m, vocab) # 모델생성
#     model = tf.keras.models.load_model('./models/gen_title_model.hdf5')
#     bestmodel = tf.keras.models.load_model('./models/best_model.hdf5')
#     print('방탄소년단 : ', sentence_generation(model, tok, '방탄소년단', 10, m))

'''========================끝============================'''
#-------유진

def DataViewFunc(request):
    data = pd.read_csv("C:\work\py_sou\pyweb\myapp\static\KRtrending.csv")
    #print(data)
    
    
    # 시청 수 - 좋아요 수 
    likes_num = data.likes
    views_num = data.views
    plt.scatter(views_num, likes_num)
    plt.gcf()
    plt.savefig("C:/work/py_sou/pyweb/myapp/static/views-likes.png")
    plt.clf()

    # 시청 수 - 싫어요 수
    dislikes_num = data.dislikes
    views_num = data.views
    plt.scatter(views_num, dislikes_num)
    plt.gcf()
    plt.savefig("C:/work/py_sou/pyweb/myapp/static/views-dislikes.png")
    plt.clf()
    
    # 시청 수 - 댓글 수 
    comments_num = data.comment_count
    views_num = data.views
    plt.scatter(views_num, comments_num)
    plt.gcf()
    plt.savefig("C:/work/py_sou/pyweb/myapp/static/views-comments.png")
    plt.clf()
    
    # 카테고리별 Trend youtube video 수
    cate_id_num = data.groupby('category_id').size()
    labels = ['1', '2', '10', '15', '17', '19', '20', '22', '23', '24', '25', '26', '27', '28', '29', '43', '44']
    indexs = np.arange(len(labels))
    plt.bar(indexs, cate_id_num)
    plt.xticks(indexs, labels, fontsize=7)
    plt.gcf()
    plt.savefig("C:/work/py_sou/pyweb/myapp/static/sizeofgroupbyid.png")
    plt.clf()
    
    # 시청률 집단 분류 후 집단 별 수
    view_data = pd.DataFrame(data[['views']])
    
    view_data['view_section'] = 'under10000'
    view_data['view_section'][(view_data['views'] >= 10000) & (view_data['views'] < 100000)] = '10000 ~ 100000'
    view_data['view_section'][(view_data['views'] >= 100000) & (view_data['views'] < 500000)] = '100000 ~ 500000'
    view_data['view_section'][(view_data['views'] >= 500000) & (view_data['views'] < 1000000)] = '500000 ~ 1000000'
    view_data['view_section'][view_data['views'] >= 1000000] = 'over1000000'
    
    view_section_size = view_data.groupby('view_section').size()
    
    label = ['under10000', '10000 ~ 100000', '100000 ~ 500000', '500000 ~ 1000000', 'over1000000']
    index = np.arange(len(label))
    plt.bar(index, view_section_size)
    plt.xticks(index, label, fontsize=7)
    plt.title('Graph of ViewGroup')
    plt.gcf()
    plt.savefig("C:/work/py_sou/pyweb/myapp/static/viewgroup_count.png")
    plt.clf()
    
    # 전체 상관관계    
    dataset = data[['views', 'likes', 'dislikes', 'comment_count']]
    data_corr = dataset.corr()
    sns.heatmap(data_corr, cmap='viridis')
    plt.title('Heatmap of Youtube')
    plt.gcf()
    plt.savefig("C:/work/py_sou/pyweb/myapp/static/heatmapofyoutube.png")
    plt.clf()
    
    
    return render(request, 'youtube_data.html')
#----------지훈


dir = 'C:/work/py_sou/pyweb/myapp/static/files/'
def AnalysisFunc(request):
    '''
    # 데이터 읽기 -> 저장 해서 주석처리
    df_kr = Data_read()
    
    # 가공한 파일 저장하기  -> 주석 처리
    df_kr.to_csv(dir+ 'df_kr.csv', encoding='utf-8')
    '''
    # 가공한 파일 읽어서 가져오기 
    df_kr = pd.read_csv(dir+'df_kr.csv', encoding='utf-8')
    #print(df_kr.head(2))
             
    # 각 분야 별 조회수와 업로드수
    Bar_chart(df_kr)
    
    # legacy 가설 검정 -  Two-Way ANOVA
    # 각 상호 작용 그래프
    Data_Anova(df_kr)
    
    # 도넛 차트 보기
    Donut_chart(df_kr)
    
    return render(request, 'upload_date&views.html')

 
# 카테고리 명칭 넣기  
def Category_id_json(nation_json):
    import json
    id_to_category = {}
    
    with open(dir + nation_json, 'r') as f:
        data = json.load(f)
        for category in data['items']:
            id_to_category[category['id']] = category['snippet']['title']
    #print(id_to_category)
    
    return id_to_category

# 데이터 읽어 오기    
def Data_read():
    # 한국 데이터 
    df_kr = pd.read_csv(dir + 'KRtrending.csv', engine='python')
    df_kr['nation'] = 'KR'
    df_kr['category_id'] = df_kr['category_id'].astype(str)
    df_kr = df_kr[['category_id', 'nation', 'views', 'publish_time']]
    
    
    # 카테고리 명칭 넣기 
    files = 'KR_category_id.json'
    id_to_category = Category_id_json(files) 
    df_kr['category_name'] = df_kr['category_id'].map(id_to_category)


    # 날짜를 요일과 시간으로 바꾸기
    df_kr["publishing_day"] = df_kr["publish_time"].apply(
    lambda x: datetime.datetime.strptime(x[:10], "%Y-%m-%d").date().strftime('%a'))
    df_kr["publishing_hour"] = df_kr["publish_time"].apply(lambda x: x[11:13])
    df_kr.drop(labels='publish_time', axis=1, inplace=True)
    #print(set(df_kr["publishing_day"]))

    # 각각 숫자로 치환하기
    df_kr['category_id'] = df_kr['category_id'].astype('int64')
    df_kr['publishing_hour'] = df_kr['publishing_hour'].astype('int64')
    df_kr['publishing_day_num'] = df_kr['publishing_day'].map({
        'Mon':0, 'Tue':1, 'Wed':2, 'Thu':3, 'Fri':4, 'Sat':6, 'Sun':7        
    })
    
    return df_kr

def Bar_chart(df_kr):
    # 등록 요일에 따라 정렬
    df_kr = df_kr.sort_values(by = ['publishing_day_num'], ascending=True, axis=0)
    
    # 카테고리별, 요일별로, 시간대 별로 조회 수 보기 
    plt.figure(figsize=(20, 6))
    
    plt.subplot(5, 1, 1)
    sns.barplot(df_kr['category_name'], df_kr['views'])
    plt.xlabel('카테고리별')
    plt.ylabel('조회수')
    plt.xticks(rotation=20)
    
    plt.subplot(5, 1, 3)
    sns.barplot(df_kr['publishing_day'], df_kr['views'])
    plt.xlabel('요일')
    plt.ylabel('조회수')
    plt.xticks(rotation=20)
    
    plt.subplot(5, 1, 5)
    sns.barplot(df_kr['publishing_hour'], df_kr['views'])
    plt.xlabel('시간')
    plt.ylabel('조회수')
    plt.xticks(rotation=20)
    
    
    plt.gcf()  
    plt.savefig(os.path.dirname(os.path.realpath(__file__)) + '/static/images_plot/barplot.png')
    plt.cla()
    print('barplot.png 저장 성공')
    
    # 동영상 등록 개수
    plt.figure(figsize=(20, 6))
    
    plt.subplot(5, 1, 1)
    sns.countplot(x='category_name', data=df_kr)
    sns.barplot(x=df_kr.category_name.value_counts().index, y=df_kr.category_name.value_counts())
    plt.xlabel('카테고리별')
    plt.ylabel('동영상 개수')
    plt.xticks(rotation=20) 
    
    plt.subplot(5, 1, 3)
    sns.countplot(x='publishing_day', data=df_kr)
    sns.barplot(x=df_kr.publishing_day.value_counts().index, y=df_kr.publishing_day.value_counts())
    plt.xlabel('요일')
    plt.ylabel('동영상 개수')
    plt.xticks(rotation=20)
    
    plt.subplot(5, 1, 5)
    sns.countplot(x='publishing_hour', data=df_kr)
    sns.barplot(x=df_kr.publishing_hour.value_counts().index, y=df_kr.publishing_hour.value_counts())
    plt.xlabel('시간') 
    plt.ylabel('동영상 개수')
    plt.xticks(rotation=20)
    
    plt.gcf()  
    plt.savefig(os.path.dirname(os.path.realpath(__file__)) + '/static/images_plot/barplot2.png')
    plt.cla()
    print('barplot.png2 저장 성공')


    
def Data_Anova(df_kr):
    # 귀무 : 카테고리, 동영상 등록 요일, 시간에 따라 조회수에 차이가 없다. 
    # 대립 : 카테고리, 동영상 등록 요일, 시간에 따라 조회수에 차이가 있다.
     
    '''
    # 정규성 -  표본수가 30이 넘으면 정규분포를 따른다고 가정한다
    # 카테고리 / 조회수 
    formula = 'views  ~ C(category_id)+ C(publishing_day) + C(publishing_hour)\
                + C(category_id):C(publishing_day) + C(category_id):C(publishing_hour) + C(publishing_hour):C(publishing_hour)\
                + C(category_id):C(publishing_day):C(publishing_hour)'
    Reg1 = ols(formula =formula, data = df_kr)
    Fit1 = Reg1.fit()
    aov_table  = anova_lm(Fit1) 
    
    #print(Fit1.summary())
    #print(aov_table)
    
    # 결론: PR(>F) < 0.05 이므로 귀무 기각, 대립 채택~ 카테고리, 동영상 등록 요일, 시간에 따라 조회수에 차이가 있다. !!!
    # 오래 걸려서 주석처리 하였음
    '''
    
        
    # 각각 영역에 대한 상호작용 그래프 그리기
    plt.rcParams['figure.figsize'] = [16, 8]
    fig = interaction_plot(df_kr['publishing_hour'],  df_kr['publishing_day'], df_kr['views'],
             ylabel='views', xlabel='publishing_hour', legendtitle='publishing_day')
    plt.xticks(rotation=30)
    plt.title('interaction effect plot')
    
    plt.gcf()  
    plt.savefig(os.path.dirname(os.path.realpath(__file__)) + '/static/images_plot/anova_interaction_plot1.png')
    plt.cla()
    
    fig = interaction_plot(df_kr['category_name'],  df_kr['publishing_day'], df_kr['views'],
             ylabel='views', xlabel='category_name')
    plt.xticks(rotation=30)
    plt.title('interaction effect plot') 
    
    plt.gcf()  
    plt.savefig(os.path.dirname(os.path.realpath(__file__)) + '/static/images_plot/anova_interaction_plot2.png')
    plt.cla() 
    
    fig = interaction_plot(df_kr['category_name'],  df_kr['publishing_hour'], df_kr['views'],
             ylabel='views', xlabel='category_name')
    plt.xticks(rotation=30)
    plt.title('interaction effect plot')
    
    plt.gcf()   
    plt.savefig(os.path.dirname(os.path.realpath(__file__)) + '/static/images_plot/anova_interaction_plot3.png')
    plt.cla()
    
    print('anova_interaction_plot.png 저장 성공')  
        
    

def Donut_chart(df_kr):
    # 도넛 파이 구하기  
    
    # 작은 수의 카테고리 삭제  
    drop_category = ['Travel & Events', 'Vehicles', 'Trailers', 'Nonprofits & Activism',
                     'Autos & Vehicles','Howto & Style','Pets & Animals','Sports','Shows',
                     'Science & Technology']
    
    for i in drop_category:
        df_kr_drop = df_kr[df_kr['category_name'] == i].index
        df_kr = df_kr.drop(df_kr_drop)
    
    #print('-',set(df_kr['category_name']))
    
    # 순차적으로 나오기 위해서 정렬하기
    df_kr = df_kr.sort_values(by = ['category_name', 'publishing_day_num', 'publishing_hour'], ascending=True, axis=0)
    df_kr.to_csv(dir+ 'df_kr2.csv', encoding='utf-8') 
    df_kr = pd.read_csv(dir+'df_kr2.csv', encoding='utf-8')
    
    
    # 카테고리 이름 , 개수 - 가장 바깥 고리
    name_dict = {}
    for categorys in df_kr['category_name']:
        if categorys in name_dict:
            name_dict[categorys] = name_dict[categorys] + 1
        else :
            name_dict[categorys] = 1    
          
    #print('++++',name_dict)
    
    name_keys = []
    name_values = []
    for k, v in name_dict.items():
        name_keys.append(k)
        name_values.append(v)
        
    #print(name_keys)    
    #print(name_values)    
    
    
    # 요일, 개수 - 중간 고리
    day_dict = {}
    for i in range(len(df_kr)):
        temp = df_kr['category_name'][i] + df_kr['publishing_day'][i]
        #print('===',tt)
        if temp in day_dict:
            day_dict[temp] = day_dict[temp] + 1
        else :
            day_dict[temp] = 1    
               
    #print(day_dict) 
    #print(day_dict.keys()) 
    

    
    category_day_key = []
    category_day_value = []
    category_days = []   # 요일만 잘라서 담기 
    for key in day_dict: 
        for name in name_keys:       
            if key.startswith(name) and key.endswith(('Mon','Tue', 'Wed', 'Fri', 'Thu')):
                #print('----', key, '    ', day_dict[key])                    
                category_day_key.append(key) 
                category_day_value.append(day_dict[key])
                category_days.append(key[-3:])
                
            elif key.startswith(name) and key.endswith(('Sat', 'Sun')):
                #print('*****', key, '    ', day_dict[key])
                category_day_key.append(key) 
                category_day_value.append(day_dict[key])
                category_days.append(key[-3:])

    # 시간에 따라 - 안쪽 고리
    hour_dict = {}
    for i in range(len(df_kr)):
        temp = df_kr['category_name'][i] + df_kr['publishing_day'][i] + '0' + str(df_kr['publishing_hour'][i]) # 한자리수 숫자2자로 만들기
        #print('^^^^', temp)
        if temp in hour_dict:
            hour_dict[temp] = hour_dict[temp] + 1
        else :
            hour_dict[temp] = 1 
               
    #print(hour_dict) 
    #print(hour_dict.keys())
     
    hour_keys = []
    hour_values = []
    k_keys = []
    
    for k, v in hour_dict.items():
        hour_keys.append(k)
        hour_values.append(v)
        
    #print('+', hour_keys)    
    #print(hour_values) 
    
    for k in hour_keys:
        #print('########',k[-2:])
        k_keys.append(k[-2:])  # 일자 2자리수로 담기
    
    #print('----------------',k_keys)

    # 시간대별로 색깔지정하기
    colors_list = []
    
    daybreak_time = ['00', '01', '02','03', '04', '05'] # 새벽
    morning_time = ['06', '07', '08','09', '10', '11']  # 오전
    afternoon_time = ['12', '13', '14','15','16', '17'] # 오후
    night_time = ['18', '19', '20','21','22','23']      # 밤
    
    for i in k_keys:
        for day in daybreak_time:        
            if day == i:
                colors_list.append('blue')   # 새벽
                
        for day in morning_time:        
            if day == i:
                colors_list.append('orange') # 오전
                 
        for day in afternoon_time:        
            if day == i:
                colors_list.append('green')  # 오후   
                
        for day in night_time:        
            if day == i:
                colors_list.append('red')    # 밤         
                     
    #print(colors_list)
    
    
    plt.rcParams['figure.figsize'] = [15, 6]
 
    radius = 1.6
    fig, ax = plt.subplots()
    ax.axis('equal')
    pie_outside, _ = ax.pie(name_values, radius = radius, labeldistance=1.0, labels=name_keys)
    plt.setp(pie_outside, width= 0.2, edgecolor='white') 
    
    pie_middle, _ = ax.pie(category_day_value, radius = (radius - 0.2), labeldistance=1.0, labels=category_days, colors=['lightskyblue','lightskyblue','lightskyblue','lightskyblue','lightskyblue', 'lightcoral','lightcoral'])
    plt.setp(pie_middle, width= 0.2, edgecolor='white')
    
    pie_inside, _ = ax.pie(hour_values, radius = (radius - 0.2 - 0.2), colors=colors_list)
    plt.setp(pie_inside, width= 0.2)
    
    #plt.title('Donut_Plot')
    
    plt.gcf()   
    plt.savefig(os.path.dirname(os.path.realpath(__file__)) + '/static/images_plot/donut_plot.png')
    plt.cla()
    
    print('donut.png 저장성공')
#-------준혜
    
    

# ------------재홍

'''
jaehong 
'''
def jaehong(request):
    
    ''' 자료 utf-8로 다시 인코딩
    import csv
    video_id = []
    trending_date = []
    title = []
    hannel_title = []
    category_id = []
    publish_time = []
    tags = []
    views = []
    likes = []
    dislikes = []
    comment_count = []
    thumbnail_link = []
    comments_disabled = []
    ratings_disabled = []
    video_error_or_removed = []
    description = []
    with open('C:\work\py_sou\pyweb\myapp\static\youtubeDS/KRvideos.csv', 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        next(reader)
        linenumber = 1
        try:
            for row in reader:
                print(linenumber)
                print(row)
                linenumber += 1
                video_id.append(row[0])
                trending_date.append(row[1])
                title.append(row[2])
                channel_title.append(row[3])
                category_id.append(row[4])
                publish_time.append(row[5])
                tags.append(row[6])
                views.append(row[7])
                likes.append(row[8])
                dislikes.append(row[9])
                comment_count.append(row[10])
                thumbnail_link.append(row[11])
                comments_disabled.append(row[12])
                ratings_disabled.append(row[13])
                video_error_or_removed.append(row[14])
                description.append(row[15])
             
        except Exception as e:
            print('Error line {}: {}'.format(linenumber, e))
 
 
    df = pd.DataFrame(list(zip(video_id,trending_date,title,channel_title,category_id,publish_time,tags,views,likes,dislikes,comment_count,thumbnail_link,comments_disabled,ratings_disabled,video_error_or_removed,description)), columns=['video_id','trending_date','title','channel_title','category_id','publish_time','tags','views','likes','dislikes','comment_count','thumbnail_link','comments_disabled','ratings_disabled','video_error_or_removed','description'])
    df.to_csv('C:\work\py_sou\pyweb\myapp\static\youtubeDS/KRtrending.csv', index=False, sep=',', na_rep='NaN', encoding='utf-8')

    '''
    data=pd.read_csv("C:/work/py_sou/pyweb/myapp/static/KRtrending.csv")
    cat_data=pd.read_json("C:/work/py_sou/pyweb/myapp/static/files/KR_category_id.json")
    cat_items=cat_data['items'] #json에서 item 태그 정보 가져오기
    cat_items.count()
    for idx in range(0, cat_items.count()):
        cat_data.loc[idx,'id'] = cat_items[idx]['id']                           
        cat_data.loc[idx,'category'] = cat_items[idx]['snippet']['title']       #카테고리이름을 data와 매핑시키기
    cat_data=cat_data.drop(columns=['kind','etag','items'])
    # cat_data.info()
 
    cat_data['id']=cat_data['id'].astype('int64')
    data=pd.merge(data, cat_data, left_on='category_id', right_on='id', how='left') #카테고리 데이터와 원본데이터 합병
    # print('datahead',data[['video_id','category_id']].head(10))
 
    data['category_id'].loc[data['id'].isnull()==True].value_counts()
    data['id'].fillna(29, inplace=True)
    data['category'].fillna('Nonprofits & Activism', inplace=True)
    # data.info()
    # print('datahead',data.head(10))
 
    idx=(data['video_error_or_removed']==False) & (data['ratings_disabled']==False) # & (data['comments_disabled']==False)
    data=data.loc[idx,:]
    data[['comments_disabled','ratings_disabled','video_error_or_removed']].describe()
    data=data.drop(columns=['comments_disabled','ratings_disabled','video_error_or_removed'])
 
    data['video_id'].describe()
    idx=(data['video_id']!='#NAME?')
    data=data.loc[idx,:]
    data['video_id'].describe()
 
    #category_id: str 타입변환
    data['category_id'] = data['category_id'].astype(str)
    # category_id, 
    # tags, title, chanel_title => text analysis 
    # publish_time, trending_date  => 날짜 차이필요
    #
    # likes, dislike, comment_count : X,Y의 선후 인과 관계 문제
    # thumbnail_link : 불필요 컬럼
 
    # data.info()
  
    plt.rc('font', family = 'malgun gothic') #한글 안깨지게
  
    '''
    선형회귀 분석 : 조회수 대비 좋아요 , 싫어요 , 댓글 
    '''
    data.reset_index(inplace=True)
    cols = ['likes','dislikes','comment_count']
    X = data[cols].copy()
    y = data['views']
    from sklearn.linear_model import LinearRegression
    from sklearn import metrics
    from sklearn.model_selection import train_test_split
   
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2) #80% , 20%
    ''' 시각화 , 이미지 저장하는 부분 
    svm = sns.pairplot(data, x_vars=['likes','dislikes','comment_count'], y_vars='views', height=5, aspect=0.7, kind='reg')
    svm.savefig('C:/work/py_sou/pyweb/myapp/static/images/linear.png', dpi=400)
     
    line_fitter = LinearRegression()
    line_fitter.fit(X_train, y_train)
    y_predicted = line_fitter.predict(X_test)
#     print('R^2=', line_fitter.score(X_train,y_train))   #0.8058971129088008
#     print('coef=',line_fitter.coef_, ', intercept=', line_fitter.intercept_)    #coef= [ 19.13275772  54.98759001 -24.94934284] , intercept= 213392.6998093983
#  
#     print('\nlinear1')
#     print('MAE =',round(metrics.mean_absolute_error(y_test, y_predicted),3))            #MAE = 320208.803
#     print('MSE = ',round(metrics.mean_squared_error(y_test, y_predicted),3))            #MSE =  1673071611668.016
#     print('RMSE = ',round(np.sqrt(metrics.mean_squared_error(y_test, y_predicted)),3))  #RMSE =  1293472.695
    
    '''
    '''
    xgboost
    '''
    import xgboost
    model = xgboost.XGBRegressor(learning_rate=0.1,max_depth=5, n_estimators=100,random_state=0) 
    model.fit(X_train,y_train)
    y_predicted = model.predict(X_test)
    ''' 시각화 , 이미지 저장하는 부분 
    svm = sns.scatterplot(y_test, y_predicted)
    figure4 = svm.get_figure()    
    figure4.savefig('C:/work/py_sou/pyweb/myapp/static/images/xgboost.png', dpi=400)
#     print('\nxgboost')
#     print('MAE =',round(metrics.mean_absolute_error(y_test, y_predicted),3))            #MAE = 171324.749
#     print('MSE = ',round(metrics.mean_squared_error(y_test, y_predicted),3))            #MSE =  590656242610.098
#     print('RMSE = ',round(np.sqrt(metrics.mean_squared_error(y_test, y_predicted)),3))  #RMSE =  768541.634
    '''
    datdesc = data.describe()
    
    '''# 시각화 , 이미지 저장하는 부분 
    figure5 = plt.figure(figsize = (12,6))
   
    plt.subplot(221)
    g1 = sns.distplot(data['views'], hist=True, rug=True)
    g1.set_title("조회수 분포", fontsize=16)
   
    plt.subplot(222)
    g4 = sns.distplot(data['comment_count'], color='yellow', hist=True, rug=True)
    g4.set_title("댓글 분포", fontsize=16)
   
    plt.subplot(223)
    g2 = sns.distplot(data['likes'],color='green', hist=True, rug=True)
    g2.set_title('좋아요 분포', fontsize=16)
   
    plt.subplot(224)
    g3 = sns.distplot(data['dislikes'], color='red', hist=True, rug=True)
    g3.set_title("싫어요 분포", fontsize=16)
   
    plt.subplots_adjust(wspace = 0.2, hspace = 0.4,top = 0.9)
   
    figure5.savefig('C:/work/py_sou/pyweb/myapp/static/images/dist.png', dpi=400)
    '''
    
    '''
    상관분석 : 조회수 좋아요 싫어요 댓글    corr matrix
    '''
    data_dr=data[['views','likes','dislikes','comment_count']]
    # print(data_dr.head(10))
    sns.set_style('whitegrid')
    sns.set()
    data_dr_corr=data_dr.corr()
    plt.rc('font', family = 'malgun gothic') #한글 안깨지게
    '''
    plt.figure(figsize = (15,15))
    plt.title('유투브 데이터 상관분석',fontsize=30)
    svm = sns.heatmap(data_dr_corr,
            cmap='coolwarm', cbar=True, annot=True, square=True, fmt='.2f')
    figure7  = svm.get_figure()    
    figure7.savefig('C:/work/py_sou/pyweb/myapp/static/images/heatmap.png', dpi=400)
    '''
    
    ''' LR , DTree , RF , XGB 네가지 방식으로 조회수 학습및 예측 '''
    lr = LinearRegression()
    dtree = DecisionTreeRegressor(random_state=0)
    forest = RandomForestRegressor(n_estimators=100,max_depth=5, random_state=0)
    boost = XGBRegressor(learning_rate=0.1,max_depth=5, n_estimators=100,random_state=0)
    lr.fit(X_train, y_train) 
    dtree.fit(X_train, y_train) 
    forest.fit(X_train, y_train) 
    boost.fit(X_train, y_train) 
    lr_y_pred = lr.predict(X_test)
    dtree_y_pred = dtree.predict(X_test)
    forest_y_pred = forest.predict(X_test)
    boost_y_pred = boost.predict(X_test)
 
    '''# 시각화 , 이미지 저장하는 부분 
    fig8, axs = plt.subplots(2,2)
 
    plt.rcParams["figure.figsize"] = (20,15)
    plt.rcParams['lines.linewidth'] = 1
    plt.rcParams['lines.color'] = 'r'
    plt.rcParams['axes.grid'] = True 
 
 
    guided_xy = range(1,y_test.max(),1000000)
                  
    axs[0, 0].set_title("1. LinearRegression", size=20)
    axs[0, 1].set_title("2. DTREE", size=20)
    axs[1, 0].set_title("3. RandomForest", size=20)
    axs[1, 1].set_title("4. XGBoost", size=20)
 
    axs[0,0].set_xlabel('예측 조회수')
    axs[0,0].set_ylabel('실제 조회수')
    axs[1,0].set_xlabel('예측 조회수')
    axs[1,0].set_ylabel('실제 조회수')
 
 
    axs[0,0].scatter(lr_y_pred, y_test)
    axs[0,1].scatter(dtree_y_pred, y_test)
    axs[1,0].scatter(forest_y_pred, y_test)
    axs[1,1].scatter(boost_y_pred, y_test)
 
    axs[0,0].scatter(guided_xy, guided_xy)
    axs[0,1].scatter(guided_xy, guided_xy)
    axs[1,0].scatter(guided_xy, guided_xy)
    axs[1,1].scatter(guided_xy, guided_xy)
 
    plt.legend()
 
    fig8.tight_layout()
    fig8.savefig('C:/work/py_sou/pyweb/myapp/static/images/predict.png', dpi=400)
    '''
    return render(request, 'jaehong.html', {'active_page' : 'list.html' , 'df' : datdesc.to_html()})
    



