from django.shortcuts import render
from django.http import HttpResponse
from sentence_transformers import SentenceTransformer
import os
import pickle
import io
import base64
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
import re
import urllib
import seaborn as sns

# -- Globals -------------------------------------------------------------------
global uname, graph, en_rf, hi_rf

# -- Load BERT (multilingual -- handles English AND Hindi natively) -------------
bert = SentenceTransformer('bert-base-multilingual-cased')
print("BERT initialization completed")

accuracy  = []
precision = []
recall    = []
fscore    = []

# -- Metrics helper ------------------------------------------------------------
def calculateMetrics(algorithm, y_test, predict):
    a = round(accuracy_score(y_test, predict) * 100, 3)
    p = round(precision_score(y_test, predict, average='macro') * 100, 3)
    r = round(recall_score(y_test, predict, average='macro') * 100, 3)
    f = round(f1_score(y_test, predict, average='macro') * 100, 3)
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)

# -- Helper: verify a .npy file contains float vectors (real BERT output) ------
def _npy_is_float(path):
    try:
        arr = np.load(path, allow_pickle=True)
        return np.issubdtype(arr.dtype, np.floating)
    except Exception:
        return False

# -- Load dataset --------------------------------------------------------------
dataset = pd.read_csv('Dataset/data-en-hi-de-fr.csv')
dataset = dataset.values

# Check if valid float-encoded embeddings are already cached
_en_ok = os.path.exists("model/en_X.npy") and _npy_is_float("model/en_X.npy")
_hi_ok = os.path.exists("model/hi_X.npy") and _npy_is_float("model/hi_X.npy")
_y_ok  = os.path.exists("model/Y.npy")

if _en_ok and _hi_ok and _y_ok:
    print("Loading cached BERT embeddings...")
    en_X = np.load("model/en_X.npy")
    hi_X = np.load("model/hi_X.npy")
    Y    = np.load("model/Y.npy")
else:
    # Build embeddings from scratch
    en_X_text = []
    hi_X_text = []
    Y_list    = []

    for i in range(len(dataset)):
        label   = str(dataset[i, 0]).strip().lower()
        english = str(dataset[i, 1]).strip().lower()
        hindi   = str(dataset[i, 2]).strip()
        if len(label) > 0 and len(english) > 0:
            english = re.sub('[^a-z]+', ' ', english)
            en_X_text.append(english)
            hi_X_text.append(hindi)
            Y_list.append(0 if label == "ham" else 1)
            print(str(i) + " " + label + " " + english + " " + hindi)

    Y = np.asarray(Y_list)
    np.save("model/Y", Y)

    # BERT encode English
    print("Encoding English text with BERT (may take a few minutes)...")
    en_X = bert.encode(en_X_text, convert_to_tensor=True).numpy()
    np.save("model/en_X", en_X)
    print("English BERT vectors shape: " + str(en_X.shape))

    # BERT encode Hindi -- multilingual BERT supports Hindi script natively
    print("Encoding Hindi text with BERT (may take a few minutes)...")
    hi_X = bert.encode(hi_X_text, convert_to_tensor=True).numpy()
    np.save("model/hi_X", hi_X)
    print("Hindi BERT vectors shape: " + str(hi_X.shape))

    # Remove stale trained model so classifiers retrain on fresh embeddings
    if os.path.exists("model/models.pckl"):
        os.remove("model/models.pckl")
        print("Removed stale models.pckl -- will retrain with fresh embeddings")

# -- Train/test split (must happen BEFORE model loading/training) --------------
X_train,    X_test,    y_train,    y_test    = train_test_split(en_X, Y, test_size=0.2)
X_train_hi, X_test_hi, y_train_hi, y_test_hi = train_test_split(hi_X, Y, test_size=0.2)

# -- Load or train classifiers -------------------------------------------------
if os.path.exists("model/models.pckl"):
    print("Loading saved classifiers...")
    with open('model/models.pckl', 'rb') as f:
        models = pickle.load(f)
    en_rf, hi_rf = models
else:
    print("Training English Random Forest...")
    en_rf = RandomForestClassifier()
    en_rf.fit(X_train, y_train)

    print("Training Hindi Random Forest...")
    hi_rf = RandomForestClassifier()
    hi_rf.fit(X_train_hi, y_train_hi)

    with open('model/models.pckl', 'wb') as f:
        pickle.dump([en_rf, hi_rf], f)
    print("Models saved.")

# -- Evaluate ------------------------------------------------------------------
predict     = en_rf.predict(X_test)
conf_matrix = confusion_matrix(y_test, predict)
calculateMetrics("MBERT Spam Detection", y_test, predict)

predict_hi = hi_rf.predict(X_test_hi)
calculateMetrics("Random Forest URL Classification", y_test_hi, predict_hi)

# -- URL feature extractor -----------------------------------------------------
def get_features(df):
    needed_cols = ['url', 'domain', 'path', 'query', 'fragment']
    for col in needed_cols:
        df[col + '_length']           = df[col].str.len()
        df['qty_dot_' + col]          = df[col].apply(lambda x: x.count('.'))
        df['qty_hyphen_' + col]       = df[col].apply(lambda x: x.count('-'))
        df['qty_slash_' + col]        = df[col].apply(lambda x: x.count('/'))
        df['qty_questionmark_' + col] = df[col].apply(lambda x: x.count('?'))
        df['qty_equal_' + col]        = df[col].apply(lambda x: x.count('='))
        df['qty_at_' + col]           = df[col].apply(lambda x: x.count('@'))
        df['qty_and_' + col]          = df[col].apply(lambda x: x.count('&'))
        df['qty_exclamation_' + col]  = df[col].apply(lambda x: x.count('!'))
        df['qty_space_' + col]        = df[col].apply(lambda x: x.count(' '))
        df['qty_tilde_' + col]        = df[col].apply(lambda x: x.count('~'))
        df['qty_comma_' + col]        = df[col].apply(lambda x: x.count(','))
        df['qty_plus_' + col]         = df[col].apply(lambda x: x.count('+'))
        df['qty_asterisk_' + col]     = df[col].apply(lambda x: x.count('*'))
        df['qty_hashtag_' + col]      = df[col].apply(lambda x: x.count('#'))
        df['qty_dollar_' + col]       = df[col].apply(lambda x: x.count('$'))
        df['qty_percent_' + col]      = df[col].apply(lambda x: x.count('%'))

# -- Load XGBoost URL classifier -----------------------------------------------
with open('model/xgb.txt', 'rb') as file:
    xgb_cls = pickle.load(file)


# ==============================================================================
# VIEWS
# ==============================================================================

def TrainModels(request):
    if request.method == 'GET':
        global accuracy, precision, recall, fscore, conf_matrix
        labels  = ['Ham', 'Spam']
        output  = '<table border=1 align=center width=100%><tr>'
        output += '<th><font color="black">Algorithm Name</th>'
        output += '<th><font color="black">Accuracy</th>'
        output += '<th><font color="black">Precision</th>'
        output += '<th><font color="black">Recall</th>'
        output += '<th><font color="black">FSCORE</th></tr>'
        algorithms = ['MBERT Spam Detection', 'Random Forest URL Classification']
        for i in range(len(algorithms)):
            output += ('<tr><td>' + algorithms[i] + '</td><td>' + str(accuracy[i]) + '</td>'
                       '<td>' + str(precision[i]) + '</td><td>' + str(recall[i]) + '</td>'
                       '<td>' + str(fscore[i]) + '</td></tr>')
        output += "</table></br>"

        df = pd.DataFrame([
            ['MBERT Spam Detection',             'Accuracy',  accuracy[0]],
            ['MBERT Spam Detection',             'Precision', precision[0]],
            ['MBERT Spam Detection',             'Recall',    recall[0]],
            ['MBERT Spam Detection',             'FSCORE',    fscore[0]],
            ['Random Forest URL Classification', 'Accuracy',  accuracy[1]],
            ['Random Forest URL Classification', 'Precision', precision[1]],
            ['Random Forest URL Classification', 'Recall',    recall[1]],
            ['Random Forest URL Classification', 'FSCORE',    fscore[1]],
        ], columns=['Parameters', 'Algorithms', 'Value'])

        figure, axis = plt.subplots(nrows=1, ncols=2, figsize=(10, 3))
        axis[0].set_title("Confusion Matrix Prediction Graph")
        axis[1].set_title("All Algorithms Performance Graph")
        ax = sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels,
                         annot=True, cmap="viridis", fmt="g", ax=axis[0])
        ax.set_ylim([0, len(labels)])
        df.pivot("Parameters", "Algorithms", "Value").plot(ax=axis[1], kind='bar')
        plt.title("All Algorithms Performance Graph")
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        plt.clf()
        plt.cla()
        context = {'data': output, 'img': img_b64}
        return render(request, 'UserScreen.html', context)


def prediction(msg, model):
    msg_vec = bert.encode([msg], convert_to_tensor=True).numpy()
    pred    = model.predict(msg_vec)
    if pred[0] == 1:
        return "<font size=4 color=red>SPAM</font>"
    return "<font size=4 color=green>HAM</font>"


def SMSPredictAction(request):
    if request.method == 'POST':
        global bert, en_rf, hi_rf
        msg  = request.POST.get('t1', False)
        lang = request.POST.get('t2', False)
        if lang == "English":
            msg    = re.sub('[^a-z]+', ' ', msg.strip().lower())
            output = prediction(msg, en_rf)
        else:
            output = prediction(msg, hi_rf)
        context = {'data': 'SMS Predicted As : ' + output}
        return render(request, 'SMSPredict.html', context)


def URLPredictAction(request):
    if request.method == 'POST':
        global xgb_cls
        testURL = request.POST.get('t1', False)
        data    = pd.DataFrame([[testURL]], columns=['url'])
        urls    = data['url'].tolist()
        data['protocol'], data['domain'], data['path'], data['query'], data['fragment'] = \
            zip(*[urllib.parse.urlsplit(x) for x in urls])
        get_features(data)
        data = data.drop(columns=['url', 'protocol', 'domain', 'path', 'query', 'fragment'])
        data = normalize(data.values)
        pred = xgb_cls.predict(data)[0]
        if pred == 0:
            output = testURL + " <font size=3 color=green>====> Predicted AS Normal Link</font>"
        else:
            output = testURL + " <font size=3 color=red>====> Predicted AS Malicious Link</font>"
        context = {'data': output}
        return render(request, 'URLPredict.html', context)


def URLPredict(request):
    if request.method == 'GET':
        return render(request, 'URLPredict.html', {})


def SMSPredict(request):
    if request.method == 'GET':
        return render(request, 'SMSPredict.html', {})


def UserLogin(request):
    if request.method == 'GET':
        return render(request, 'UserLogin.html', {})


def index(request):
    if request.method == 'GET':
        return render(request, 'index.html', {})


def UserLoginAction(request):
    if request.method == 'POST':
        global uname
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        if username == "admin" and password == "admin":
            context = {'data': 'welcome ' + username}
            return render(request, 'UserScreen.html', context)
        else:
            context = {'data': 'login failed'}
            return render(request, 'UserLogin.html', context)


def LoadDataset(request):
    if request.method == 'GET':
        output  = "Total Records found in Dataset = " + str(en_X.shape[0]) + "<br/>"
        output += "<br/>Labels found in Dataset = Ham &amp; Spam<br/>"
        output += "<br/>Dataset Train &amp; Test Split Details<br/>"
        output += "80% records used to train Algorithms : " + str(X_train.shape[0]) + "<br/>"
        output += "20% records used to test  Algorithms : " + str(X_test.shape[0])  + "<br/><br/>"

        ds      = pd.read_csv("Dataset/data-en-hi-de-fr.csv", usecols=['labels', 'text', 'text_hi'])
        columns = ds.columns
        ds      = ds.values
        output += '<table border=1 align=center width=100%><tr>'
        for col in columns:
            output += '<th><font size="3" color="black">' + str(col) + '</th>'
        output += '</tr>'
        for row in ds:
            output += '<tr>'
            for cell in row:
                output += '<td><font size="3" color="black">' + str(cell) + '</td>'
            output += '</tr>'
        output += "</table></br></br></br></br>"
        context = {'data': output}
        return render(request, 'UserScreen.html', context)