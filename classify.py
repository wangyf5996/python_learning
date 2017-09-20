# -*- coding: utf-8 -*-
import numpy as np
from sklearn import preprocessing
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import cross_validation
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
import sys
from numpy.linalg import lstsq
import os
import cPickle
import xgboost as xgb

def save_model(model, file_name): 
    #xx.pkl
    with open(file_name, 'wb') as fid:
        cPickle.dump(model, fid)


def load_model(file_name):
    with open(file_name, 'rb') as fid:
        model = cPickle.load(fid)
        return model
 

def logistic_regresson_for_multiclass():
    text_file = "/home/web_server/wangyuanfu/age/temp"
    dataset = np.loadtxt(text_file, delimiter=" ")
    X = dataset[:,1:]
    y = dataset[:,0:1]
    min_max_scaler = preprocessing.MinMaxScaler()
    normalized_X = min_max_scaler.fit_transform(X)
    print len(normalized_X)

    X_train, X_test, y_train, y_test = train_test_split(normalized_X, y, test_size=0.1, random_state=7)


    for multi_class in ('multinomial', 'ovr'):
        clf = LogisticRegression(solver='sag', max_iter=100, random_state=42, multi_class=multi_class, penalty='l2',C=0.001).fit(X_train, y_train)

        # print the training scores
        print("training score : %.3f (%s)" % (clf.score(X_train, y_train), multi_class))

        #print clf.coef_
        #print clf.intercept_

        # make predictions
        predicted = clf.predict(X_test)
        probability =  clf.predict_proba(X_test)
        length_predicted = len(predicted)
        #for i in range(0,length_predicted):
            #print predicted[i],y_test[i],probability[i]
            #print X_test[i,:],predicted[i],y_test[i],probability[i]
        # summarize the fit of the model
        print(metrics.classification_report(y_test, predicted))
        print(metrics.confusion_matrix(y_test, predicted))
        print(metrics.precision_score(y_test, predicted, average='micro'))
        #print metrics.roc_auc_score(y_test, predicted)


def svm_for_multiclass():
    text_file = "/home/web_server/wangyuanfu/age/temp1"
    dataset = np.loadtxt(text_file, delimiter=" ")
    X = dataset[:,1:]
    y = dataset[:,0:1]
    min_max_scaler = preprocessing.MinMaxScaler()
    normalized_X = min_max_scaler.fit_transform(X)
    print len(normalized_X)

    X_train, X_test, y_train, y_test = train_test_split(normalized_X, y, test_size=0.1, random_state=7)


    clf = LinearSVC(random_state=0, C=1, multi_class='ovr', penalty='l2')
    clf = clf.fit(X_train, y_train.reshape(-1))
    # print the training scores
    print("training score : %.3f " % (clf.score(X_train, y_train)))

    # make predictions
    predicted = clf.predict(X_test)
    length_predicted = len(predicted)
    print predicted.shape
    #for i in range(0,length_predicted):
    #    print predicted[i],y_test[i]
        #print X_test[i,:],predicted[i],y_test[i],probability[i]
    # summarize the fit of the model
    print(metrics.classification_report(y_test, predicted))
    print(metrics.confusion_matrix(y_test, predicted))
    print(metrics.precision_score(y_test, predicted, average='micro'))

   #print metrics.roc_auc_score(y_test, predicted)

def randomforest_for_multiclass():
    text_file = "/home/web_server/wangyuanfu/age/temp1"
    dataset = np.loadtxt(text_file, delimiter=" ")
    X = dataset[:,1:]
    y = dataset[:,0:1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=7)
    clf = RandomForestClassifier(n_estimators=25, max_depth=6)
    clf.fit(X_train, y_train.reshape(-1))

    scores = clf.score(X_test, y_test)
    predicted = clf.predict(X_test)
    print predicted.shape

    #print(clf.feature_importances_)
    probability =  clf.predict_proba(X_test)
    print(metrics.classification_report(y_test, predicted))
    print(metrics.confusion_matrix(y_test, predicted))
    print(metrics.precision_score(y_test, predicted, average='micro'))
    #score = log_loss(y_test, clf_probs)


def print_importance(gbt):
    for i, f in sorted(((i, f) for (f, i) in gbt.get_fscore().iteritems()), reverse=True):
        print i, f

def show_tree(gbt):
    plt.figure(figsize=(60, 32))
    ax = plt.subplot()
    xgb.plotting.plot_tree(gbt, ax=ax) #, feature_names=ft.features)
    plt.show()

def xgboost_for_multiclass():
    text_file = "/home/web_server/wangyuanfu/age/temp1"
    params={'booster':'gbtree',
    # 这里手写数字是0-9，是一个多类的问题，因此采用了multisoft多分类器，
    'objective': 'multi:softmax', 
    'num_class':7, # 类数，与 multisoftmax 并用
    'gamma':0.05,  # 在树的叶子节点下一个分区的最小损失，越大算法模型越保守 。[0:]
    'max_depth':7, # 构建树的深度 [1:]
    'lambda':0.1,  # L2 正则项权重
    'subsample':0.5, # 采样训练数据，设置为0.5，随机选择一般的数据实例 (0:1]
    'colsample_bytree':0.7, # 构建树树时的采样比率 (0:1]
    #'min_child_weight':12, # 节点的最少特征数
    #'eval_metric':'mlogloss',
    'silent':1 ,
    'eta': 0.05, # 如同学习率
    'seed':710,
    'nthread':8# cpu 线程数,根据自己U的个数适当调整
    }

    #加载numpy数组到DMatrix中
    dataset = np.loadtxt(text_file, delimiter=" ")
    X = dataset[:,1:]
    y = dataset[:,0:1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=7)
    
    xgtrain = xgb.DMatrix(X_train, label=y_train)
    xgval = xgb.DMatrix(X_test, label=y_test)
    #xgtest = xgb.DMatrix(test)

    # return 训练和验证的错误率
    num_rounds = 50 # 迭代你次数
    watchlist = [(xgtrain, 'train'),(xgval, 'val')]
    watchlist2 = [(xgtrain, 'train')]
    features = []

    # training model 
    # early_stopping_rounds 当设置的迭代次数较大时，early_stopping_rounds 可在一定的迭代次数内准确率没有提升就停止训练
    #model = xgb.train(param, xgtrain, num_rounds, watchlist, early_stopping_rounds=100)
    model = xgb.train(params, xgtrain, num_rounds, watchlist, early_stopping_rounds=100)
    #model.save_model('./xgb.model') # 用于存储训练出的模型
    #model = xgb.load_model('./xgb.model')

    predicted = model.predict(xgval, ntree_limit=model.best_iteration)
    print predicted
    print predicted.shape

    #print(clf.feature_importances_)
    print(metrics.classification_report(y_test, predicted))
    print(metrics.confusion_matrix(y_test, predicted))
    print(metrics.precision_score(y_test, predicted, average='micro'))
    #score = log_loss(y_test, clf_probs)

def xgboost_for_multiclass_svm_input_format():
    params={'booster':'gbtree',
    # 这里手写数字是0-9，是一个多类的问题，因此采用了multisoft多分类器，
    'objective': 'multi:softmax', 
    'num_class':8, # 类数，与 multisoftmax 并用
    'gamma':0.05,  # 在树的叶子节点下一个分区的最小损失，越大算法模型越保守 。[0:]
    'max_depth':7, # 构建树的深度 [1:]
    'lambda':0.1,  # L2 正则项权重
    'subsample':0.5, # 采样训练数据，设置为0.5，随机选择一般的数据实例 (0:1]
    'colsample_bytree':0.7, # 构建树树时的采样比率 (0:1]
    #'min_child_weight':12, # 节点的最少特征数
    #'eval_metric':'mlogloss',
    'silent':1 ,
    'eta': 0.05, # 如同学习率
    'seed':710,
    'nthread':8# cpu 线程数,根据自己U的个数适当调整
    }

    # #dtrain.cache  Using XGBoost External Memory Version
    test_file = "test_temp1"
    xgtrain = xgb.DMatrix('temp1#dtrain.cache')
    xgtest = xgb.DMatrix(test_file)    
    #xgtest = xgb.DMatrix(test)

    # return 训练和验证的错误率
    num_rounds = 2 # 迭代你次数
    watchlist2 = [(xgtrain, 'train')]
    features = []

    # training model 
    # early_stopping_rounds 当设置的迭代次数较大时，early_stopping_rounds 可在一定的迭代次数内准确率没有提升就停止训练
    #model = xgb.train(param, xgtrain, num_rounds, watchlist, early_stopping_rounds=100)
    model = xgb.train(params, xgtrain, num_rounds, watchlist2, early_stopping_rounds=50)
    #model.save_model('./xgb.model') # 用于存储训练出的模型
    #model = xgb.load_model('./xgb.model')

    predicted = model.predict(xgtest, ntree_limit=model.best_iteration)
    print predicted
    print predicted.shape

    y_test = []
    for line in open(test_file):
        y_test.append(line.strip().split(' ')[0])

    print(metrics.classification_report(y_test, predicted))
    print(metrics.confusion_matrix(y_test, predicted))
    print(metrics.precision_score(y_test, predicted, average='micro'))
    #score = log_loss(y_test, clf_probs)

def eval_classification(gbt, path):
    data = xgb.DMatrix(path)
    data = data_target_transfrom(data, lambda x: x > 0.)
    y = data.get_label()
    y_pred = gbt.predict(data)
    auc = metrics.roc_auc_score(y, y_pred)
    fpr, tpr, _ = metrics.roc_curve(y, y_pred)
    plt.plot(fpr, tpr)
    plt.xlabel('false positive rate')
    plt.ylabel('true postive rate')
    plt.show()
    auc = metrics.auc(fpr, tpr, reorder=True)
    print 'auc', auc
    #print 'f1', metrics.auc(y, y_pred > 0.5)
    print 'accuracy', metrics.accuracy_score(y, y_pred > 0.5)


def data_process():
    file_writer = open('/media/disk1/fordata/wangyuanfu/feature_detail','w')
    dimension = [1170,4,10,4,360,47,2,38,359]
    #float_feature_dimension = 827
    #total_feature_dimension = 837
    float_feature_dimension = 377
    total_feature_dimension = 387
    rootDir = "/media/disk1/fordata/wangyuanfu/zz_wyf_label_age_small"
    for lists in os.listdir(rootDir):
        path = os.path.join(rootDir, lists)
        if path.find("_") == -1:
            continue
        for line in open(path):
            fields = line.strip().split('\x01')
            towrite = fields[0]
            for i in range(2, float_feature_dimension):
                towrite = towrite + " " + fields[i]
            '''
            for i in range(float_feature_dimension, total_feature_dimension-1):
                temp_list = [0 for j in range(dimension[i-float_feature_dimension])]
                temp_list[int(fields[i])-1] = 1
                temp_list_str = [str(m) for m in temp_list]
                towrite = towrite + " " + ' '.join(temp_list_str)
            '''
            print >>file_writer, towrite
    file_writer.close()

def data_process_svm_light():
    file_writer = open('/home/web_server/wangyuanfu/age/feature_detail_svm_no_data_clean','w')
    #float_feature_dimension = 827
    #total_feature_dimension = 837
    float_feature_dimension = 377
    total_feature_dimension = 387
    dimension = [1170,4,10,4,360,47,2,38,359]
    offset = float_feature_dimension
    rootDir = "/home/web_server/wangyuanfu/age/zz_wyf_label_age_small"
    #rootDir = "/media/disk1/fordata/wangyuanfu/temp"
    for lists in os.listdir(rootDir):
        path = os.path.join(rootDir, lists)
        if path.find("_") == -1:
            continue
        for line in open(path):
            fields = line.strip().split('\x01')
            towrite = fields[0]
            offset = float_feature_dimension
            for i in range(2, float_feature_dimension):
                towrite = towrite + " " + str(i-1) + ":" + fields[i]
            for i in range(float_feature_dimension, total_feature_dimension-1):
                towrite = towrite + " " + str(offset+int(fields[i])-1) + ":1"
                offset = offset + dimension[i-float_feature_dimension]
            topic_list = fields[total_feature_dimension-1].split(':')
            if len(topic_list) < 2:
                continue
            towrite = towrite + " " + str(offset+int(topic_list[0])) + ":" + topic_list[1]
            print >>file_writer, towrite
    file_writer.close()

if __name__ == '__main__':
    '''
    last_user = ""
    file_writer = open("one_user",'w')
    for line in sys.stdin:
        line = line.strip()
        fields = line.split('\x01')
        if fields[1] != last_user and last_user != "":
            file_writer.close()
            stardard_logistic_regression(last_user)
            file_writer = open("one_user",'w')
        last_user = fields[1]
        print >>file_writer, line
    file_writer.close()
    stardard_logistic_regression(last_user)
    '''
    #logistic_regresson_for_multiclass()
    #svm_for_multiclass()
    #randomforest_for_multiclass()
    #xgboost_for_multiclass()
    #xgboost_for_multiclass_svm_input_format()

    data_process_svm_light()
    #data_process()
