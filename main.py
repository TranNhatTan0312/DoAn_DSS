from cgi import test
from re import X
from tkinter import Y
from PyQt5 import QtCore, QtGui, QtWidgets


# app = FastAPI(debug=True)

# @app.get('/')
# def home():
#     return {'text':'Khang dan nhu con cho'}

# if __name__ == '__main__':
#     uvicorn.run(app)

#  phần xử lí
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os 
import sys
import pickle
import numpy as np
#For training
def train() -> None:
    with open('DoAn_DSS/heart_Disease_prediction_new_final.csv') as f:
        df = pd.read_csv(f)
    # df = df.drop(['contact', 'poutcome'], axis=1)
    df_filtered = df.replace('unknown',np.nan)
    df_filtered.dropna(inplace=True)
    df_filtered.reset_index(drop=True, inplace=True)
    dataset = df_filtered.copy()
    accuracies = {}
    times = {}
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for col in dataset.columns[ [i == object for i in dataset.dtypes] ]:
        dataset.loc[:,col] = le.fit_transform(dataset[col])
    dataset = dataset[['age', 'cp', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope','target']]
 
    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
 
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    ct = ColumnTransformer(transformers=[], remainder='passthrough' )
    x = np.array(ct.fit_transform(x))
 
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y)

    
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    ct = ColumnTransformer(transformers=[], remainder='passthrough' )
    x = np.array(ct.fit_transform(x))
 
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y)
 
#train test split
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators =10, criterion='entropy', random_state=0)
    classifier.fit(x_train, y_train)
    R= classifier.fit(x_train,y_train)
 
    
#Save Model As Pickle File 
    with open('R.pkl','wb') as m:
        pickle.dump(R,m)
    test(x_test,y_test)
 
#Test accuracy of the model 
def test(X_test,Y_test):
    with open('R.pkl','rb') as mod: 
        p=pickle.load(mod)
    pre=p.predict(X_test)
    print (accuracy_score(Y_test,pre)) #Prints the accuracy of the model
 
def find_data_file(filename):
    if getattr(sys, "frozen", False): # The application is frozen.
        datadir = os.path.dirname(sys.executable)
    else:
# The application is not frozen.
        datadir = os.path.dirname( __file__)
    return os.path.join(datadir, filename)
 
def check_input(data) ->int :
    df=pd.DataFrame(data=data,index=[0])
    with open(find_data_file('R.pkl'),'rb') as model:
        p=pickle.load(model)
    op=p.predict(df)
    return op


from PyQt5 import QtCore, QtGui, QtWidgets

# from f import *

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 524)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(30, 10, 741, 51))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(60)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(30, 80, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setKerning(False)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(30, 110, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setKerning(False)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(30, 140, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setKerning(False)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(30, 170, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setKerning(False)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.label_1 = QtWidgets.QLabel(self.centralwidget)
        self.label_1.setGeometry(QtCore.QRect(30, 50, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setKerning(False)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label_6")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(30, 200, 111, 31))
        # font = QtGui.QFont()
        # font.setPointSize(14)
        # font.setKerning(False)
        # self.label.setFont(font)
        # self.label.setAlignment(QtCore.Qt.AlignCenter)
        # self.label.setObjectName("label_8")
        # self.label_8 = QtWidgets.QLabel(self.centralwidget)
        # self.label_8.setGeometry(QtCore.QRect(30, 260, 111, 31))
        # font = QtGui.QFont()
        # font.setPointSize(14)
        # font.setKerning(False)
        # self.label.setFont(font)
        # self.label.setAlignment(QtCore.Qt.AlignCenter)
        # self.label.setObjectName("label_9")
        # self.label_9 = QtWidgets.QLabel(self.centralwidget)
        # self.label_9.setGeometry(QtCore.QRect(30, 290, 111, 31))
        # font = QtGui.QFont()
        # font.setPointSize(14)
        # font.setKerning(False)
        # self.label.setFont(font)
        # self.label.setAlignment(QtCore.Qt.AlignCenter)
        # self.label.setObjectName("label_10")
        # self.label_10 = QtWidgets.QLabel(self.centralwidget)
        # self.label_10.setGeometry(QtCore.QRect(30, 310, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setKerning(False)
        self.label_1.setFont(font)
        self.label_1.setObjectName("label_1")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_2.setGeometry(QtCore.QRect(150, 80, 611, 31))
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.lineEdit_3 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_3.setGeometry(QtCore.QRect(150, 110, 611, 31))
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.lineEdit_4 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_4.setGeometry(QtCore.QRect(150, 140, 611, 31))
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.lineEdit_1 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_1.setGeometry(QtCore.QRect(150, 50, 611, 31))
        self.lineEdit_1.setObjectName("lineEdit_1")
        self.lineEdit_5 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_5.setGeometry(QtCore.QRect(150, 170, 611, 31))
        self.lineEdit_5.setObjectName("lineEdit_5")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(50, 360, 251, 71))
        self.lineEdit_7 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_7.setGeometry(QtCore.QRect(150, 230, 611, 31))
        self.lineEdit_7.setObjectName("lineEdit_7")
        # self.lineEdit_8 = QtWidgets.QLineEdit(self.centralwidget)
        # self.lineEdit_8.setGeometry(QtCore.QRect(150, 260, 611, 31))
        # self.lineEdit_8.setObjectName("lineEdit_8")
        # self.lineEdit_9 = QtWidgets.QLineEdit(self.centralwidget)
        # self.lineEdit_9.setGeometry(QtCore.QRect(150, 290, 611, 31))
        # self.lineEdit_9.setObjectName("lineEdit_9")
        # self.lineEdit_10 = QtWidgets.QLineEdit(self.centralwidget)
        # self.lineEdit_10.setGeometry(QtCore.QRect(150, 320, 611, 31))
        # self.lineEdit_10.setObjectName("lineEdit_10")
        # self.lineEdit_10 = QtWidgets.QLineEdit(self.centralwidget)
        # self.lineEdit_10.setGeometry(QtCore.QRect(150, 350, 611, 31))
        # self.lineEdit_10.setObjectName("lineEdit_10")

        font = QtGui.QFont()
        font.setPointSize(32)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(490,360, 251, 71))
        font = QtGui.QFont()
        font.setPointSize(32)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        self.lineEdit_6 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_6.setGeometry(QtCore.QRect(150, 200, 611, 31))
        self.lineEdit_6.setObjectName("lineEdit_6")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(30, 230, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setKerning(False)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
 
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
 
        self.pushButton.clicked.connect(self.Crun)
        self.pushButton_2.clicked.connect(self.Clr)
        # train()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Phần mềm dự đoán"))
        self.label.setText(_translate("MainWindow", "DỰ ĐOÁN KHẢ NĂNG MẮC BỆNH TIM"))
        self.label_1.setText(_translate("MainWindow", "age"))
        self.label_2.setText(_translate("MainWindow", "cp"))
        self.label_3.setText(_translate("MainWindow", "trestbps"))
        self.label_4.setText(_translate("MainWindow", "chol"))
        self.label_5.setText(_translate("MainWindow", "thalach"))
        self.label_6.setText(_translate("MainWindow", "oldpeak"))
        self.label_7.setText(_translate("MainWindow", "slope"))
        # self.label_9.setText(_translate("MainWindow", "sex"))
        # self.label_10.setText(_translate("MainWindow", "slope"))
        self.pushButton.setText(_translate("MainWindow", "CHẠY"))
        self.pushButton_2.setText(_translate("MainWindow", "XÓA"))
        # self.label_6.setText(_translate("MainWindow", "thalach"))
    def Clr(self) -> None:
        self.lineEdit_1.clear()
        self.lineEdit_2.clear()
        self.lineEdit_3.clear()
        self.lineEdit_4.clear()
        self.lineEdit_5.clear()
        self.lineEdit_6.clear()
        self.lineEdit_7.clear()
        # self.lineEdit_8.clear()
        # self.lineEdit_9.clear()
        # self.lineEdit_10.clear()
    def Crun(self) -> None:
        my_dict =   {"age":float(self.lineEdit_1.text()), "cp":float(self.lineEdit_2.text()), "trestbps":float(self.lineEdit_3.text())
        , "chol":float(self.lineEdit_4.text()), "thalach":float(self.lineEdit_5.text()), "oldpeak":float(self.lineEdit_6.text()), 
        "slope":float(self.lineEdit_7.text())} 
        t=str('Bệnh nhân')
        print(my_dict)
    
        output = check_input(my_dict)
        print(output)  
 
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Information)
        a = ""
        if output == 0:
            a="KHÔNG CÓ KHẢ NĂNG"
        else:
            a="CÓ KHẢ NĂNG"
            msg.setInformativeText(" {} {}  bị mắc bệnh tim".format(t,str(a)))
        msg.setWindowTitle("Kết quả")
        msg.exec_() 
    
    # from sklearn.metrics import accuracy_score

if __name__ == '__main__':
        train()        
        app = QtWidgets.QApplication(sys.argv)
        MainWindow = QtWidgets.QMainWindow()
        ui = Ui_MainWindow()
        ui.setupUi(MainWindow)
        MainWindow.show()
        sys.exit(app.exec_())




