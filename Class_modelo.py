import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV,StratifiedKFold, KFold
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve, auc,precision_score,recall_score, f1_score
from sklearn.preprocessing import RobustScaler
import os
import random
import string
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

## 
import pandas as pd
import plotly_express as px
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind,mannwhitneyu

class model_final:
    def __init__(self) -> None:
        #Data
        data = pd.read_csv('./creditcard.csv')
        data = data.drop_duplicates()
        self.data = data
    def preprocessing(self,imbalanced_tec=None, selecao_feature=None):
        data = self.data
        #X e y
        X = data.iloc[:,:-1]
        y = data.iloc[:,-1]
        
        #Seleção de Features:
        
        # Metodo1: Variáveis indicadas pelo T-test
        if selecao_feature == 'Metodo1':
            X = X.drop(columns=['V13','V15','V22','V23','V25','V26'])
            #Irá identificar nos arquivos o método
            pp = 'M1'
        # Metodo2: Variáveis indicadas pelo U-test
        elif selecao_feature == 'Metodo2':
            X = X.drop(columns=['V13','V15','V22','V25'])
            pp = 'M2'
        # Metodo3: Variáveis indicadas pela Correlação Pearson:
        elif selecao_feature == 'Metodo3':
            X = X.drop(columns=['V13','V15','V22','V23','V24','V25','V26'])
            pp = 'M3'

        #Train e Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
            
        #Aplicando Roubust Scaler para deixar as variáveis em mesma escala
        scaler = RobustScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        
        #Método de Resample:
        if imbalanced_tec:
            X_train,y_train = imbalanced_tec.fit_resample(X_train,y_train)
        
        
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.pp = pp
        
    
    def modelo(self,modelo,nome_modelo_file,GridSearch = False):
        modelo.fit(self.X_train,self.y_train)
        y_pred = modelo.predict(self.X_test)
        y_pred_proba = modelo.predict_proba(self.X_test)

        
        self.modelo = modelo
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba
        self.GridSearch = GridSearch
        self.nome_modelo_file = nome_modelo_file
    
    def Relatorio_modelo(self):
        # Nome Modelo
        if self.GridSearch:
            pos = str(self.modelo.best_estimator_).find('(')
            nome_modelo = str(self.modelo.best_estimator_)[:pos]
            params = self.modelo.best_params_
        else:
            pos_1 = str(self.modelo).find('(')
            pos_2 = str(self.modelo).find(')')
            params = str(self.modelo)[pos_1 + 1:pos_2]
            nome_modelo = str(self.modelo)[:pos_1]
        #Data Frame Relatorio Classificacao Threshold = 0.5
        dict_relatorio_classificacao = {
                  'Modelo':[nome_modelo],
                  'Parametros_Modelo': [params],
                  'f1score_0':[f1_score(self.y_test,self.y_pred,pos_label=0)],  
                  'precision_0':[precision_score(self.y_test,self.y_pred,pos_label=0)],
                  'recall_0':[recall_score(self.y_test,self.y_pred,pos_label=0)],
                  'f1score_1':[f1_score(self.y_test,self.y_pred,pos_label=1)],  
                  'precision_1':[precision_score(self.y_test,self.y_pred,pos_label=1)],
                  'recall_1':[recall_score(self.y_test,self.y_pred,pos_label=1)]                              
                  }
        dict_resultados_pred = {
                        'index': self.y_test.index,
                        'y_test': self.y_test.values,
                        'y_pred':self.y_pred,
                        'y_proba_0': self.y_pred_proba[:,0],
                        'y_proba_1': self.y_pred_proba[:,1]       
                        }
        
        
        # Relatorio Amplo Classificacao para diferentes Threshold:
        
        # Calculo da precision-recall
        precision, recall, thresholds = precision_recall_curve(self.y_test, self.y_pred_proba[:,1])

        # DataFrame com os resultados
        df_threshold = pd.DataFrame({
            'Threshold': thresholds,
            'Precision': precision[:-1],
            'Recall': recall[:-1]
        })

        # Queremos valores acima da aleatóriedade
        df_threshold = df_threshold[(df_threshold['Precision'] > 0.5)&(df_threshold['Recall'] > 0.5)]
        df_threshold.reset_index(inplace=True,drop=True)

        # F1-Score
        df_threshold['F1-score'] = 2 * (df_threshold['Precision'] * df_threshold['Recall']) / (df_threshold['Precision'] + df_threshold['Recall'])

        # Abastecendo o TP e FP
        for pos,threshold in enumerate(df_threshold['Threshold']):
            y_pred_new = np.where(self.y_pred_proba[:,1] >= threshold, 1 ,0)
            cm = confusion_matrix(self.y_test,y_pred_new)
            TP = cm[1,1]
            FP = cm[0,1]
            df_threshold.loc[pos,'TP'] = TP
            df_threshold.loc[pos,'FP'] = FP



        df_relatorio = pd.DataFrame(dict_relatorio_classificacao)
        df_resultados = pd.DataFrame(dict_resultados_pred)
        
        
        self.df_relatorio = df_relatorio
        self.df_resultados =  df_resultados
        self.df_threshold = df_threshold

    def salvar_modelo(self):
        
        # Criando a pasta do Modelo
        nome_pasta = f'./Modelos_Salvos/{self.nome_modelo_file}_{self.pp}'
        # Verifica se a pasta já existe
        if os.path.exists(nome_pasta):
            # Se a pasta existe, gera uma sequência aleatória de letras
            sufixo_aleatorio = ''.join(random.choices(string.ascii_lowercase, k=3))
            # Adiciona o sufixo ao nome do modelo
            nome_modelo_file += f'_{sufixo_aleatorio}'
            # Cria o novo caminho da pasta
            nome_pasta = f'./Modelos_Salvos/{self.nome_modelo_file}'
        else:
            os.makedirs(nome_pasta)
        
        # Salvando os dados sobre Modelo
        self.df_resultados.to_csv(f'{nome_pasta}/resultados.csv')
        self.df_relatorio.to_csv(f'{nome_pasta}/relatorio.csv')
        self.df_threshold.to_csv(f'{nome_pasta}/thresholds.csv')
        #Salvando o Modelo
        with open(f'{nome_pasta}/{self.nome_modelo_file}.pkl','wb') as file:
            pickle.dump(self.modelo,file)
    
            
    
            