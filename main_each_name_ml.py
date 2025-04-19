from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import logging
from sklearn.tree import export_text
from datetime import datetime


#오늘 날짜 지정
today = datetime.today().strftime('%Y%m%d')

log_dir = '/dshome/ddualab/yohan/finance/log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_filename = os.path.splitext(os.path.basename(__file__))[0] + f'{today}.log'
logging.basicConfig(
    filename=os.path.join(log_dir, log_filename),
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)



le = LabelEncoder()

# 데이터 샘플링 및 train/val/test 나누기
def preprocessing_df(path, pran_name):
    df = pd.read_csv(path)
    df = df[['원화환산금액', '시스템 등록일시', '가맹점아이디', '고객 카드번호 암호화값','label', 'new_업태']]
    df['시스템 등록일시'] = pd.to_datetime(df['시스템 등록일시'])
    df['시스템 등록일시'] = df['시스템 등록일시'].astype(np.int64) // 10**9
    df['new_업태'] = le.fit_transform(df['new_업태'])
    df['고객 카드번호 암호화값'] = le.fit_transform(df['고객 카드번호 암호화값'])
    
    df_norm = df[df['label'] == 0]
    
    df_fds = df[df['label'] == 1]
    df_fds = df_fds[df_fds['가맹점아이디'] == pran_name]
    
    # 50% 랜덤 샘플링 (seed = 42)
    df_sampled = df_norm.sample(n=int(len(df_norm)*0.5), random_state=42)
    df_sampled = df_sampled[df_sampled['가맹점아이디'] == pran_name]
    
    # df = pd.concat([df_sampled, df_fds], axis=0).reset_index(drop=True)
    
    # 3:1:1 분할 (train:val:test)
    df_train_0, df_temp_0 = train_test_split(df_sampled, test_size=0.4, random_state=42)
    df_val_0, df_test_0 = train_test_split(df_temp_0, test_size=0.5, random_state=42)
    
    df_train_1, df_temp_1 = train_test_split(df_fds, test_size=0.4, random_state=42)
    df_val_1, df_test_1 = train_test_split(df_temp_1, test_size=0.5, random_state=42)
    
    # 각 라벨 데이터를 합침
    df_train = pd.concat([df_train_0, df_train_1])
    df_train = df_train[['원화환산금액', '시스템 등록일시', 'new_업태', '고객 카드번호 암호화값','label']]
    df_val = pd.concat([df_val_0, df_val_1])
    df_val = df_val[['원화환산금액', '시스템 등록일시', 'new_업태', '고객 카드번호 암호화값','label']]
    df_test = pd.concat([df_test_0, df_test_1])
    df_test = df_test[['원화환산금액', '시스템 등록일시', 'new_업태', '고객 카드번호 암호화값','label']]
    
    X_train = df_train.drop(columns=['label'])
    y_train = df_train['label']
    X_val = df_val.drop(columns=['label'])
    y_val = df_val['label']
    X_test = df_test.drop(columns=['label'])
    y_test = df_test['label']
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# 그리드서치 및 평가 함수
def grid_search_and_evaluate(models, param_grids, X_train, y_train, X_test, y_test, pran_name):
    for model_name, model in models.items():
        print(f"Grid search for {model_name}")
        logging.info(f"Grid search for {model_name}_{pran_name}")
        
        # GridSearchCV 설정
        grid_search = GridSearchCV(model, param_grids[model_name], cv=3, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        # 최적의 모델 및 하이퍼파라미터 출력
        print(f"Best parameters for {model_name}: {grid_search.best_params_}")
        logging.info(f"{pran_name}_Best parameters for {model_name}: {grid_search.best_params_}")
        best_model = grid_search.best_estimator_
        # 트리 룰 저장
        if model_name == 'Random Forest': 
            # 트리 규칙을 텍스트로 저장
            tree_rules = export_text(best_model.estimators_[0], feature_names=list(X_train.columns))
            
            # 텍스트 파일로 저장
            if pran_name != '/KMJNdw=':
                with open(f'/dshome/ddualab/yohan/finance/result/가맹점명별/{model_name}_rule_tree_{pran_name}_{today}.txt', 'w') as f:
                    f.write(tree_rules)
            else:
                pran_name = 'KMJNdw='
                with open(f'/dshome/ddualab/yohan/finance/result/가맹점명별/{model_name}_rule_tree_{pran_name}_{today}.txt', 'w') as f:
                    f.write(tree_rules)
                    
        # 최적 모델로 테스트셋 평가
        threshold = [0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15 ,0.1, 0.05]
        
        for thrsh in threshold:
            y_pred = best_model.predict(X_test)
            # print(y_pred[:100])
            y_pred_prob = best_model.predict_proba(X_test)[:, 1]
            # print(y_pred_prob[:100])
            y_pred = (y_pred_prob >= thrsh).astype(int)
            # print(y_pred[:100])
            
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            # 혼동 행렬 및 성능 지표 출력
            plt.figure(figsize=(6, 4))
            sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', xticklabels=[0, 1], yticklabels=[0, 1])
            plt.title(f'Confusion Matrix: {model_name}')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            if pran_name != '/KMJNdw=':
                plt.savefig(f'/dshome/ddualab/yohan/finance/result/가맹점명별/{model_name}_confusion_matrix_thrsh_{thrsh}_{pran_name}_{today}.png')
            elif pran_name == '/KMJNdw=':
                pran_name = 'KMJNdw='
                plt.savefig(f'/dshome/ddualab/yohan/finance/result/가맹점명별/{model_name}_confusion_matrix_thrsh_{thrsh}_{pran_name}_{today}.png')
            plt.close()

            # 성능 지표 출력
            print(f"{model_name}_threshold_{thrsh}_pranchise_{pran_name} - Accuracy: {accuracy_score(y_test, y_pred)}, F1-Score: {f1_score(y_test, y_pred)}")
            logging.info(f"{model_name}_threshold_{thrsh}_pranchise_{pran_name}  - Accuracy: {accuracy_score(y_test, y_pred)}, F1-Score: {f1_score(y_test, y_pred)}")

# 모델 및 하이퍼파라미터 그리드 설정
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
}

param_grids = {
    'Random Forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 3, 4],
    },
    'XGBoost': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'learning_rate': [0.01, 0.1, 0.2]
    },
    'Logistic Regression': {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l2']
    }
}

available_name = ['nfgZ69lWmQ==','di7FqMsiZ4d+0dQ=','/KMJNdw=','lrWX+FgZzQ==']
# 데이터 전처리 및 모델 학습

for name in available_name:
    # X_train, X_val, X_test, y_train, y_val, y_test = preprocessing_df('/dshome/ddualab/yohan/finance/data_csv/total_df_align_KRW_1021.csv', pran_name=name)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessing_df('/dshome/ddualab/yohan/finance/공유/last_total.csv', pran_name=name)
    # print(X_train)
    grid_search_and_evaluate(models, param_grids, X_train, y_train, X_test, y_test, pran_name=name)
