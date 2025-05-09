# 비정상 거래 탐지 모델 개발

이 프로젝트는 금융 거래 데이터에서 비정상 거래(매입 취소)를 탐지하기 위한 머신러닝 모델을 개발합니다.

## 파일 설명

1. `preprocessing.ipynb`: 데이터 전처리 과정을 담은 Jupyter Notebook
2. `main_private.py`: 개인 거래 별 학습 및 평가를 위한 메인 실행 파일
3. `main_each_pran_ml.py`: 업태 내 거래 별 학습 및 평가를 위한 실행 파일
4. `main_each_name_ml.py`: 가맹점 내 거래 별 학습 및 평가를 위한 실행 파일

## 목차

1. 데이터 전처리
2. 대표 업태 및 가맹점 아이디 선정
3. 시각화
4. 매입 취소 탐지 모델
5. 결론

## 1. 데이터 전처리

### 1.1. 데이터 형태
- 총 데이터: 3,785,142건
- 총 변수: 27가지
```text
["거래번호", "가맹점아이디", "업태", "업종", "사업자구분", "사업자번호", "법인등록번호",
 "매입사", "결제통화", "금액", "결제사", "고객 카드 BIN", "고객 카드번호 암호화값",
 "고객 메일", "고객 이름", "고객 상품명", "구매 상품금액", "구매 수량", "등록일",
 "시스템 등록일시", "결제수단", "결제유형", "거래상태", "가맹점 지역", "승인 지역",
 "연령층", "영업시간"]

```
- 매입취소(거래상태): 20,888건
### 1.2. 전처리 과정
1. 개인정보 삭제
2. 업종 & 업태 NAN 처리
3. 업종/업태 정보 상호 보완
4. 가맹점 지역 주소 기준 좌표값(lat, long) 추가
5. '거래상태'열 '승인' 및 '승인취소' 삭제
6. 정상/비정상 라벨링
7. 업태 통일화를 위한 수기 재분류
8. 결제 통화 통일화(원화 환산)

### 1.3. 전처리 후 데이터

- 총 데이터: 2,415,101건  
- 총 변수: 27가지  




## 2. 대표 업태 및 가맹점 아이디 선정
업태와 가맹점별 거래 데이터는 **각각 고유의 패턴과 이상 거래 특성**을 가질 수 있음
⇒ 이들을 그룹화한 모델을 따로 개발하여 비교 및 분석
### 2.1. 업태 선정
- 선정된 업태 중 **비정상 데이터가 존재하는 업태만** 추출
    - **비정상 데이터가 존재하지 않을 경우 모델학습 불가**
- 선정 업태(정상, 비정상)
  - 도소매 (1,330,180 정상 / 3,453 비정상)
  - 서비스 (842,024 / 2,331)
  - 통신업 (83,974 / 29)
  - General e_comm (3,136 / 21)

### 2.2. 가맹점 아이디 선정
- 전처리 과정 이후 총 163가지의 가맹점 아이디 존재
- 가맹점 아이디 중 정상 / 비정상이 포함된 상위 4개 선정
  - **비정상 데이터가 존재하지 않을 경우 모델학습 불가**
- 선정 가맹점 아이디 (정상, 비정상)
  - nfgZ69lWmQ== (658,573 / 98)
  - di7FqMsiZ4d+0dQ= (494,529 / 498)
  - /KMJNdw= (183,608 / 620)
  - lrWX+FgZzQ== (121,984 / 687)

## 3. 시각화

금액 분포를 확인하기 위해 개별 거래, 업태 그룹별, 가맹점 그룹별 시각화 수행 (amount < 90,000 필터 적용)
### 3.1. 개별 거래 별 시각화
- 금액 분포 확인
![개별 거래 별 시각화](https://github.com/KimYohan0317/Finance_AD/blob/main/image/%EA%B0%9C%EB%B3%84%20%EA%B1%B0%EB%9E%98%20%EB%B3%84%20%EC%8B%9C%EA%B0%81%ED%99%94.png?raw=true)
### 3.2. (업태로 그룹핑 후 각 그룹 내) 개별 거래 별 시각화
- 금액 분포 확인
 -  도소매
   ![도소매](https://github.com/KimYohan0317/Finance_AD/blob/main/image/%EB%8F%84%EC%86%8C%EB%A7%A4.png)
 -  서비스  
   ![서비스](https://github.com/KimYohan0317/Finance_AD/blob/main/image/%EC%84%9C%EB%B9%84%EC%8A%A4.png?raw=true)

 -  통신업  
   ![통신업](https://github.com/KimYohan0317/Finance_AD/blob/main/image/%ED%86%B5%EC%8B%A0%EC%97%85.png?raw=true)

 -  General_e_comm  
   ![General_e_comm](https://github.com/KimYohan0317/Finance_AD/blob/main/image/General_e_comm.png?raw=true)
### 3.3. (가맹점으로 그룹핑 후 각 그룹 내) 개별 거래 별 시각화
- 금액 분포 확인
 -  nfgZ69lWmQ==  
   ![nfgZ69lWmQ==](https://github.com/KimYohan0317/Finance_AD/blob/main/image/nfgZ69lWmQ==.png?raw=true)

 -  di7FqMsiZ4d+0dQ=  
   ![di7FqMsiZ4d+0dQ=](https://github.com/KimYohan0317/Finance_AD/blob/main/image/di7FqMsiZ4d+0dQ=.png?raw=true)

 -  /KMJNdw=  
   ![/KMJNdw=](https://github.com/KimYohan0317/Finance_AD/blob/main/image/KMJNdw%3D.png?raw=true)

 -  lrWX+FgZzQ==  
   ![lrWX+FgZzQ==](https://github.com/KimYohan0317/Finance_AD/blob/main/image/lrWX%2BFgZzQ%3D%3D.png?raw=true)
## 4. 매입 취소 탐지 모델

### 4.1. 분석 모델 선정
- Random Forest
  - **모델 설명**: 랜덤 포레스트는 여러 개의 결정 트리를 앙상블로 묶어 분류를 수행하는 알고리즘, 각 트리는 랜덤 샘플링과 특징 선택을 통해 학습되며, 최종으로 여러 트리의 결과를 종합하여 결정
  - 선정 이유
      1. **비선형 데이터 처리**: 랜덤 포레스트는 변수 간의 비선형 관계를 잘 처리 할 수 있기 때문에, 복잡한 금융 거래 패턴을 분석하는 데 용이함
      2. **과적합 방지**: 여러 트리를 앙상블로 묶기 때문에 많은 변수를 가진 거래 데이터가 모델이 특정 패턴에 과도하게 맞춰지는 현상 방지 가능
- XGBoost
  - **모델  설명**
      - 부스팅 기법의 일종으로, 이전 모델이 잘못 예측한 샘플에 가중치를 두어 새로운 모델을 학습하는 방식(여러 약한 모델을 결합하여 강력한 학습 모델 구축)
  - 선정 이유
      1. **고성능 및 효율성**: XGBoost는 속도 성능 면에서 뛰어나며 대규모 데이터셋에서 빠른 학습과 분류가 가능, 2백만 건 이상의 거래 데이터를 빠르게 처리하고, 최적화된 성능 도출 가능
      2. **불균형 데이터 처리**: 비정상 거래건 수가 굉장히 불균형한 특징을 가지고 있는데, XGBoost는 learning rate, max depth, weight normalization 등 하이퍼파라미터 조절을 통해 과적합 방지에 유리
### 4.2. 데이터 샘플링
- **4.2.1.샘플링 이유**
    - 정상 거래건 2,409,266, 비정상 거래건 5,835개로 **비정상 데이터가 매우 적은 불균형 문제**를 가지고 있음
    - 정상 데이터의 과도한 양으로 인해 모델이 비정상 데이터의 특성을 간과하고, 정상 데이터의 특성에 치우친 학습할 가능성 존재
    - 이를 방지하기 위해 정상 데이터에서 50% 랜덤 샘플링하여 학습에 사용함으로써, 데이터 불균형 완화
- **4.2.2 샘플링 과정**
    - 정상 거래 건의 50%를 랜덤 샘플링 하였으며, 비정상 거래 건은 샘플링하지 않고 그대로 사용
    - 따라서 레이블의 비율이 약 400:1에서 200:1로 변경 되었음
    - 이후 데이터를 train: validation: test = 3: 1: 1 비율로 나누어서 실험
        1. 정상 거래 50% 랜덤 샘플링
        2. 정상 거래 3 : 1 : 1로 분리
        3. 비정상 거래 3 : 1 :1로 분리
        4. 정상 및 비정상 거래 통합
### 4.3. 최종 feature 선정
- 본 프로젝트에서는 대규모 금융 거래 데이터에서 비정상 거래를 탐지하기 위한 머신러닝 모델 개발을 위해 학습에 사용할 feature를 선정
- 다양한 feature 중 모델 성능과 해석 가능성 측면에서 중요한 몇가지 핵심 feature를 선정
    - '원화환산금액'
        - 거래 금액은 금융 거래에서 가장 기본적인 특징으로, 비정상 거래가 발생할 가능성이 높은 패턴을 포착하는 데 중요한 역할을 할 가능성이 있음
    - '고객 카드번호 암호화값'
        - 비정상 거래가 특정 고객에게 집중될 가능성이 높기 때문에, 비정상 패턴을 탐지하는 데 중요한 feature가 될 수 있음
        - 같은 고객이 반복적인 비정상 거래가 발생한다면, 이 변수는 모델이 효과적으로 감지할 수 있음
    - '가맹점아이디'
        - 가맹점 정보는 거래가 이루어진 지점을 나타내며, 특정 가맹점에서 발생하는 거래 패턴을 분석하는 역할을 함
    - 'new_업태’
        - 업태 별로 정상, 비정상 거래의 패턴이 상이할 가능성이 있기 때문에 선정
    - '시스템 등록일시'
        - 비정상 거래의 경우 특정 시간대에 집중되거나, 통상적인 거래 시간대와는 다른 시간에 이루어지는 경우가 많아, 비정상 거래 탐지에 유용할 것
    - 'label'
        - 정상, 비정상 거래를 나타내는 이진 변수로, 모델 학습을 위한 타겟 변수임

### 4.4. 실험 결과
- **개별 거래 분류 모델**: Random Forest (Accuracy: 0.994, F1-score: 0.1865)  
  ![개별 거래 분류 모델 결과](https://github.com/KimYohan0317/Finance_AD/blob/main/image/%EA%B0%9C%EB%B3%84%20%EA%B1%B0%EB%9E%98%20%EB%B6%84%EB%A5%98%20%EB%AA%A8%EB%8D%B8%20%EC%8B%A4%ED%97%98%20%EA%B2%B0%EA%B3%BC.png?raw=true)

- **업태 그룹별 모델**: 도소매 (Random Forest Accuracy: 0.9916, F1-score: 0.1382)  
  ![업태 그룹별 모델 결과](https://github.com/KimYohan0317/Finance_AD/blob/main/image/%EC%97%85%ED%83%9C%20%EA%B7%B8%EB%A3%B9%EB%B3%84%20%EB%AA%A8%EB%8D%B8%20%EC%8B%A4%ED%97%98%20%EA%B2%B0%EA%B3%BC.png?raw=true)

- **가맹점 그룹별 모델**: di7FqMsiZ4d+0dQ= (Random Forest Accuracy: 0.9978, F1-score: 0.2128)  
  ![가맹점 그룹별 모델 결과](https://github.com/KimYohan0317/Finance_AD/blob/main/image/%EA%B0%80%EB%A7%B9%EC%A0%90%20%EA%B7%B8%EB%A3%B9%EB%B3%84%20%EB%AA%A8%EB%8D%B8%20%EC%8B%A4%ED%97%98%20%EA%B2%B0%EA%B3%BC.png?raw=true)

## 5. 결론
- Random Forest가 가장 좋은 성능을 보임
- 데이터 불균형이 큰 상황에서도 높은 정확도 유지
- 비정상 거래 탐지 성능 향상을 위해 추가적인 feature engineering과 데이터 증강 필요
- 실제 Fraud 거래에 대한 명확한 label이 있다면 더 정확한 FDS 모델 개발 가능

## 실행 방법

1. 전처리:
```bash
jupyter notebook preprocessing.ipynb
```
2. 모델 실행:
```bash
# 개인 거래 별 모델
python main_private.py

# 업태 별 모델
python main_each_pran_ml.py

# 가맹점 별 모델
python main_each_name_ml.py
```
