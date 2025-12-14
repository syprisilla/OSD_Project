# OSD_Project

충북대학교 오픈소스개발프로젝트 기말 과제로 진행한  
**미국 교통사고 데이터 기반 날씨별 사고 분석 및 사고 위험도 예측 프로젝트**이다.

---

## 프로젝트 개요

본 프로젝트는 미국 전역에서 발생한 교통사고 데이터를 활용하여  
날씨 조건에 따른 사고 발생 특성을 분석하고,  
기상 변수 기반으로 **사고 위험도가 높은 날을 예측**하는 것을 목표로 한다.

---

## 사용 데이터셋

- **데이터셋 이름**: US Accidents (March 2023)
- **출처**: Kaggle  
  https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents

---

## 사용 기술 스택

- **Python**
- **Pandas / NumPy** : 데이터 처리 및 분석
- **Matplotlib / Seaborn** : 데이터 시각화
- **Scikit-learn** : 데이터 분할 및 사고 위험도 예측 모델
- **Git / GitHub** : 형상 관리 및 오픈소스 프로젝트 관리

---

## 실행 방법

1. Kaggle에서 데이터 다운로드
2. `data/` 폴더에 CSV 파일 저장
3. 아래 명령 실행

```bash
python analysis.py
```
