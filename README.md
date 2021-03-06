# 늦게 내는 가위바위보
## 팀 당근당근
- 201811149 박슬비
- 201811159 차형준
- 201811160 최보원
---
## 유투브 영상 링크
<img width="540" alt="carrotcarrot_thumnail_2" src="https://user-images.githubusercontent.com/56291868/101260447-48f95a80-3773-11eb-8472-07bea1008843.png">
https://youtu.be/JuCXrMKks2A

---
## 작품의 필요성
- 팀 당근당근에서 만들고자 하는 게임은 두뇌 트레이닝 훈련 게임으로 인지능력 저하를 막고, 치매 예방, 유아 두뇌 발달 등의 효과를 기대하고 제작하고 있다.

- 현재 대한민국은 고령화 사회로 진입하였으며, 65세 이상 고령 인구가 증가함에 따라 치매 환자의 수도 함께 증가하고 있다.
- ‘중앙 치매센터(센터장 김기웅)가 발간한 ‘대한민국 치매현황 2018’ 보고서에 따르면, 65세 이상 치매 환자는 70만 5473명으로 추정되며 치매 유병률은 10%로 나타났다. 이와 함께 치매 환자 수는 지속적으로 증가해 2024년에는 100만명, 2039년에는 200만명, 2050년에는 300만명을 넘어설 것으로 예상됐다.
- (‘ 박선혜, “국내 65세 이상 노인 10명 중 1명 치매”, MEDICAL Observer, https://www.monews.co.kr/news/articleView.html?idxno=201496)

- 치매를 예방하기 위해서 화투를 친다는 말이 있듯, 인지능력 저하를 막기 위한 방법으로 두뇌 트레이닝 훈련 게임을 통해서 치매를 예방할 수 있다고 생각한다.
- ‘연구에 따르면, 뇌 훈련 게임이 기억력을 강화하고 치매 발병 위험을 낮출 수 있는 것으로 나타났다. 캠브리지 대학의 연구진은 비디오 게임이 알츠하이머 전조 증상으로 볼 수 있는 기억력 조기 감소 문제를 보인 뇌 기능을 개선할 수 있는 것을 발견했다.’
- (기억력을 증진하고 치매를 예방하는 뇌 훈련 게임, Barbara Sahakian, Cambridge University, 
http://www.telegraph.co.uk/science/2017/07/02/brain-training-games-boost-memory-may-reduce-risk-dementia-research/)
---
## 작품 설명
- 컴퓨터와 늦게 내는 가위바위보를 하는 프로그램. 늦게 내는 가위바위보는, 상대방이 어떤 수를 냈는지와 승리 조건을 보고 제한된 시간 안에 승리 조건에 만족하는 수를 제시하는 게임이다.

- 게임의 진행은 세 단계로 진행된다.
1. AI는 화면에 가위, 바위, 보, 셋 중 한가지를 제시한다. 
2. 이기세요, 비기세요, 지세요, 셋 중 한가지의 승리 조건을 같이 제시한다.
3. 사용자는 화면에 표시된 컴퓨터의 수와 승리 조건을 보고 제한된 시간(3초) 안에 승리하는 수를 웹캠을 통해 입력해야한다.

- 내부적으로는 제시된 시간을 카운트하여, 0초가 될 때 웹캠 화면을 캡처한다. 캡처한 이미지는 전처리하여 분류 모델에 입력, 모델은 학습한 이미지를 바탕으로 분류한 결과 값을 출력하게 된다. 컴퓨터가 제시한 수와 조건, 그리고 AI가 분류한 사용자의 수를 바탕으로 사용자의 승리 여부를 판단하여 화면에 출력한다.
## AI 기술 설명 
- 이 게임에는 이미지를 입력받았을 때 가위바위보로 분류하는 AI를 사용할 것이다. 학습에 사용될 가위바위보 이미지는 Kaggle에서 PNG형식의 dataset을 수집하였고 Python의 OpenCV 라이브러리를 활용하여 이미지를 흑백으로 불러오고 Laplacian 필터를 씌워 경계선 검출 방법으로 모든 이미지를 전처리 하였다. 

- AI모델에 입력되는 이미지는 PC 웹캠으로 받고 CNN(인공신경망)방식으로 분류기를 학습시켰다.

- 전처리와 학습은 Google Colab으로 하였으며 그 외의 Python 코드는 Pycharm으로 작성하였다.
---
## 기대 효과
- 인지능력 훈련을 통해서 노화로 인한 뇌 기능 감소를 예방하는 것은 가능하다. 하지만 현대인들에게는 시간적인 여유와 공간적인 제약을 받기 때문에 인지능력 훈련을 하는 것은 거의 불가능하다. 인지 능력 훈련 프로그램들은 지루하고, 많은 시간을 필요로 하기 때문이다. (양영욱, 임희석 (2011). 스마트디바이스를 활용한 인지 능력 훈련 기능성 게임 개발. 한국게임학회 논문지, 11(6), 23-31)

- 대한민국의 스마트폰 사용률은 60대가 넘어가는 노령인구도 남성 85%, 여성 72%를 기록할 만큼 높은 편이다. 
- (한국갤럽조사연구소, 2012-2020 스마트폰 사용률 & 브랜드, 스마트워치, 무선이어폰에 대한 조사,
https://www.gallup.co.kr/gallupdb/reportContent.asp?seqNo=1134)

- 팀 당근당근이 제작하고 있는 게임은 카메라와 디스플레이만 있다면 가볍게 즐길 수 있는 게임이다. 초기 개발 버전은 PC와 웹캠을 이용하여 플레이 할 수 있지만, 모바일 기기로의 이식도 충분히 가능한 게임이다. 스마트폰의 카메라를 이용하여 기능성 게임을 만들어서 보급한다면 언제 어디서나 가볍게 즐길 수 있어 긍정적인 효과를 불러 일으킬 것이라 생각된다.
---
