import cv2
import numpy as np
from tensorflow import keras
import os

# Pycharm에서 TensorFlow 쓰려면 꼭 이거대로 깔아야 합니닷
# https://flowingtime.tistory.com/entry/PyCharm%EC%97%90%EC%84%9C-TensorFlow-%EC%9D%B4%EC%9A%A9%EC%9D%84-%EC%9C%84%ED%95%9C-%EA%B0%9C%EB%B0%9C%ED%99%98%EA%B2%BD-%EC%84%A4%EC%A0%95%EC%9C%88%EB%8F%84%EC%9A%B010



def classify(folder):
    # 캡쳐본이 input.png로 프로젝트 파일 안에 저장된다고 가정
    # 바로 밑 두 줄이 전처리. 수정될 수도 있음
    images=[]
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename),cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)

    kernel_size = 3
    Laplacian = cv2.Laplacian(images[0], -1, ksize=kernel_size)
    cv2.imwrite("D:/PyCharmProject/input/laplacian.png", Laplacian)

    # 모델 입력 형태로 변환
    X = np.array([Laplacian])

    # 프로젝트 파일 안에 'my_model'파일 안에서 미리 저장해둔 모델 가져옴
    model = keras.models.load_model('D:/PyCharmProject/trained_model/my_model')

    # 분류 결과 확인
    class_names = ['rock', 'scissors', 'paper']
    print(np.argmax(model.predict(X)))
    print(class_names[np.argmax(model.predict(X))])

    return class_names[np.argmax(model.predict(X))]
"""
if __name__ == '__main__':
    classify('./input')
"""