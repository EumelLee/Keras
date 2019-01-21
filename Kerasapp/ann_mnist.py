##############################################
# Modeling
##############################################
from keras import layers, models

#기본 형태 분류 ANN 구성
def ANN_models_func(Nin, Nh, Nout):
    x = layers.Input(shape=(Nin,)) #입력계층
    h = layers.Activation('relu')(layers.Dense(Nh)(x)) #은닉계층
    y = layers.Activation('softmax')(layers.Dense(Nout)(h)) # 출력계층
    model = models.Model(x, y)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


#연쇄방식으로 구현


def ANN_seq_func(Nin, Nh, Nout):
    model = models.Sequential(). #분산과 다르게 모델을 먼저 설정
    #연쇄방식은 모델 구조를 정의하기 전에 Sequential()로 모델을 먼저 정의해야 함
    
    #모델 구조 설정
    model.add(layers.Dense(Nh, activation='relu', input_shape=(Nin,))) #입력계층과 은닉계층의 형태 동시에 정해짐
    #입력 노드 Nin개는 완전 연결 계층 Nh개로 구성된 은닉 계층으로 보내짐
    
    #은닉계층의 출력은 출력이 Nout개인 출력 노드로 보내짐
    #출력 노드의 활성화 함수는 softmax 연산
    model.add(layers.Dense(Nout, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


#분산방식 모델링을 포함하는 객체지향형 구현
class ANN_models_class(models.Model): #models.Model로부터 특성을 상속해온다. 신경망에서 사용하는 학습, 예측, 평가와 같은 다양한 함수 제공
    def __init__(self, Nin, Nh, Nout):
        # Prepare network layers and activate functions
        
        #은닉계층이 하나이므로 출력변수로 hidden 하나만 사용
        hidden = layers.Dense(Nh)
        
        #노드 수 Nout개인 출력 계층 정의
        output = layers.Dense(Nout)
        
        #비선형성을 넣어주는 activation 함수 정의
        relu = layers.Activation('relu') #relu 0보다 큰수는 그대로 출력 작으면 0으로 출력
        softmax = layers.Activation('softmax')
        
        # Connect network elements
        x = layers.Input(shape=(Nin,))
        h = relu(hidden(x))
        y = softmax(output(h))
        
        #상속받은 부모의 클래스 초기화
        super().__init__(x, y)
        self.compile(loss='categorical_crossentropy',
                     optimizer='adam', metrics=['accuracy'])

#케라스의 장점! 만들어진 모델을 불러와서 사용하면 되기 때문에 복잡한 AI수식을 일일이 파악할 수고가 줄어든다


#연쇄 방식 모델링을 포함하는 객체지향형 구현
#신경망 모델이 연속적인 하나의 고리로 연결되어 있다는 가정 하에 모델링이 이루어진다
class ANN_seq_class(models.Sequential):
    def __init__(self, Nin, Nh, Nout):
        
        #직접 모델링은 컴파일 하기 직전에 초기화 했지만, 연쇄방식에서는 부모 클래스의 초기화 함수를 자기 초기화 함수 가장 앞단에서 부른다
        super().__init__()
        
        #입력계층을 별도로 정의하지 않고 은닉 계층부터 추가해 나간다
        self.add(layers.Dense(Nh, activation='relu', input_shape=(Nin,))) #은닉 계층 붙일 때 변수 중 하나로 입력 계층의 노드 수를 포함해주어 간단히 입력계층을 정의함, 활성화 함수도 어떤걸 사용할지 아규먼트로 지정함
        
        #출력계층 지정
        self.add(layers.Dense(Nout, activation='softmax'))
        self.compile(loss='categorical_crossentropy',
                     optimizer='adam', metrics=['accuracy'])


##############################################
# Data
#인공지능으로 처리할 데이터를 불러온다
##############################################
import numpy as np
from keras import datasets  # mnist
from keras.utils import np_utils  # to_categorical


def Data_func():
    
    #X_와 y_를 이니셜로하는 입력과 출력 변수에 각각 저장한다.
    #학습에 사용되는것은 _train, 성능 평가에 사용하는 데이터는 _test로 끝나는 두 변수에 저장한다.
    (X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()
    
    
    #0~9까지 숫자로 구성도니 출력값을 0과 1로 표현되는 벡터 10개로 바꾼다
    #ANN을 이용한 분류작업 시 정수보다 이진 벡터로 출력 변수를 구성하는 것이 효율적이기 때문이다.
    Y_train = np_utils.to_categorical(y_train)
    Y_test = np_utils.to_categorical(y_test)
    
    L, W, H = X_train.shape
    
    #X_train, X_test 에 (x,y)축에 대한 픽셀 정보가 들어있는 3차원 데이터인 실제 학습 및 평가용 이미지를 2차원으로 조정
    #학습 데이터 샘플이 L개 > L * W * H 와 같은 모양의 텐서로 저장되어 있음
    #멤버변수 shape에는 2D이미지 데이터를 저장하는 저장소의 규격이 들어있다. 이를 ANN으로 학습하려면 벡터 이미지 형태로 바꾸어야 한다
    #바꾸는데 reshape() 멤버함수를 사용한다
    X_train = X_train.reshape(-1, W * H) #첫번째 -1은 행렬의 행을 자동으로 설정하게 만든다
    X_test = X_test.reshape(-1, W * H)
    
    #ANN의 최적화를 위해 아규먼트 정규화 0~255사이 정수로 구성된 입력값을 255로 나누어 0~1사이 실수로 바꾸어준다.
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    
    return (X_train, Y_train), (X_test, Y_test)


##############################################
# Plotting
##############################################
import matplotlib.pyplot as plt


def plot_acc(history, title=None):
    # summarize history for accuracy
    if not isinstance(history, dict):
        history = history.history
    
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    if title is not None:
        plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Verification'], loc=0)
# plt.show()


def plot_loss(history, title=None):
    # summarize history for loss
    if not isinstance(history, dict):
        history = history.history
    
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    if title is not None:
        plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Verification'], loc=0)
# plt.show()


##############################################
# Main
##############################################

#ANN에 사용할 파라미터 정의
def main():
    Nin = 784 #입력 Nin : 길이가 784인 데이터
    Nh = 100 #은닉계층의 노드 수
    number_of_class = 10
    Nout = number_of_class #출력 노드 수는 분류 할 데이터의 클래스 수와 같다.
    
    # model = ANN_models_func(Nin, Nh, Nout)
    # model = ANN_models_class(Nin, Nh, Nout)
    model = ANN_seq_class(Nin, Nh, Nout) #앞서 만들었던 모델의 인스턴스를 만들고 데이터도 불러온다
    (X_train, Y_train), (X_test, Y_test) = Data_func()
    
    ##############################################
    # Training
    ##############################################
    
    #fit()이용해 학습
    history = model.fit(X_train, Y_train, epochs=15, batch_size=100, validation_split=0.2)
    #batch_size 데이터를 얼마나 나누어서 넣을 것인지 / validation_split은 전체 학습 데이터 중에서 학습 진행 중 성능 검증에 데이터를 얼마나 사용할 것인지 즉 20%를 활용한다
    
    #학습이나 검증에 사용되지 않은 데이터로 성능을 최종 평가한 결과
    performace_test = model.evaluate(X_test, Y_test, batch_size=100)
    print('Test Loss and Accuracy ->', performace_test)
    
    plot_loss(history)
    plt.show()
    plot_acc(history)
    plt.show()


# Run code
if __name__ == '__main__':
    main()
