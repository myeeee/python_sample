import numpy as np


def clprob(y):
    if y.shape[0] == 0:
        return np.zeros((y.shape[1]))

    # クラス数のカウント
    y_class = y.argmax(axis=1)
    uclass, ucount = np.unique(y_class, return_counts=True)

    prob = np.zeros(y.shape[1])
    prob[uclass] = ucount

    return prob / y.shape[0]


def infgain(ly, ry):
    size = [ly.shape[0], ry.shape[0]]
    if 0 in size:
        return np.inf

    # クラス確率計算
    total_size = sum(size)
    prob = [clprob(ly), clprob(ry)]

    # loss算出
    entropy = 0
    for i, p in enumerate(prob):
        entropy += size[i] * sum(p[p != 0.0] * np.log2(p[p != 0.0]))

    return - entropy / total_size


def gini(ly, ry):
    size = [ly.shape[0], ry.shape[0]]
    if 0 in size:
        return np.inf

    # クラス確率計算
    total_size = sum(size)
    prob = [clprob(ly), clprob(ry)]

    # loss算出
    entropy = 0
    for i, p in enumerate(prob):
        entropy += size[i] * sum(p[p != 0.0] ** 2)

    return 1.0 - entropy / total_size


class ZeroRule:
    def __init__(self):
        self.mean = None
        self.clnum = 0

    def fit(self, train_X, train_Y):
        self.mean = np.mean(train_Y, axis=0)
        self.clnum = train_Y.shape[1]

    def predict(self, test_X):
        z = np.zeros((test_X.shape[0], self.clnum))

        z += self.mean
        return z


class DecisionStump:

    def __init__(self, classifier, metrix=infgain):
        self.left = classifier
        self.right = classifier
        self.metrix = metrix
        self.feat = None
        self.val = None

    def fit(self, train_X, train_Y):
        # 各次元のデータをsort
        xindex = np.argsort(train_X, axis=0)
        ysort = np.take(train_Y, xindex, axis=0)

        # 初期化
        score = np.inf
        leftX = None
        rightX = None
        leftY = None
        rightY = None
        for i in range(1, train_X.shape[0]):
            ly = ysort[:i]
            ry = ysort[i:]
            loss_val = np.array([self.metrix(ly[:, dim], ry[:, dim]) for dim in range(train_X.shape[1])])
            if True in (loss_val < score):
                self.feat = np.argmin(loss_val)
                self.val = train_X[xindex[i, self.feat], self.feat]
                score = np.min(loss_val)
                leftX = np.take(train_X, xindex[:i, self.feat], axis=0)
                rightX = np.take(train_X, xindex[i:, self.feat], axis=0)
                leftY = ly[:, self.feat]
                rightY = ry[:, self.feat]

        self.left.fit(leftX, leftY)
        self.right.fit(rightX, rightY)

    def makeSplit(self, data_X):
        lid = np.where(data_X[:, self.feat] < self.val)
        rid = np.where(data_X[:, self.feat] >= self.val)
        return lid, rid

    def predict(self, test_X):
        # 分割
        lid, rid = self.makeSplit(test_X)

        lz = self.left.predict(test_X[lid])
        rz = self.right.predict(test_X[rid])

        z = np.empty((test_X.shape[0], lz.shape[1]))
        z[lid] = lz
        z[rid] = rz

        return z


class DecisionTree(DecisionStump):

    def __init__(self, classifier, metrix=infgain, depth=1, max_depth=3):
        self.depth = depth
        self.max_depth = max_depth

        if depth + 1 < max_depth:
            clssfr = DecisionTree(classifier, metrix, depth+1, max_depth)
        else:
            clssfr = classifier

        super().__init__(clssfr, metrix)


if __name__ == '__main__':
    import pandas as pd
    from sklearn.model_selection import train_test_split

    pd.set_option('display.max_columns', None)

    #data_airfoil = pd.read_table('../airfoil_self_noise.dat', header=None, names=['Frequency', 'AttackAngle', 'ChordLength', 'StreamVelocity', 'DisplacementThickness', 'PressureLevel'])
    data_iris = pd.read_csv('../iris.data', header=None, names=['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Class'])

    # クラスラベルをindexに変換
    data_iris['Class'] = data_iris['Class'].apply(sorted(list(set(data_iris['Class']))).index)

    #data_airfoil = data_airfoil.values
    data_iris = data_iris.values

    train_data, test_data = train_test_split(data_iris, train_size=0.6, test_size=0.4, random_state=1, shuffle=True)
    train_X = train_data[:, :-1]
    train_Y = train_data[:, -1]
    test_X = test_data[:, :-1]
    test_Y = test_data[:, -1]

    u = np.unique(train_Y)
    prob_Y = np.zeros((train_Y.shape[0], u.shape[0]))
    for i in range(train_Y.shape[0]):
        prob_Y[i, int(train_Y[i])] = 1.0

    # テストケース作成
    '''
    x = np.array([[1, 5],
                  [3, 4],
                  [2, 6],
                  [2, 6]])
    y = np.array([[1.0, 0.5, 0],
                  [1.0, 1.7, 0.4],
                  [0.2, 0.1, 2],
                  [2, 2, 4]])
    '''

    #clssfr = DecisionStump()
    #clssfr.fitSplit(x, y, loss='infgain')
    #exit()

    model = DecisionTree(ZeroRule(), metrix=infgain, max_depth=2)
    #model.fit(train_X, train_Y)
    model.fit(train_X, prob_Y)
    test_Z = model.predict(test_X)
    test_Z = np.argmax(test_Z, axis=1)
    acc = np.count_nonzero(test_Y == test_Z) / test_Z.shape[0]
    print(acc)
