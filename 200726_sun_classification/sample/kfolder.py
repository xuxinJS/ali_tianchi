from sklearn.model_selection import RepeatedStratifiedKFold
import numpy as np

x = np.array(list(range(20)))
y = np.array(list([0] * 10 + [1] * 10))

seed = 7
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2)

for train_index, val_index in rskf.split(x, y):
    # print(train_index, val_index)
    print('train:', x[train_index], y[train_index])
    print('val:', x[val_index], y[val_index])
    # model.fit(x=x[train],y=yTrain[train], epochs=15, batch_size=10, verbose=0)  # 训练模型


