from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
import numpy as np

x = list(range(15))
y = list([0] * 5 + [1] * 10)

seed = 7
# rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2)
# for train_index, val_index in rskf.split(x, y):
#     print(train_index, val_index)
#     # print('train:', x[train_index], y[train_index])
#     # print('val:', x[val_index], y[val_index])
#     # model.fit(x=x[train],y=yTrain[train], epochs=15, batch_size=10, verbose=0)  # 训练模型


skf = StratifiedKFold(n_splits=5, shuffle=True)
for train_index, val_index in skf.split(x, y):
    # print(train_index, val_index)
    print('train:', x[train_index])

print(x)
print(y)
