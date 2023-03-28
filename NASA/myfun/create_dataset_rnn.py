import numpy as np

def create_dataset(dataset, look_back):
    x, y = [], []
    for i in range(len(dataset) - look_back):
        a = [dataset[i: (i + look_back)]]
        x.append(a)
        y.append(dataset[i + look_back])

        dataX = np.array(x)
        dataX = np.reshape(x,(-1, look_back, 1))
    return dataX, np.array(y)
 
# dataX, dataY = create_dataset(ts, look_back)