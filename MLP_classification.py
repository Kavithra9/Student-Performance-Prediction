import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from utils import null_value_rows_remove, random_over_sampling, label_encoder, find_best_features, save_list_to_file

df = pd.read_excel('Student Academic Performance Prediction  (1) (2).xlsx')

df = null_value_rows_remove(df)

df = random_over_sampling(df, '29. Results (IS3440)')
df = random_over_sampling(df, '29. Results(IS3420)')

df = label_encoder(df)

features = df.drop(['29. Results (IS3440)', '29. Results(IS3420)'], axis=1)
targets = df[['29. Results (IS3440)', '29. Results(IS3420)']]

best_features = find_best_features(features, targets, 20)

save_list_to_file(best_features.to_list(), 'features_columns.txt')

X = features[best_features.to_list()]
y = targets

X_train, x_test, Y_train, y_test = train_test_split(X, y, test_size=0.20)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(20, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


net = Net()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

x_train, x_test, y_train, y_test = map(torch.tensor, (X_train.values, x_test.values, Y_train.values, y_test.values))

x_train = x_train.type(torch.float)
x_test = x_test.type(torch.float)

y_train = y_train.type(torch.float)
y_test = y_test.type(torch.float)

best_valid_loss = 2

for epoch in range(50000):
    outputs = net(x_train)
    optimizer.zero_grad()
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 1000 == 0:
        print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

        with torch.no_grad():
            val = net(x_test)
            val_loss = criterion(val, y_test).item()
            if val_loss < best_valid_loss:
                best_valid_loss = val_loss
                print("Weight saved! ", val_loss)
                torch.save(net.state_dict(), './weights/weight.pth')

y_pred = net(x_test)
y_pred = torch.round(y_pred)

res = torch.eq(y_test, y_pred)

first_sub_count = 0
second_sub_count = 0
final_count = 0

for i in res.tolist():
    if i[0]:
        first_sub_count = first_sub_count + 1
    if i[1]:
        second_sub_count = second_sub_count + 1

    if i[0] and i[1]:
        final_count = final_count + 1

print("First subject accuracy", (first_sub_count / y_test.shape[0]) * 100)
print("Second subject accuracy", (second_sub_count / y_test.shape[0]) * 100)
print("Both subject accuracy", (final_count / y_test.shape[0]) * 100)
