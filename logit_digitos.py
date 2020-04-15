
import matplotlib.pyplot as plt
import sklearn.datasets as skdata
import numpy as np
import sklearn.metrics as skm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression



numeros = skdata.load_digits()
target = numeros['target']
imagenes = numeros['images']
n_imagenes = len(target)

data = imagenes.reshape((n_imagenes, -1))

scaler = StandardScaler()
x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.5)

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

lrm = LogisticRegression(C=50./n_imagenes, penalty='l1', solver='saga', tol=0.1)
lrm.fit(x_train, y_train)
coefficients = lrm.coef_

plt.figure(figsize = (15,5))
for i in range(1,11):
	plt.subplot(2,5,i)
	plt.imshow(coefficients[i-1].reshape(8,8))
	plt.title(r'$\vec\beta_{num}$'.format(num = i-1))

plt.tight_layout()
plt.savefig('coeficientes.png')

confusion = skm.confusion_matrix(y_test,lrm.predict(x_test))

fig, ax = plt.subplots()
ax.matshow(confusion)
for i in range(0,10):
    for j in range(0,10):
        c = confusion[j,i]/len(y_test[y_test == j])
        ax.text(i, j, r'{:.2f}'.format(c), va = 'center', ha = 'center')

ax.xaxis.set_ticks_position('top') 
ax.xaxis.set_label_position('top') 
plt.xlabel('Predict')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('confusion.png')