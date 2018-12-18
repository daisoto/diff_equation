import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import os

os.system('cls')

def func(x):
    return math.cos(x)+math.sin(x)


def euler_func(h, y, x):
    
    def diff_eq(x, y):
        return 1/(math.cos(x)) - y*math.tan(x)
    
    return h*diff_eq(x, y)+y

start, end = 0, 1
n = int(input('Введите число шагов: '))
h=(end-start)/n

x = np.zeros(n+1)
yo = np.zeros(n+1)
y = np.zeros(n+1)
Δi = np.zeros(n+1)
Δ = np.zeros(n+1)

x[0] = start
yo[0] = func(start)
y[0] = yo[0]

for i in range(1, n+1):
    x[i] = x[i-1] + h
    yo[i] = func(x[i])
    y[i] = euler_func(h, y[i-1], x[i-1]) 
    Δi[i] = abs(yo[i] - y[i])
    Δ[i] = Δ[i-1] + Δi[i]

df = pd.DataFrame({
    'x': x,
    'y°': yo,
    'y': y,
    'Δi': Δi,
    'Δ': Δ
})

df.to_excel('result.xlsx')

plt.plot(x, y, "k--")
plt.plot(x, yo, "k.") 
plt.legend(['Аналитическое решение', 'Численное решение'])
plt.show()

plt.plot(x, Δ, "k--")
plt.plot(x, Δi, "k.") 
plt.legend(['∑Δ на i шаге', 'Δ на i шаге'])
plt.show()
