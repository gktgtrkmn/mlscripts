from tinygrad.tensor import Tensor
import tinygrad.nn as nn
from tinygrad.nn.datasets import mnist
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters
from tinygrad.engine.jit import TinyJit

class Model:
    def __init__(self):
        self.l1 = nn.Conv2d(1, 32, kernel_size=(3,3))
        self.l2 = nn.Conv2d(32, 64, kernel_size=(3,3))
        self.l3 = nn.Linear(1600, 10)
    
    def __call__(self, x: Tensor) -> Tensor:
        x = self.l1(x).relu().max_pool2d(kernel_size=(2,2))
        x = self.l2(x).relu().max_pool2d(kernel_size=(2,2))
        return self.l3(x.flatten(1).dropout(0.5))
    
X_train, Y_train, X_test, Y_test = mnist()
print(X_train.shape, X_train.dtype, Y_train.shape, Y_train.dtype)

model = Model()

acc = (model(X_test).argmax(axis=1) == Y_test).mean()

optim = Adam(get_parameters(model))
batch_size = 128

def step():
    Tensor.training = True
    samples = Tensor.randint(batch_size, high=X_train.shape[0])
    X, Y = X_train[samples], Y_train[samples]
    optim.zero_grad()
    out = model(X)
    loss = out.sparse_categorical_crossentropy(Y).backward()
    loss.backward()
    optim.step()
    return loss.realize()

jit_step = TinyJit(step)

import timeit
timeit.repeat(jit_step, repeat=5, number=1)

for s in range(7000):
  loss = jit_step()
  if s%100 == 0:
    Tensor.training = False
    preds = model(X_test).argmax(axis=1)
    acc = (preds == Y_test).mean().item()
    print(f"step {s:4d}, loss {loss.item():.2f}, acc {acc*100.:.2f}%")
