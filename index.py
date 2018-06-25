import random

random.seed(1)

a = [1, 3, 4, 5, 5]
for i in range(10):
    print(random.choice(a), '  ',  end="")
print('..........')
for i in range(10):
    print(random.choice(a), '  ',  end="")
