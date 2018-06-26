def test(a):
    if a < 4:
        return 3, 5

if __name__ == '__main__':
    a, b = test(2)
    print(a)
    print(b)
