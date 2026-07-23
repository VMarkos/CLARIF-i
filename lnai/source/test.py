# test.py
#
# Temporary file used for testing

from utils import monitor_sequences

if __name__ == '__main__':
    log = dict()

    @monitor_sequences(log)
    def foo(ls):
        for x in ls[:2]:
            print(ls, x)
            ls.append(x)

    a = [1, 2, 3]
    foo(a)
    print(log)
