import time

from blessings import Terminal


term = Terminal()

print('.')
print('.')
print('.')

ar = ['a', 'b', 'c']
x = 1
for i in ar:
    c = 1
    while c < 4:
        with term.location(c*3-3, term.height - (5-x)):
            print(str(i)+str(c-1)+',')
            time.sleep(0.5)
        c += 1   
    x += 1
