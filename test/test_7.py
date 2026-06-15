import numpy as np

l = 4

xl_1 = np.linspace(0, 1, l)
xl   = np.linspace(0, 1, l+1)

xs = np.unique(np.concatenate((xl_1, xl)))
xs.sort()

print(xl_1)
print(xl)
print(xs)

print("\n")

for k in range(len(xs)-1):

    a = xs[k]
    b = xs[k+1]

    # indice del nodo di xl immediatamente a sinistra di a
    i_left = np.searchsorted(xl, a, side='right') - 1

    # indice del nodo di xl immediatamente a destra di b
    i_right = np.searchsorted(xl, b, side='left')

    # protezione sugli estremi
    i_left = max(0, i_left)
    i_right = min(len(xl)-1, i_right)

    print(f"[{a:.6f}, {b:.6f}] --> ({i_left}, {i_right})",
          f"= [{xl[i_left]}, {xl[i_right]}]")
          
print("\n")          
          
for k in range(len(xs)-1):

    a = xs[k]
    b = xs[k+1]

    # indice del nodo di xl immediatamente a sinistra di a
    i_left = np.searchsorted(xl_1, a, side='right') - 1

    # indice del nodo di xl immediatamente a destra di b
    i_right = np.searchsorted(xl_1, b, side='left')

    # protezione sugli estremi
    i_left = max(0, i_left)
    i_right = min(len(xl)-1, i_right)

    print(f"[{a:.6f}, {b:.6f}] --> ({i_left}, {i_right})",
          f"= [{xl_1[i_left]}, {xl_1[i_right]}]")           