# example of doing matrix operations

from vectorlib.matrix import Matrix, IMAT

#a = Matrix(3, 3, 1, [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
a = Matrix(3, 3, 1, [[1.0, 2.0, 3.0], [2.0, 4.0, 1.0], [9.0, 9.0, 3.0]])

# [1 2 3]    [9 9 3]    [9   9       3]    [9   9     3] | 1 0 0
# [2 4 1] => [2 4 1] => [0   2    0.33] => [0   2  0.33] | 0 1 0
# [9 9 3]    [1 2 3]    [0   1    2.67]    [0   0   2.5] | 0 0 1
u, l, p = a.lu_fact()

print(f"L block: {l}")
print(f"U block: {u}")
print(f"LU: {l.multiply(u)}")
print(f"PA: {p.multiply(a)}")

imat = IMAT(3)
ans = a.solve(imat)

print(f"Inverse: {ans}")
print(f"VERT: {a.multiply(ans)}")
