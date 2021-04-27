# Code made for Sergio Andrés Díaz Ariza
# 24 Abril 2021
# License MIT
# Transport Phenomena: Python Program-Assessment 2.1

import time
import numpy as np

start_time = time.time()

# Matrix for Pi's Buckingham
A = np.array([[0, 1, 1, 1], [0, -3, -1, 0], [0, -1, 0, 0], [1, 1, -1, -3]])

# We solve for the first Pi
B1 = np.array([-1, 3, 1, 0])
C1 = np.linalg.solve(A, B1)

# We solve for the second Pi
B2 = np.array([0, 2, 1, -2])
C2 = np.linalg.solve(A, B2)

# We solve for the third Pi
B3 = np.array([0, 0, 0, -1])
C3 = np.linalg.solve(A, B3)

# We solve for the fourth Pi
B4 = np.array([0, 1, 0, 0])
C4 = np.linalg.solve(A, B4)

# We solve for the fifth Pi
B5 = np.array([-1, 1, 0, 1])
C5 = np.linalg.solve(A, B5)

print(C1)
print(C2)
print(C3)
print(C4)
print(C5)

print("\n--- %s seconds ---" % (time.time() - start_time))