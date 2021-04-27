# Code made for Sergio Andrés Díaz Ariza
# 25 Abril 2021
# License MIT
# Transport Phenomena: Python Program-Assessment 2.1

import time
import numpy as np

start_time = time.time()

# Matrix for Pi's Buckingham
A = np.array([[1, 0, 0], [-1, 3, 1], [-1, -1, 0]])

# We solve for the first Pi
B1 = np.array([-1, 1, 2])
C1 = np.linalg.solve(A, B1)

# We solve for the second Pi
B2 = np.array([0, 0, 0])
C2 = np.linalg.solve(A, B2)

# We solve for the third Pi
B3 = np.array([-1, 3, 0])
C3 = np.linalg.solve(A, B3)

# We solve for the fourth Pi
B4 = np.array([0, -1, 0])
C4 = np.linalg.solve(A, B4)

print(C1)
print(C2)
print(C3)
print(C4)

print("\n--- %s seconds ---" % (time.time() - start_time))
