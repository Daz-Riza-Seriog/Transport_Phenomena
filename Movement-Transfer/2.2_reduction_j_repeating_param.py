# Code made for Sergio Andrés Díaz Ariza
# 24 Abril 2021
# License MIT
# Transport Phenomena: Python Program-Assessment 2.1

import time
import numpy as np

start_time = time.time()


# Matrix for Pi's Buckingham
A = np.array([[1, 1, 0, 0], [-2, -1, 0, 1], [1, -1, 1, -2], [1, 0, 0, 0]])

# We solve for the first Pi
B1 = np.array([-1, 1, 2, 0])
C1 = np.linalg.solve(A, B1)

# We solve for the second Pi
B2 = np.array([-1, 3, 0, 1])
C2 = np.linalg.solve(A, B2)

print(C1)
print(C2)

# Matrix for Pi's Buckingham-Reducted
A2 = np.array([[1, 1, -1], [1, -1, 0], [0, -2, 0]])

# We solve for the first Pi
B2_1 = np.array([-1, 2, 0])
C2_1 = np.linalg.solve(A2, B2_1)

# We solve for the second Pi
B2_2 = np.array([0, -1, 0])
C2_2 = np.linalg.solve(A2, B2_2)

# We solve for the third Pi
B2_3 = np.array([-1, 0, 1])
C2_3 = np.linalg.solve(A2, B2_2)

print(C2_1)
print(C2_2)
print(C2_3)

print("\n--- %s seconds ---" % (time.time() - start_time))