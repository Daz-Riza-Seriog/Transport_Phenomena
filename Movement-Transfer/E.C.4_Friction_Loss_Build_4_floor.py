# Code made for Sergio Andrés Díaz Ariza
# 04 August 2021
# License MIT
# Transport Phenomena: Python Case of Study Friction Loss  in Build

from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.integrate import quad
from sklearn.linear_model import LinearRegression
import seaborn as sns
import numpy as np
import time

start_time = time.time()
sns.set()

# Tank Parameters
Vol = 1.5  # [L] Capacity o the Tank
Dt = 1.2  # [m] Diameter of the tank
# Vol = A * z_t  --> Area * height of tank
z_t = (4 * Vol) / (np.pi * (Dt ** 2))

# Parameters  of Friction
eps = 0
Ds = 0.0254  # Diameter of outer [m]
alpha = 1.05

L1 = 9  # Length of pipe [m]
L2 = 6  # Length of pipe [m]
L3 = 3  # Length of pipe [m]
rho = 987  # Density of carbohydrates at 38 C  [Kg/m^3]
miu = 0.891e-3  # [Kg/m*s]

# Bernoulli Eq for the floors

g = 9.89  # [m/s^2] Value of Gravity acceleration
Patm = 101325  # [Pa]
P1 = P2 = P3 = P4 = Patm
z_4 = 9 + z_t  # Height of the tank
z_3 = 6  # Height of the third floor
z_2 = 3  # Height of the second floor
z_1 = 0  # Height of the first floor
L = L1


class Point_A:
    #  Resolve for Bernoulli

    V_4 = 0
    V_3 = np.sqrt(2 * g * (z_4 - z_3))
    V_2 = np.sqrt(2 * g * (z_4 - z_2))
    V_1 = np.sqrt(2 * g * (z_4 - z_1))

    # Find the Caudal Qi

    Q_1 = (np.pi * (Ds ** 2) * V_1) / 4
    Q_2 = (np.pi * (Ds ** 2) * V_2) / 4
    Q_3 = (np.pi * (Ds ** 2) * V_3) / 4
    Q_4 = (np.pi * (Ds ** 2) * V_4) / 4

    # Resolve the Balance of Mechanic Energy

    class Optimice:
        def objective_Colebrook(self, x, *args):
            x1 = x[0]  # Darcy factor
            x2 = x[1]  # Velocity Average

            return (1 / np.sqrt(x1)) + (2.0 * np.log10(((eps / Ds) / 3.7) + (2.51 / (
                    (rho * Ds * x2 / miu) * np.sqrt(x1)))))

        def constraint_Vavg_eq_hl(self, x, *args):
            # Parameters
            x1 = x[0]  # Darcy factor
            x2 = x[1]  # Velocity Average
            return (x2 ** 2) - ((2 * g / alpha) * ((z_4 - z_1) - (x1 * L * (x2 ** 2) / Ds * 2 * g)))

    Opt = Optimice()
    constraint_equal = {'type': 'eq', 'fun': Opt.objective_Colebrook}
    constraint_equal1 = {'type': 'eq', 'fun': Opt.constraint_Vavg_eq_hl}

    bnd_f = (0.001, 0.1)
    bnd_v = (1, 14.29)
    bnd = [bnd_f, bnd_v]
    constraint = [constraint_equal]
    L = L1
    x0 = [0.05, 10]  # This inital values are extracted from a first solution given by the method
    sol = minimize(Opt.objective_Colebrook, x0, method='SLSQP', constraints=constraint, bounds=bnd,
                   args=(eps, Ds, rho, miu, L, g, z_4, z_1, alpha),
                   options={'maxiter': 1000})
    L = L2
    bnd_v = (1, 12.04)
    bnd = [bnd_f, bnd_v]
    x0 = [0.05, 8]
    sol2 = minimize(Opt.objective_Colebrook, x0, method='SLSQP', constraints=constraint, bounds=bnd,
                    args=(eps, Ds, rho, miu, L, g, z_4, z_2, alpha),
                    options={'maxiter': 1000})

    L = L3
    bnd_v = (1, 9.25)
    bnd = [bnd_f, bnd_v]
    x0 = [0.05, 5]
    sol3 = minimize(Opt.objective_Colebrook, x0, method='SLSQP', constraints=constraint, bounds=bnd,
                    args=(eps, Ds, rho, miu, L, g, z_4, z_3, alpha),
                    options={'maxiter': 1000})

    Q_1_f = (np.pi * (Ds ** 2) * sol.x[1]) / 4
    Q_2_f = (np.pi * (Ds ** 2) * sol2.x[1]) / 4
    Q_3_f = (np.pi * (Ds ** 2) * sol3.x[1]) / 4

    print("Velocities without friction floor 1,2 & 3:\t", V_1, '\t', V_2, '\t', V_3, "\n")
    print("Velocities with friction floor 1,2 & 3:\t", sol.x[1], '\t', sol2.x[1], '\t', sol3.x[1], "\n")
    print("Caudal without friction floor 1,2 & 3:\t", Q_1, '\t', Q_2, '\t', Q_3, "\n")
    print("Caudal with friction floor 1,2 & 3:\t", Q_1_f, '\t', Q_2_f, '\t', Q_3_f, "\n")


class Point_B:
    P_A = Point_A

    # Time to empty the tank to floor 1 without frictions

    ###
    t_floor_1 = (-2 / np.sqrt(2 * g)) * ((Dt / Ds) ** 2) * (np.sqrt(L1 - z_1) - np.sqrt(L1 + z_t - z_1))
    ###

    print("Time to empty tank to floor 1 without friction:\t", t_floor_1, "\n")

    # Time to empty the tank to floor 1 with frictions
    # h_l
    f_floor_1 = P_A.sol.x[0]
    V_avg_floor_1 = P_A.sol.x[1]

    h_l_1 = (f_floor_1 * L1 * (V_avg_floor_1 ** 2)) / (Ds * 2 * g)

    # Define the Integral
    def Tim_tank(x, z_1, h_l):
        f = 1 / (np.sqrt((x - z_1 - h_l)))
        return f

    # Solve the Integrate
    sol_I = quad(Tim_tank, L1 + z_t, L1, args=(z_1, h_l_1))

    t_floor_friction_1 = (-1 * np.sqrt(alpha) / np.sqrt(2 * g)) * ((Dt / Ds) ** 2) * sol_I[0]


    ###
    t_floor_1_f = (-2 * np.sqrt(alpha) / np.sqrt(2 * g)) * ((Dt / Ds) ** 2) * (
            np.sqrt(L1 - z_1 - h_l_1) - np.sqrt(L1 + z_t - z_1 - h_l_1))
    ###

    print("Time to empty tank to floor 1 with maximum friction:\t", t_floor_friction_1*2, "\n")


class Point_C:
    P_B = Point_B
    P_A = Point_A
    f_floor_1 = P_A.sol.x[0]
    z_t_i = z_t
    # Resolve the Balance of Mechanic Energy

    Z_T = np.linspace(L1 + z_t, L1, 200)

    class Optimice:
        def objective_Colebrook(self, x, *args):
            x1 = x[0]  # Darcy factor
            x2 = x[1]  # Velocity Average

            return (1 / np.sqrt(x1)) + (2.0 * np.log10(((eps / Ds) / 3.7) + (2.51 / (
                    (rho * Ds * x2 / miu) * np.sqrt(x1)))))

        def constraint_Vavg_eq_hl(self, x, *args):
            # Parameters
            x1 = x[0]  # Darcy factor
            x2 = x[1]  # Velocity Average
            return (x2 ** 2) - ((2 * g / alpha) * ((z_t - z_1) - (x1 * L * (x2 ** 2) / Ds * 2 * g)))

    Opt = Optimice()
    constraint_equal = {'type': 'eq', 'fun': Opt.objective_Colebrook}
    constraint_equal1 = {'type': 'eq', 'fun': Opt.constraint_Vavg_eq_hl}

    bnd_f = (0.001, 0.1)
    bnd_v = (1, 14.29)
    bnd = [bnd_f, bnd_v]
    constraint = [constraint_equal]
    L = L1
    x0 = [0.05, 10]  # This inital values are extracted from a first solution given by the method

    f_c = []
    vel = []
    for z_t in Z_T:
        sol = minimize(Opt.objective_Colebrook, x0, method='SLSQP', constraints=constraint, bounds=bnd,
                       args=(eps, Ds, rho, miu, L, g, z_t, z_1, alpha),
                       options={'maxiter': 1000})

        bnd_f_u = sol.x[1]
        bnd_f_ = (1, bnd_f_u)
        bnd = [bnd_f, bnd_f_]
        x0 = [sol.x[0] - 0.00133, sol.x[1] - 0.0133]

        vel.append(sol.x[1])
        f_c.append(sol.x[0])

    z_t_x_axe = Z_T.reshape((-1, 1))
    vel_y_axe = vel

    model = LinearRegression(n_jobs=-1).fit(z_t_x_axe, vel_y_axe)

    #  Obtain data from Linear Regression
    R_sq = model.score(z_t_x_axe, vel_y_axe)  # Coefficient of Determination
    Intercept = model.intercept_  # Intercept of the Regression
    Slope = model.coef_[0]  # Slope of the Regression


    #   Solve for Vel variable in time
    # Define the Integral
    def Tim_tank(x, Slope, Intercept):
        return Slope * x + Intercept

    # Solve the Integrate
    sol_I = quad(Tim_tank, L1 + z_t_i, L1, args=(Slope, Intercept))

    t_floor_friction_variables_1 = (-1 * np.sqrt(alpha) / np.sqrt(2 * g)) * ((Dt / Ds) ** 2) * sol_I[0]

    ####

    t_f_fric_var_1 = (-1 * np.sqrt(alpha) / np.sqrt(2 * g)) * ((Dt / Ds) ** 2) * \
                     (((Slope / 2) * (L1 ** 2)) + (Intercept * L1) - ((Slope / 2) * ((L1 + z_t_i) ** 2)) - (
                             Intercept * (L1 + z_t_i)))
    ###

    print("Time to empty tank to floor 1 with variable friction:\t", t_floor_friction_variables_1 / 3)

    # Plot Linear Regression

    Z_T_ = np.linspace(L1, L1 + z_t_i, 200)

    def Vel_Reg(self, Z_T, Slope, Intercept):
        vel_reg = Slope * Z_T + Intercept
        return vel_reg

    vel_reg = Vel_Reg(0, Z_T_, Slope, Intercept)

    plt.figure(1)
    plt.plot(Z_T, vel, '|', label='Points from Iteration')
    plt.plot(Z_T_, vel_reg, label="Linear Regression")
    plt.title("Linear Regression for\n function Velocity in term of $Z_T$", fontsize=16)
    plt.ylabel("Velocity $[m/s]$", fontsize=14)
    plt.xlabel("Level of Water in Tank $[m]$ ", fontsize=14)
    plt.text(9.2, 3.0, r'$\ R^2 = {:.10f} $'.format(R_sq))
    plt.legend()


class Point_D:
    P_A = Point_A
    P_C = Point_C

    Z_T = np.linspace(0, z_t, 1000)

    # Function of The level Tank with Time - Ideal
    t_floor_1 = (-2 / np.sqrt(2 * g)) * ((Dt / Ds) ** 2) * (np.sqrt(L1 - z_1) - np.sqrt(L1 + Z_T - z_1))

    # Function of The level Tank with Time - Maximum Friction
    f_floor_1 = P_A.sol.x[0]
    V_avg_floor_1 = P_A.sol.x[1]

    h_l_1 = (f_floor_1 * L1 * (V_avg_floor_1 ** 2)) / (Ds * 2 * g)

    t_floor_1_f = (-2 * np.sqrt(alpha) / np.sqrt(2 * g)) * ((Dt / Ds) ** 2) * (
            np.sqrt(L1 - z_1 - h_l_1) - np.sqrt(L1 + Z_T - z_1 - h_l_1))

    # Function of The level Tank with Time - Variable Friction
    Slope = Point_C.Slope
    Intercept = Point_C.Intercept

    t_f_fric_var_1 = (-1 * np.sqrt(alpha) / np.sqrt(2 * g)) * ((Dt / Ds) ** 2) * \
                     (((Slope / 2) * (L1 ** 2)) + (Intercept * L1) - ((Slope / 2) * ((L1 + Z_T) ** 2)) - (
                             Intercept * (L1 + Z_T)))

    plt.figure(2)
    plt.plot(t_f_fric_var_1 / 3, Z_T, label="Friction Variability")
    plt.plot(t_floor_1_f*2, Z_T, label="Maximum Friction")
    plt.plot(t_floor_1, Z_T, label="Without Friction")
    plt.title("Level of Tank $Z_T$ \n$vs$\n Time", fontsize=16)
    plt.ylabel("Level of Water in Tank $Z_T$ $[m]$ ", fontsize=14)
    plt.xlabel("Time $[s]$ ", fontsize=14)
    plt.legend()
    plt.show()

    print("\n--- %s seconds ---" % (time.time() - start_time))
