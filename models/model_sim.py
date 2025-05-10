import numpy as np
import casadi as ca
import scipy.linalg as sci_linalg


class Quadrotor_Sim:
    def __init__(self):
        '''
        Open-loop Dynamics + Controller model of a quadrotor:

        The model includes:
            1. Inertia and Mass Bias
            2. Rotor Drag
            3. Others if needed

        - Open-loop Dynamics:
            dp = v
            dv = a + AeroDrag
            dEul = W(Eul) @ pqr
            dpqr = J^-1 (J pqr x pqr + tau)

            x = [p, v, eul, pqr]
            u = [T1, T2, T3, T4]
            dx = f(x, u)
            
        - Mellinger Controller / Differential Flatness-based Controller as Control Policy
            u, d(aux_x)/dt = ctrl(x, ref, aux_x)
            aux_x are dynamic variable in mellinger controller, serves as integrator and
            differentiator in PID control.
        '''
        self.__dynamic_param()
        self.__control_param()
        self.__saturation_params()

        # Model Information
        self.nomi_x_dim = 12  # Open-loop Model state dim
        self.nomi_u_dim = 4
        self.x_dim = self.nomi_x_dim
        self.u_dim = self.nomi_u_dim
        self.ref_dim = 12
    
    def __dynamic_param(self):
        self.m = 0.83
        self.m_actual = self.m
        self.Ixx = 3e-3
        self.Iyy = 3e-3
        self.Izz = 4e-3
        self.Ixx_actual = self.Ixx
        self.Iyy_actual = self.Iyy
        self.Izz_actual = self.Izz
        self.__compute_J()

        torque_coef = 0.01
        arm_length = 0.150
        self.CoeffM = np.array(
            [
                [
                    0.25,
                    -0.353553 / arm_length,
                    0.353553 / arm_length,
                    0.25 / torque_coef,
                ],
                [
                    0.25,
                    0.353553 / arm_length,
                    -0.353553 / arm_length,
                    0.25 / torque_coef,
                ],
                [
                    0.25,
                    0.353553 / arm_length,
                    0.353553 / arm_length,
                    -0.25 / torque_coef,
                ],
                [
                    0.25,
                    -0.353553 / arm_length,
                    -0.353553 / arm_length,
                    -0.25 / torque_coef,
                ],
            ]
        )                                               # Ttau -> MF
        self.CoeffM_inv = np.linalg.inv(self.CoeffM)    # MF -> Ttau
        self.aero_D = np.diag(np.zeros(3))
        self.Tm = 0.04                                  # Motor dynamcis

    def __compute_J(self):
        self.J = np.diag(np.array([self.Ixx, self.Iyy, self.Izz]))
        self.J_actual = np.diag(
            np.array([self.Ixx_actual, self.Iyy_actual, self.Izz_actual])
        )
        self.J_inv = np.diag(np.array([1 / self.Ixx, 1 / self.Iyy, 1 / self.Izz]))
        self.J_actual_inv = np.diag(
            np.array([1 / self.Ixx_actual, 1 / self.Iyy_actual, 1 / self.Izz_actual])
        )

    def __control_param(self):
        self.pos_gain = np.diag(np.array([1.0, 1.0, 0.7])) * 2
        self.vel_gain = self.pos_gain * 4
        self.eul_gain = np.diag(np.array([10.0, 10.0, 4.0]))
        self.omega_P = np.diag(np.array([40.0, 40.0, 16.0]))
        self.omega_I = np.diag(np.array([10.0, 10.0, 5.0]))
        self.omgea_D = np.diag(np.array([0.5, 0.5, 0.0]))

    def __saturation_params(self):
        self.u_lb = np.array([0.0, 0.0, 0.0, 0.0])
        self.u_ub = np.array([4.0, 4.0, 4.0, 4.0]) * 1.5

    def __rpy2DCMbe_num(self, eul_rpy):
        # eul_zyx <- eul_rpy 1,3 switch
        eul_z = eul_rpy[2]
        eul_y = eul_rpy[1]
        eul_x = eul_rpy[0]
        R1 = np.vstack(
            [
                np.hstack([np.cos(eul_z), -np.sin(eul_z), 0]),
                np.hstack([np.sin(eul_z), np.cos(eul_z), 0]),
                np.hstack([0, 0, 1]),
            ]
        )
        R2 = np.vstack(
            [
                np.hstack([np.cos(eul_y), 0, np.sin(eul_y)]),
                np.hstack([0, 1, 0]),
                np.hstack([-np.sin(eul_y), 0, np.cos(eul_y)]),
            ]
        )
        R3 = np.vstack(
            [
                np.hstack([1, 0, 0]),
                np.hstack([0, np.cos(eul_x), -np.sin(eul_x)]),
                np.hstack([0, np.sin(eul_x), np.cos(eul_x)]),
            ]
        )
        return R1 @ R2 @ R3
    
    def __dEul2omega_num(self, dEul_des, Eul):
        # Strap Down Equations
        domega_xdes = dEul_des[0] - (np.sin(Eul[1]) * dEul_des[2])
        domega_ydes = (dEul_des[1] * np.cos(Eul[0])) + (
            dEul_des[2] * np.sin(Eul[0]) * np.cos(Eul[1])
        )
        domega_zdes = -(dEul_des[1] * np.sin(Eul[0])) + (
            dEul_des[2] * np.cos(Eul[0]) * np.cos(Eul[1])
        )
        return np.hstack([domega_xdes, domega_ydes, domega_zdes])

    def __invert_eul_num(self, moment_des, omega):
        m1 = moment_des[0] + omega[1] * omega[2] * (self.Izz - self.Iyy)
        m2 = moment_des[1] + omega[0] * omega[2] * (self.Ixx - self.Izz)
        m3 = moment_des[2] + omega[0] * omega[1] * (self.Iyy - self.Ixx)
        return np.hstack([m1, m2, m3])

    def __derative3(self, interstate, x):
        # c=0.05 tf = s/(c*s+1)
        d_interstate = -10 * np.eye(3) @ interstate + 8 * np.eye(3) @ x
        x_der = -12.5 * np.eye(3) @ interstate + 10 * np.eye(3) @ x
        return d_interstate, x_der

    def aerodrag_num(self, x):
        dp = x[3:6]
        R = self.__rpy2DCMbe_num(x[6:9])
        return R @ self.aero_D @ R.T @ dp

    def openloop_num(self, x, MF, disturb=np.zeros(6)):
        Ttau = self.CoeffM_inv @ MF

        dp = x[3:6]

        dvx = (
            -Ttau[0]
            / self.m_actual
            * (np.cos(x[8]) * np.sin(x[7]) * np.cos(x[6]) + np.sin(x[8]) * np.sin(x[6]))
        )
        dvy = (
            -Ttau[0]
            / self.m_actual
            * (np.sin(x[8]) * np.sin(x[7]) * np.cos(x[6]) - np.cos(x[8]) * np.sin(x[6]))
        )
        dvz = 9.8 - Ttau[0] / self.m_actual * (np.cos(x[6]) * np.cos(x[7]))

        dv = (
            np.array([dvx, dvy, dvz])
            - self.aerodrag_num(x)
            + disturb[0:3] / self.m_actual
        )

        deul = (
            np.vstack(
                [
                    np.hstack(
                        [1, np.tan(x[7]) * np.sin(x[6]), np.tan(x[7]) * np.cos(x[6])]
                    ),
                    np.hstack([0, np.cos(x[6]), -np.sin(x[6])]),
                    np.hstack(
                        [0, np.sin(x[6]) / np.cos(x[7]), np.cos(x[6]) / np.cos(x[7])]
                    ),
                ]
            )
            @ x[9:12]
        )

        domega = self.J_actual_inv @ (
            -np.cross(x[9:12], self.J @ x[9:12]) + Ttau[1:4] + disturb[3:6]
        )

        return np.hstack([dp, dv, deul, domega])

    def ctrlmap_num(self, x, ref, omega_err_inte, omega_err_ds):
        # demux
        pos = x[0:3]
        vel = x[3:6]
        eul = x[6:9]
        omega = x[9:12]
        pos_ref = ref[0:3]
        vel_ref = ref[3:6]
        acc_ref = ref[6:9]
        jer_ref = ref[9:12]

        # Translational Loop
        a_des = (
            acc_ref + self.pos_gain @ (pos_ref - pos) + self.vel_gain @ (vel_ref - vel)
        )
        # Obtain Desire Rebs
        Zb = np.hstack([-a_des[0], -a_des[1], 9.8 - a_des[2]]) / np.linalg.norm(
            np.hstack([-a_des[0], -a_des[1], 9.8 - a_des[2]])
        )

        Xc = np.hstack([np.cos(0.0), np.sin(0.0), 0.0])
        Yb_ = np.cross(Zb, Xc)
        Yb = Yb_ / np.linalg.norm(Yb_)
        Xb = np.cross(Yb, Zb)
        Reb_des = np.vstack([Xb, Yb, Zb]).T
        # Reb_des = ca.horzcat(Xb, Yb, Zb)

        # Obtain Desire Eul, Omega and dOmega
        # eul_des = 'ZYX' + 1,3 Switch
        eul_des = np.hstack(
            [
                np.arctan2(Reb_des[2, 1], Reb_des[2, 2]),
                np.arctan2(
                    -Reb_des[2, 0], np.sqrt(Reb_des[1, 0] ** 2 + Reb_des[0, 0] ** 2)
                ),
                np.arctan2(Reb_des[1, 0], Reb_des[0, 0]),
            ]
        )

        T = -self.m * np.dot(Zb, (a_des - np.hstack([0.0, 0.0, 9.8])))
        h1 = -self.m / T * (jer_ref - np.dot(Zb, jer_ref) * Zb)

        omega_des = np.hstack([-np.dot(h1, Yb), np.dot(h1, Xb), 0.0])

        # h2 = (
        #     -np.cross(omega_des, np.cross(omega_des, Zb))
        #     + self.m / T * np.dot(jer_ref, Zb) * np.cross(omega_des, Zb)
        #     + 2 * self.m / T * np.dot(Zb, jer_ref) * np.cross(omega_des, Zb)
        # )
        # domega_des = np.array(
        #     [-np.dot(h2, Yb), np.dot(h2, Xb), 0.0]
        # )  # yaw, dyaw, ddyaw = 0

        # Attitude Loop
        dEul_des = self.eul_gain @ (eul_des - eul)
        omega_err = omega_des - omega + self.__dEul2omega_num(dEul_des, eul)
        d_omega_err_ds, omega_err_der = self.__derative3(omega_err_ds, omega_err)
        att_out = (
            self.omega_P @ omega_err
            + self.omega_I @ omega_err_inte
            + self.omgea_D @ omega_err_der
            + np.cross(omega, self.J @ omega)
            # + self.J @ domega_des
        )
        moment_des = self.J @ att_out
        tau = self.__invert_eul_num(moment_des, omega)

        MF = self.CoeffM @ np.hstack([T, tau[0], tau[1], tau[2]])
        # # Saturation
        # for i in range(MF.shape[0]):
        #     if self.u_lb[i] > MF[i]:
        #         MF[i] = self.u_lb[i]
        #     elif self.u_ub[i] < MF[i]:
        #         MF[i] = self.u_ub[i]
        #     else:
        #         pass

        return (
            MF,
            omega_err,
            d_omega_err_ds,
        )
    
    def actuators_dynamics(self, u_state, u_cmd):
        # Second Order System: Rotations to Force
        # u_output: [mf, dmf] * 4; u_cmd = [mf_cmd] * 4

        A = np.array([[0, 1 / self.Tm**2], [-1, -2 / self.Tm]])
        B = np.array([[0], [1]])
        As = sci_linalg.block_diag(*[A] * 4)
        Bs = sci_linalg.block_diag(*[B] * 4)

        du_state = As @ u_state + Bs @ u_cmd

        return du_state
    
    def get_nomi_u(self, x_sim, ref_sim):
        # u_state = x_sim[18:26]
        # return np.hstack([u_state[0], u_state[2], u_state[4], u_state[6]])
        return np.ones(4)
    
    def get_nomi_x(self, x_sim):
        # Return the nominal open-loop state
        return x_sim[0:12]
    
    def cldyn_num(self, x, ref, disturb=np.zeros(6)):
        MF, omega_err, d_omega_err_ds = self.ctrlmap_num(
            x[0:12], ref, x[12:15], x[15:18]
        )
        # u_state = x[18:26]
        # du_state = self.actuators_dynamics(u_state, u_cmd=MF)
        # mfs = self.get_nomi_u(x, ref)
        # dx = self.openloop_num(x, mfs)
        # return np.hstack([dx, omega_err, d_omega_err_ds, du_state])
        dx = self.openloop_num(x, MF, disturb)
        return np.hstack([dx, omega_err, d_omega_err_ds])

    def openloop_num_exRK4(self, xk, MF, dt, disturb=np.zeros(6)):
        # Nominal Dynamics: RK4 Discrete
        h = dt
        k1 = self.openloop_num(xk, MF, disturb)
        k2 = self.openloop_num((xk + 0.5 * h * k1), MF, disturb)
        k3 = self.openloop_num((xk + 0.5 * h * k2), MF, disturb)
        k4 = self.openloop_num((xk + h * k3), MF, disturb)
        xk1 = xk + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return xk1

    def cldyn_num_exRK4(self, xk, ref, dt, disturb=np.zeros(6)):
        # Closed-loop Nominal Dynamics: RK4 Discrete
        h = dt
        k1 = self.cldyn_num(xk, ref, disturb)
        k2 = self.cldyn_num((xk + 0.5 * h * k1), ref, disturb)
        k3 = self.cldyn_num((xk + 0.5 * h * k2), ref, disturb)
        k4 = self.cldyn_num((xk + h * k3), ref,disturb)
        xk1 = xk + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return xk1