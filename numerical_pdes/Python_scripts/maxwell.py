"""Routines that are used to solve Maxwell's equations."""

import sys
from numpy import zeros, linspace, exp, sqrt

class OperatorsOrder2:
    """ This class contains all of the derivative operators that we will
    ever need. The operators are implemented to the second order.
    """

    def __init__(self, n_grid, delta):
        """Constructor for class OperatorsOrder2."""

        print(" Setting up 2nd-order derivative operators ")
        self.n_grid = n_grid
        self.delta = delta


    def laplace(self, fct):
        """Compute Laplace operator of function fct in interior of grid."""

        n_grid = self.n_grid
        ood2 = 1.0 / self.delta ** 2
        lap = zeros((n_grid, n_grid, n_grid))

        for i in range(1, n_grid - 1):
            for j in range(1, n_grid - 1):
                for k in range(1, n_grid - 1):
                    # see (B.26)
                    dfddx = (fct[i + 1, j, k] - 2.0 * fct[i, j, k] +
                             fct[i - 1, j, k])
                    dfddy = (fct[i, j + 1, k] - 2.0 * fct[i, j, k] +
                             fct[i, j - 1, k])
                    dfddz = (fct[i, j, k + 1] - 2.0 * fct[i, j, k] +
                             fct[i, j, k - 1])
                    lap[i, j, k] = ood2 * (dfddx + dfddy + dfddz)

        return lap

    def gradient(self, fct):
        """Compute gradient of function fct in interior of grid."""

        n_grid = self.n_grid
        oo2d = 0.5 / self.delta
        grad_x = zeros((n_grid, n_grid, n_grid))
        grad_y = zeros((n_grid, n_grid, n_grid))
        grad_z = zeros((n_grid, n_grid, n_grid))

        for i in range(1, n_grid - 1):
            for j in range(1, n_grid - 1):
                for k in range(1, n_grid - 1):
                    # see (A.12)
                    grad_x[i, j, k] = oo2d * (fct[i + 1, j, k] -
                                              fct[i - 1, j, k])
                    grad_y[i, j, k] = oo2d * (fct[i, j + 1, k] -
                                              fct[i, j - 1, k])
                    grad_z[i, j, k] = oo2d * (fct[i, j, k + 1] -
                                              fct[i, j, k - 1])

        return grad_x, grad_y, grad_z


    def divergence(self, v_x, v_y, v_z):
        """Compute divergence of vector v in interior of grid."""

        n_grid = self.n_grid
        oo2d = 0.5 / self.delta
        div = zeros((n_grid, n_grid, n_grid))

        for i in range(1, n_grid - 1):
            for j in range(1, n_grid - 1):
                for k in range(1, n_grid - 1):
                    # see (B.12)
                    v_xx = v_x[i + 1, j, k] - v_x[i - 1, j, k]
                    v_yy = v_y[i, j + 1, k] - v_y[i, j - 1, k]
                    v_zz = v_z[i, j, k + 1] - v_z[i, j, k - 1]
                    div[i, j, k] = oo2d * (v_xx + v_yy + v_zz)

        return div


    def grad_div(self, A_x, A_y, A_z):
        """Compute gradient of divergence in interior of grid."""

        n_grid = self.n_grid
        ood2 = 1.0 / self.delta ** 2
        grad_div_x = zeros((n_grid, n_grid, n_grid))
        grad_div_y = zeros((n_grid, n_grid, n_grid))
        grad_div_z = zeros((n_grid, n_grid, n_grid))

        for i in range(1, n_grid - 1):
            for j in range(1, n_grid - 1):
                for k in range(1, n_grid - 1):
                    # xx_A_x denotes \partial_x \partial_x A_x, etc...
                    xx_A_x = (A_x[i + 1, j, k] - 2.0 * A_x[i, j, k] +
                              A_x[i - 1, j, k])
                    xy_A_y = (A_y[i + 1, j + 1, k] - A_y[i + 1, j - 1, k] -
                              A_y[i - 1, j + 1, k] + A_y[i - 1, j - 1, k])
                    xz_A_z = (A_z[i + 1, j, k + 1] - A_z[i + 1, j, k - 1] -
                              A_z[i - 1, j, k + 1] + A_z[i - 1, j, k - 1])
                    grad_div_x[i, j, k] = ood2 * (xx_A_x + 0.25 * xy_A_y +
                                                  0.25 * xz_A_z)
                    yx_A_x = (A_x[i + 1, j + 1, k] - A_x[i + 1, j - 1, k] -
                              A_x[i - 1, j + 1, k] + A_x[i - 1, j - 1, k])
                    yy_A_y = (A_y[i, j + 1, k] - 2.0 * A_y[i, j, k] +
                              A_y[i, j - 1, k])
                    yz_A_z = (A_z[i, j + 1, k + 1] - A_z[i, j + 1, k - 1] -
                              A_z[i, j - 1, k + 1] + A_z[i, j - 1, k - 1])
                    grad_div_y[i, j, k] = ood2 * (0.25 * yx_A_x + yy_A_y +
                                                  0.25 * yz_A_z)
                    zx_A_x = (A_x[i + 1, j, k + 1] - A_x[i + 1, j, k - 1] -
                              A_x[i - 1, j, k + 1] + A_x[i - 1, j, k - 1])
                    zy_A_y = (A_y[i, j + 1, k + 1] - A_y[i, j + 1, k - 1] -
                              A_y[i, j - 1, k + 1] + A_y[i, j - 1, k - 1])
                    zz_A_z = (A_z[i, j, k + 1] - 2.0 * A_z[i, j, k] +
                              A_z[i, j, k - 1])
                    grad_div_z[i, j, k] = ood2 * (0.25 * zx_A_x + 0.25 *
                                                  zy_A_y + zz_A_z)

        return grad_div_x, grad_div_y, grad_div_z

    def partial_derivs(self, fct, i, j, k):
        """Compute partial derivatives. Unlike all
        other operators in this class, this function returns derivatives
        at one grid point (i,j,k) only, but including on the boundaries."""

        oo2d = 0.5 / self.delta

        # use (A.10) in interior, and (A.5) on boundaries
        # partial f / partial x
        if i == 0:
            f_x = oo2d * (-3.0 * fct[i, j, k] + 4.0 * fct[i + 1, j, k] -
                          fct[i + 2, j, k])
        elif i == self.n_grid - 1:
            f_x = oo2d * (3.0 * fct[i, j, k] - 4.0 * fct[i - 1, j, k] +
                          fct[i - 2, j, k])
        else:
            f_x = oo2d * (fct[i + 1, j, k] - fct[i - 1, j, k])

        # partial f / partial y
        if j == 0:
            f_y = oo2d * (-3.0 * fct[i, j, k] + 4.0 * fct[i, j + 1, k] -
                          fct[i, j + 2, k])
        elif j == self.n_grid - 1:
            f_y = oo2d * (3.0 * fct[i, j, k] - 4.0 * fct[i, j - 1, k] +
                          fct[i, j - 2, k])
        else:
            f_y = oo2d * (fct[i, j + 1, k] - fct[i, j - 1, k])

        # partial f / partial z
        if k == 0:
            f_z = oo2d * (-3.0 * fct[i, j, k] + 4.0 * fct[i, j, k + 1] -
                          fct[i, j, k + 2])
        elif k == self.n_grid - 1:
            f_z = oo2d * (3.0 * fct[i, j, k] - 4.0 * fct[i, j, k - 1] +
                          fct[i, j, k - 2])
        else:
            f_z = oo2d * (fct[i, j, k + 1] - fct[i, j, k - 1])
        return f_x, f_y, f_z



class Maxwell:
    """ The base class for evolution of Maxwell's equations.
    """

    def __init__(self, n_grid, x_out, filename_stem, n_vars):
        """Constructor sets up coordinates, memory for variables, and
        opens output file.
        """

        print(" Initializing class Maxwell with n_grid = ", n_grid,
              ", x_out = ", x_out)
        self.n_grid = n_grid
        self.filename_stem = filename_stem
        self.n_vars = n_vars
        self.delta = float(x_out) / (n_grid - 2.0)
        delta = self.delta
        print("    grid spacing delta =",delta)

        # set up cell-centered grid on interval (0, x_out) in each
        # dimension, but pad grid with one ghost zone.  Will use symmetry
        # on "inner" boundaries, and out-going wave conditions on outer
        # boundaries.
        self.x = linspace(-delta / 2.0, x_out + delta / 2.0, n_grid)
        self.y = linspace(-delta / 2.0, x_out + delta / 2.0, n_grid)
        self.z = linspace(-delta / 2.0, x_out + delta / 2.0, n_grid)
        self.r = zeros((n_grid, n_grid, n_grid))

        for i in range(0, n_grid):
            for j in range(0, n_grid):
                for k in range(0, n_grid):
                    self.r[i, j, k] = sqrt(
                        self.x[i] ** 2 + self.y[j] ** 2 + self.z[k] ** 2
                    )

        # set up derivative operators
        self.ops = OperatorsOrder2(n_grid, delta)

        # set up all variables common to both approaches
        self.E_x = zeros((n_grid, n_grid, n_grid))
        self.E_y = zeros((n_grid, n_grid, n_grid))
        self.E_z = zeros((n_grid, n_grid, n_grid))
        self.A_x = zeros((n_grid, n_grid, n_grid))
        self.A_y = zeros((n_grid, n_grid, n_grid))
        self.A_z = zeros((n_grid, n_grid, n_grid))
        self.phi = zeros((n_grid, n_grid, n_grid))
        self.constraint = zeros((n_grid, n_grid, n_grid))

        # open file that records constraint violations
        filename = self.filename_stem + "_constraints.data"
        self.constraint_file = open(filename, "w")
        if self.constraint_file:
            self.constraint_file.write(
                "# Constraint violations and errors with n = %d\n" % (n_grid)
                )
            self.constraint_file.write("# t          C_norm\n")
            self.constraint_file.write("#=========================\n")
        else:
            print(" Could not open file",filename," for output")
            print(" Check permissions?")
            sys.exit(2)
        # keep track of time
        self.t = 0.0


    def __del__(self):
        """Close output file in destructor."""
        if self.constraint_file:
            self.constraint_file.close()


    def initialize(self):
        """Set up initial data for E^i (assuming A^i = 0 initially);
        see (B.55).
        """

        for i in range(0, self.n_grid):
            for j in range(0, self.n_grid):
                for k in range(0, self.n_grid):
                    rl = self.r[i, j, k]
                    costheta = self.z[k] / rl
                    sintheta = sqrt(1.0 - costheta ** 2)
                    rho = sqrt(self.x[i] ** 2 + self.y[j] ** 2)
                    cosphi = self.x[i] / rho
                    sinphi = self.y[j] / rho
                    E_phi = -8.0 * rl * sintheta * exp(- rl ** 2)
                    self.E_x[i, j, k] = -E_phi * sinphi
                    self.E_y[i, j, k] = E_phi * cosphi
                    self.E_z[i, j, k] = 0.0

        self.check_constraint()


    def check_constraint(self):
        """Check constraint violation."""

        self.constraint = self.ops.divergence(self.E_x, self.E_y, self.E_z)
        norm_c = 0.0

        for i in range(1, self.n_grid - 1):
            for j in range(1, self.n_grid - 1):
                for k in range(1, self.n_grid - 1):
                    norm_c += self.constraint[i, j, k] ** 2

        norm_c = sqrt(norm_c * self.delta ** 3)
        self.constraint_file.write("%e %e \n" % (self.t, norm_c))
        print(" Constraint violation at time {0:.3f} : {1:.4e}".
                  format(self.t, norm_c))
        return norm_c


    def integrate(self, courant, t_max, t_check):
        """Carry out time integration."""

        print(" Integrating to time",t_max,"with Courant factor",courant)
        print("    checking results after times", t_check)
        self.write_data()
        while self.t + t_check <= t_max:
            fields = self.wrap_up_fields()
            # call stepper to intergrate from current time t to t + t_const
            self.t, fields = self.stepper(fields, courant, t_check)
            self.unwrap_fields(fields)
            self.check_constraint()
            self.write_data()


    def stepper(self, fields, courant, t_const):
        """Stepper for Runge-Kutta integrator, integrates
        from current time t to t + t_const by repeatedly calling icn.
        """
        delta_t = courant * self.delta
        time = self.t
        t_fin = time + t_const

        while time < t_fin:
            if t_fin - time < delta_t:
                delta_t = t_fin - time
            # call icn to carry out one time step
            fields = self.icn(fields, delta_t)
            time += delta_t

        return time, fields


    def icn(self, fields, dt):
        """Carry out one 2nd-order iterative Crank-Nicholson step;
        see (B.47).
        The routine derivatives(fields) is defined in the derived classes
        Original and Reformulated, and provides time derivatives of the fields
        according to the two different versions of Maxwell's equations.
        The routine update_fields(fields, fields_dot, factor, dt) adds
        factor * dt * fields_dot to fields.
        """

        # Step 1: get derivs at t0 and update fields with half-step
        fields_dot = self.derivatives(fields)
        new_fields = self.update_fields(fields, fields_dot, 0.5, dt)

        # Step 2: get derivs at t = t0 + dt and update fields temporarily
        fields_temp = self.update_fields(fields, fields_dot, 1.0, dt)
        fields_dot = self.derivatives(fields_temp)

        # Step 3: do it again...  (iterative step)
        fields_temp = self.update_fields(fields, fields_dot, 1.0, dt)
        fields_dot = self.derivatives(fields_temp)

        # Finally: update new_fields with second half-step
        new_fields = self.update_fields(new_fields, fields_dot, 0.5, dt)
        #

        return new_fields


    def update_fields(self, fields, fields_dot, factor, dt):
        """Updates all fields in the list fields,
        by adding factor * fields_dot * dt.
        """

        new_fields = []
        for var in range(0, self.n_vars):
            field = fields[var]
            field_dot = fields_dot[var]
            new_field = field + factor * field_dot * dt
            new_fields.append(new_field)

        return new_fields

    def outgoing_wave(self, f_dot, f):
        """Computes time derivatives of fields from
        outgoing-wave boundary condition;
        see (B.52)
        """
        n_grid = self.n_grid

        i = n_grid - 1  # upper x-boundary
        for j in range(0, n_grid - 1):
            for k in range(0, n_grid - 1):
                f_x, f_y, f_z = self.ops.partial_derivs(f, i, j, k)
                f_dot[i, j, k] = (-(f[i, j, k] + self.x[i] * f_x +
                                    self.y[j] * f_y + self.z[k] * f_z)
                                  / self.r[i, j, k])

        j = n_grid - 1  # upper y-boundary
        for i in range(0, n_grid):
            for k in range(0, n_grid - 1):
                f_x, f_y, f_z = self.ops.partial_derivs(f, i, j, k)
                f_dot[i, j, k] = (-(f[i, j, k] + self.x[i] * f_x +
                                    self.y[j] * f_y + self.z[k] * f_z)
                                  / self.r[i, j, k])

        k = n_grid - 1  # upper z-boundary
        for i in range(0, n_grid):
            for j in range(0, n_grid):
                f_x, f_y, f_z = self.ops.partial_derivs(f, i, j, k)
                f_dot[i, j, k] = (-(f[i, j, k] + self.x[i] * f_x +
                                    self.y[j] *  f_y + self.z[k] * f_z)
                                  / self.r[i, j, k])


    def symmetry(self, f_dot, x_sym, y_sym, z_sym):
        """Computes time derivatives on inner boundaries from symmetry,"""

        n_grid = self.n_grid

        i = 0  # lower x-boundary
        for j in range(1, n_grid - 1):
            for k in range(1, n_grid - 1):
                f_dot[i, j, k] = x_sym * f_dot[i + 1, j, k]

        j = 0  # lower y-boundary
        for i in range(0, n_grid - 1):
            for k in range(1, n_grid - 1):
                f_dot[i, j, k] = y_sym * f_dot[i, j + 1, k]

        k = 0  # lower z-boundary
        for i in range(0, n_grid - 1):
            for j in range(0, n_grid - 1):
                f_dot[i, j, k] = z_sym * f_dot[i, j, k + 1]


    def write_data(self):
        """Write data to file."""

        HEADER1 = "# x           y             E_x            E_y           "
        HEADER2 = "A_x           A_y           phi            C\n"
        WRITER = "%e  %e  %e  %e  %e  %e  %e  %e\n"

        filename = self.filename_stem + "_fields_{0:.3f}.data".format(self.t)
        out = open(filename, "w")
        if out:
            k = 1
            out.write("# data at time {0:.3f} at z = {1:e} \n".
                          format(self.t, self.z[k]))
            out.write(HEADER1)
            out.write(HEADER2)
            n_grid = self.n_grid
        
            for i in range(1, n_grid):
                for j in range(1, n_grid):
                    out.write(WRITER % (self.x[i], self.y[j],
                                        self.E_x[i, j, k], self.E_y[i, j, k],
                                        self.A_x[i, j, k], self.A_y[i, j, k],
                                        self.phi[i, j, k],
                                        self.constraint[i, j, k],))
                out.write("\n")
            out.close()
        else:
            print(" Could not open file", filename,"in write_data()")
            print(" Check permissions?")
        return
     

class Original(Maxwell):
    """ Derived class Original contains methods
    for the original version of Maxwell's equations.

    Note: in this version, we integrate n_vars = 7 independent variables.
    """

    def __init__(self, n_grid, x_out):
        """Constructor doesn't do much..."""

        print(" Integrating original Maxwell Equations ")
        print("    using n_grid =", n_grid, "\b^3 grid points with ")
        print("    outer boundary at x_out =", x_out)
        IND_VARS = 7
        filename_stem = "Max_orig_" + str(n_grid) + "_" + str(x_out)
        Maxwell.__init__(self, n_grid, x_out, filename_stem, IND_VARS)


    def wrap_up_fields(self):
        """Bundles up the seven variables into the list fields."""

        return (self.E_x, self.E_y, self.E_z, self.A_x, self.A_y,
                self.A_z, self.phi)


    def unwrap_fields(self, fields):
        """Extracts the independent variables from the list fields."""

        self.E_x = fields[0]
        self.E_y = fields[1]
        self.E_z = fields[2]
        self.A_x = fields[3]
        self.A_y = fields[4]
        self.A_z = fields[5]
        self.phi = fields[6]


    def derivatives(self, fields):
        """Computes the time derivative of the fields: Maxwell's equations."""

        self.unwrap_fields(fields)

        #
        # compute derivatives
        #
        Lap_A_x = self.ops.laplace(self.A_x)
        Lap_A_y = self.ops.laplace(self.A_y)
        Lap_A_z = self.ops.laplace(self.A_z)
        grad_x_phi, grad_y_phi, grad_z_phi = self.ops.gradient(self.phi)
        grad_div_x_A, grad_div_y_A, grad_div_z_A = self.ops.grad_div(
            self.A_x, self.A_y, self.A_z)
        DivA = self.ops.divergence(self.A_x, self.A_y, self.A_z)

        #
        # then compute time derivatives from original version
        # of Maxwell's equations,
        # see (4.1) and (4.2)
        #
        A_x_dot = -self.E_x - grad_x_phi
        A_y_dot = -self.E_y - grad_y_phi
        A_z_dot = -self.E_z - grad_z_phi
        E_x_dot = -Lap_A_x + grad_div_x_A
        E_y_dot = -Lap_A_y + grad_div_y_A
        E_z_dot = -Lap_A_z + grad_div_z_A
        phi_dot = -DivA

        #
        # finally fix inner boundaries from symmetry, see Table B.1...
        #
        self.symmetry(E_x_dot, 1, -1, 1)
        self.symmetry(E_y_dot, -1, 1, 1)
        self.symmetry(E_z_dot, 1, 1, -1)
        self.symmetry(A_x_dot, 1, -1, 1)
        self.symmetry(A_y_dot, -1, 1, 1)
        self.symmetry(A_z_dot, 1, 1, -1)
        self.symmetry(phi_dot, -1, -1, 1)

        #
        # ...and outer boundaries using outgoing-wave boundary condition (B.52)
        #
        self.outgoing_wave(E_x_dot, self.E_x)
        self.outgoing_wave(E_y_dot, self.E_y)
        self.outgoing_wave(E_z_dot, self.E_z)
        self.outgoing_wave(A_x_dot, self.A_x)
        self.outgoing_wave(A_y_dot, self.A_y)
        self.outgoing_wave(A_z_dot, self.A_z)
        self.outgoing_wave(phi_dot, self.phi)

        #
        return (E_x_dot, E_y_dot, E_z_dot, A_x_dot, A_y_dot, A_z_dot, phi_dot)


class Reformulated(Maxwell):
    """ Derived class Reformulated contains methods for
    the reformulated version of Maxwell's equations.

    Note: in the reformulated version we integrate n_vars = 8
    independent variables
    """

    def __init__(self, n_grid, x_out):
        """Constructor doesn't do much..."""

        IND_VARS = 8
        print(" Integrating reformulated Maxwell Equations ")
        print("    using n_grid =", n_grid, "\b^3 grid points with ")
        print("    outer boundary at x_out =", x_out)

        filename_stem = "Max_reform_" + str(n_grid) + "_" + str(x_out)
        Maxwell.__init__(self, n_grid, x_out, filename_stem, IND_VARS)

        # additional independent variable Gamma:
        self.Gamma = zeros((n_grid, n_grid, n_grid))


    def wrap_up_fields(self):
        """Bundles up the eight variables into the list fields."""

        return (self.E_x, self.E_y, self.E_z, self.A_x, self.A_y, self.A_z,
                self.phi, self.Gamma,)


    def unwrap_fields(self, fields):
        """Extracts the independent variables from the list fields."""

        self.E_x = fields[0]
        self.E_y = fields[1]
        self.E_z = fields[2]
        self.A_x = fields[3]
        self.A_y = fields[4]
        self.A_z = fields[5]
        self.phi = fields[6]
        self.Gamma = fields[7]


    def derivatives(self, fields):
        """Computes the time derivative of the fields: Maxwell's equations."""

        self.unwrap_fields(fields)

        #
        # compute derivatives
        #
        Lap_A_x = self.ops.laplace(self.A_x)
        Lap_A_y = self.ops.laplace(self.A_y)
        Lap_A_z = self.ops.laplace(self.A_z)
        grad_x_Gamma, grad_y_Gamma, grad_z_Gamma = \
          self.ops.gradient(self.Gamma)
        grad_x_phi, grad_y_phi, grad_z_phi = self.ops.gradient(self.phi)
        Lap_phi = self.ops.laplace(self.phi)

        #
        # then compute time derivatives from reformulated version
        # of Maxwell's equations;
        # see (4.8) through (4.10)
        #
        A_x_dot = -self.E_x - grad_x_phi
        A_y_dot = -self.E_y - grad_y_phi
        A_z_dot = -self.E_z - grad_z_phi
        E_x_dot = -Lap_A_x + grad_x_Gamma
        E_y_dot = -Lap_A_y + grad_y_Gamma
        E_z_dot = -Lap_A_z + grad_z_Gamma
        phi_dot = -self.Gamma
        Gamma_dot = -Lap_phi

        #
        # finally fix inner boundaries from symmetry, see Table B.1
        #
        self.symmetry(E_x_dot, 1, -1, 1)
        self.symmetry(E_y_dot, -1, 1, 1)
        self.symmetry(E_z_dot, 1, 1, -1)
        self.symmetry(A_x_dot, 1, -1, 1)
        self.symmetry(A_y_dot, -1, 1, 1)
        self.symmetry(A_z_dot, 1, 1, -1)
        self.symmetry(phi_dot, -1, -1, 1)
        self.symmetry(Gamma_dot, -1, -1, 1)

        #
        # ...and outer boundaries from outgoing-wave boundary condition (B.52)
        #
        self.outgoing_wave(E_x_dot, self.E_x)
        self.outgoing_wave(E_y_dot, self.E_y)
        self.outgoing_wave(E_z_dot, self.E_z)
        self.outgoing_wave(A_x_dot, self.A_x)
        self.outgoing_wave(A_y_dot, self.A_y)
        self.outgoing_wave(A_z_dot, self.A_z)
        self.outgoing_wave(phi_dot, self.phi)
        self.outgoing_wave(Gamma_dot, self.Gamma)

        #
        return (E_x_dot, E_y_dot, E_z_dot, A_x_dot, A_y_dot, A_z_dot, phi_dot,
                Gamma_dot,)

#
#=============================================================================
# Main routine
#=============================================================================
#
def main():
    """Main routine: define parameters and carry out integration."""
    print(" ------------------------------------------------------")
    print(" --- maxwell.py --- use flag -h for list of options ---") 
    print(" ------------------------------------------------------")
    #
    # set default values for variables
    #
    # number of grid points
    n_grid = 26
    # location of outer boundary
    x_out = 6.0
    # Courant factor
    courant = 0.5
    # integrate to time t_max
    t_max = 15
    # check constraints at increments of t_const
    t_check = 0.1
    # integrate original equations
    original = True
    #
    # now look for flags to overwrite default values
    #
    for i in range(len(sys.argv)):
        if sys.argv[i] == "-h":
            usage()
            return
        if sys.argv[i] == "-n_grid":
            n_grid = int(sys.argv[i+1])
        if sys.argv[i] == "-x_out":
            x_out = float(sys.argv[i+1])
        if sys.argv[i] == "-reform":
            original = False
        if sys.argv[i] == "-courant":
            courant = float(sys.argv[i+1])
        if sys.argv[i] == "-t_max":
            t_max = float(sys.argv[i+1])
        if sys.argv[i] == "-t_check":
            t_check = float(sys.argv[i+1])
    # decide whether to integrate original or reformulated equations    
    if original:
        james_clerk = Original(n_grid, x_out)
    else:
        james_clerk = Reformulated(n_grid, x_out)
    #
    # set up initial data
    james_clerk.initialize()
    #
    # now integrate...
    james_clerk.integrate(courant, t_max, t_check)


def usage():
    print("Solves Maxwell's equations in vacuum for electromagnetic wave.")
    print("")
    print("The following options can be used to overwrite default parameters")
    print("\t-n_grid: number of grid points [Default: 26 ]")
    print("\t-x_out: location of outer boundary [6.0]")
    print("\t-reform: use reformulated rather than original equations")
    print("\t-courant: Courant factor [0.5]")
    print("\t-t_max: maximum time [15]")
    print("\t-t_check: time after which constraints data are written [0.1]")
    print("For example, to evolve reformulated equations with x_out = 6.0, call")
    print("\tpython3 maxwell.py -reform -x_out 6.0")

if  __name__ == '__main__':
    main()
