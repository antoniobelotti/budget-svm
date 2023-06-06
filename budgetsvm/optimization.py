import time
from abc import ABC, abstractmethod

import numpy as np
import itertools as it
import logging

logger = logging.getLogger(__name__)

from gurobipy import LinExpr, GRB, Model, Env, QuadExpr, quicksum
from budgetsvm.kernel import GaussianKernel


class Solver(ABC):
    """Abstract solver for optimization problems.

    The base class for solvers is :class:`Solver`: it exposes a method
    `solve` which delegates the numerical optimization process to an abstract
    method `solve_<problem_type>_problem` and subsequently clips the results to the boundaries
    of the feasible region.
    """

    def __init__(self, problem="classification"):
        self.problem = problem
        self.solve_dispatch = {
            "classification": self.solve_classification_problem,
            "regression": self.solve_regression_problem,
        }

        self.clip_dispatch = {
            "classification": self.clip_classification_solution,
            "regression": self.clip_regression_solution,
        }

    @abstractmethod
    def solve_classification_problem(self, *args, **kwargs):
        pass

    @abstractmethod
    def solve_regression_problem(self, *args, **kwargs):
        pass

    @staticmethod
    def clip_regression_solution(solution, budget, C):
        alpha, alpha_hat = solution[:2]
        if budget is not None:
            gamma = solution[-1]

        if budget is None:
            alpha_clipped = np.clip(alpha, 0, C)
            alpha_clipped[np.isclose(alpha_clipped, 0)] = 0
            alpha_clipped[np.isclose(alpha_clipped, C)] = C
            alpha_hat_clipped = np.clip(alpha_hat, 0, C)
            alpha_hat_clipped[np.isclose(alpha_hat_clipped, 0)] = 0
            alpha_hat_clipped[np.isclose(alpha_hat_clipped, C)] = C
            optimal_values = alpha_clipped, alpha_hat_clipped
        else:
            alpha_clipped = np.clip(alpha, 0, np.inf)
            alpha_clipped[np.isclose(alpha_clipped, 0)] = 0
            alpha_hat_clipped = np.clip(alpha_hat, 0, np.inf)
            alpha_hat_clipped[np.isclose(alpha_hat_clipped, 0)] = 0
            gamma_clipped = np.clip(gamma, 0, np.inf)
            gamma_clipped[np.isclose(gamma_clipped, 0)] = 0
            optimal_values = alpha_clipped, alpha_hat_clipped, gamma_clipped

        return optimal_values

    @staticmethod
    def clip_classification_solution(solution, budget, C):
        solution = np.array(solution)
        if budget is None:
            alpha = solution
        else:
            alpha, gamma = solution
            gamma[np.isclose(gamma, 0, atol=1e-6, rtol=0)] = 0
            gamma[np.isclose(gamma, 1, atol=1e-6, rtol=0)] = 1

        alpha[np.isclose(alpha, 0, atol=1e-6, rtol=0)] = 0
        alpha[np.isclose(alpha, C, atol=1e-6, rtol=0)] = C

        if budget is not None:
            alpha[gamma == 0] = 0

        return alpha

    def solve(self, X, y, C=1, kernel=GaussianKernel(), budget=None, **kwargs):
        """Solve optimization phase.

        Build and solve the constrained optimization problem on the basis
        of the fuzzy learning procedure.

        :param X: Objects in training set.
        :type X: iterable
        :param y: Membership values for the objects in `xs`.
        :type y: iterable
        :param C: constant managing the trade-off in joint radius/error
          optimization.
        :type C: float
        :param kernel: Kernel function to be used.
        :type kernel: :class:`mulearn.kernel.Kernel`
        :param budget: constant, upper bound on the number of support vectors
        :type budget: int
        :raises: ValueError if C is non-positive or if xs and mus have
          different lengths.
        :returns: `list` -- optimal values for the independent variables
          of the problem."""

        if C <= 0:
            raise ValueError("C should be positive")

        y = np.array(y)

        solve_fn = self.solve_dispatch[self.problem]
        solution, optimal, abs_mip_gap = solve_fn(X, y, C=C, kernel=kernel, budget=budget, **kwargs)

        clip_fn = self.clip_dispatch[self.problem]
        optimal_values = clip_fn(solution, budget, C)

        return optimal_values, optimal, abs_mip_gap


class GurobiSolver(Solver):
    """Solver based on gurobi.

    Using this class requires that gurobi is installed and activated
    with a software key. The library is available at no cost for academic
    purposes (see
    https://www.gurobi.com/downloads/end-user-license-agreement-academic/).
    Alongside the library, also its interface to python should be installed,
    via the gurobipy package.
    """

    def __init__(self, problem="classification", time_limit=60, initial_values=None):
        """
        Build an object of type GurobiSolver.

        :param problem: Type of problem to be solved.
        :type problem: str ('classification' and 'regression' are allowed)
        :param time_limit: Maximum time (in seconds) before stopping iterative
          optimization, defaults to 10*60.
        :type time_limit: int
        :param initial_values: Initial values for variables of the optimization
          problem, defaults to None.
        :type initial_values: iterable of floats or None
        """
        super().__init__(problem)
        self.time_limit = time_limit
        self.initial_values = initial_values

    def solve_classification_problem(
        self, X, y, C=1, kernel=GaussianKernel(), budget=None
    ):
        """Optimize the classification-based optimization problem via gurobi.

        Build and solve the constrained optimization problem at the basis
        of SV classification, either classic or budgeted, using the gurobi
        API.

        :param X: Objects in training set.
        :type X: iterable
        :param y: Binary labels for the objects in `xs`.
        :type y: iterable
        :param C: constant managing the trade-off in joint complexity/error
                  optimization.
        :type C: float
        :param kernel: Kernel function to be used.
        :type kernel: :class:`mulearn.kernel.Kernel`
        :param budget: value of the budget (None in case of standard
                       classification).
        :type budget: float
        :raises: ValueError if C is non-positive, budget is specified as a
                 negative value or if X and y have different lengths.
        :returns: `list` -- optimal values for the independent variables
          of the problem."""

        m = len(X)
        assert m == len(y)

        assert C > 0
        if budget is not None:
            assert budget > 0

        with Env(empty=True) as env:
            env.setParam("LogToConsole", 0)
            env.setParam("LogFile", "gurobi.log")
            env.start()
            with Model("svc", env=env) as model:
                model.setParam("LogToConsole", 0)
                model.setParam("TimeLimit", self.time_limit)

                model.addVars(list(range(m)), lb=0, ub=C, vtype=GRB.CONTINUOUS)

                if budget is not None:
                    for i in range(m):
                        model.addVar(name=f"gamma_{i}", vtype=GRB.BINARY)

                model.update()
                vars = model.getVars()

                alpha = np.array(vars[:m])
                if budget is not None:
                    gamma = np.array(vars[m:])

                if self.initial_values is not None:
                    for a, i in zip(alpha, self.initial_values[0]):
                        a.start = i

                    if budget is not None:
                        for g, i in zip(gamma, self.initial_values[0]):
                            g.start = i

                obj = QuadExpr()

                obj.addTerms([1.0] * len(alpha), alpha.tolist())

                if kernel.precomputed:
                    for i, j in it.product(range(m), range(m)):
                        obj.add(
                            alpha[i] * alpha[j],
                            -0.5 * y[i] * y[j] * X[i][j],
                        )
                else:
                    for i, j in it.product(range(m), range(m)):
                        obj.add(
                            alpha[i] * alpha[j],
                            -0.5 * y[i] * y[j] * kernel.compute(X[i], X[j]),
                        )

                model.setObjective(obj, GRB.MAXIMIZE)

                constEqual = LinExpr()
                constEqual.add(quicksum(alpha * y), 1.0)

                model.addLConstr(constEqual, GRB.EQUAL, 0)

                if budget is not None:
                    for a, g in zip(alpha, gamma):
                        const = QuadExpr()
                        const.add(a, 1.0)
                        model.addQConstr(const, GRB.LESS_EQUAL, C * g)

                    const = LinExpr()
                    const.add(quicksum(gamma), 1.0)
                    model.addLConstr(const, GRB.LESS_EQUAL, budget)

                model.optimize()

                if model.Status != GRB.OPTIMAL:
                    if model.Status != GRB.TIME_LIMIT or model.SolCount == 0:
                        raise ValueError("no solution found! " f"status={model.Status}")

                self.obj_val = model.getObjective().getValue()

                alpha_opt = np.array([a.x for a in alpha])
                if budget is not None:
                    gamma_opt = np.array([g.x for g in gamma])

                solution = (alpha_opt, gamma_opt) if budget is not None else alpha_opt

                mip_gap = 0 if np.isclose(model.Params.MIPGap, 0, atol=1e-4, rtol=0) else model.Params.MIPGap
                return np.array(solution), model.Status == GRB.OPTIMAL, mip_gap

    def solve_regression_problem(
        self, X, y, C=1, kernel=GaussianKernel(), epsilon=0.1, budget=None
    ):
        """Optimize via gurobi.

        Build and solve the constrained optimization problem at the basis
        of the fuzzy learning procedure using the gurobi API.

        :param X: Objects in training set.
        :type X: iterable
        :param y: Membership values for the objects in `xs`.
        :type y: iterable
        :param C: constant managing the trade-off in joint radius/error
          optimization.
        :type C: float
        :param kernel: Kernel function to be used.
        :type kernel: :class:`mulearn.kernel.Kernel`
        :raises: ValueError if C is non-positive or if xs and mus have
          different lengths.
        :returns: `list` -- optimal values for the independent variables
          of the problem."""

        m = len(X)

        with Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()
            with Model("svr", env=env) as model:
                model.setParam("OutputFlag", 0)
                model.setParam("TimeLimit", self.time_limit)

                for i in range(m):
                    if C < np.inf and budget is None:
                        model.addVar(
                            name=f"alpha_{i}", lb=0, ub=C, vtype=GRB.CONTINUOUS
                        )
                        model.addVar(
                            name=f"alphahat_{i}", lb=0, ub=C, vtype=GRB.CONTINUOUS
                        )
                    else:
                        model.addVar(name=f"alpha_{i}", lb=0, vtype=GRB.CONTINUOUS)
                        model.addVar(name=f"alphahat_{i}", lb=0, vtype=GRB.CONTINUOUS)
                if budget is not None:
                    model.addVar(name="gamma", lb=0, vtype=GRB.CONTINUOUS)

                model.update()
                vars = model.getVars()

                gamma = vars[-1]
                alpha = np.array(vars[:m])
                alpha_hat = np.array(vars[m : 2 * m])

                if self.initial_values is not None:
                    for a, i in zip(alpha, self.initial_values[0]):
                        a.start = i
                    for a, i in zip(alpha_hat, self.initial_values[1]):
                        a.start = i

                    if budget is not None:
                        gamma.start = self.initial_values[2]

                obj = QuadExpr()

                for a, a_h, y_ in zip(alpha, alpha_hat, y):
                    obj.add((a + a_h), epsilon)
                    obj.add(-(a - a_h), y_)
                    if budget is not None:
                        obj.add(gamma * budget)

                for i, j in it.product(range(m), range(m)):
                    obj.add(
                        (alpha[i] - alpha_hat[i]) * (alpha[j] - alpha_hat[j]),
                        kernel.compute(X[i], X[j]) * 0.5,
                    )

                model.setObjective(obj, GRB.MINIMIZE)

                constEqual = LinExpr()
                constEqual.add(sum(alpha - alpha_hat), 1.0)

                model.addLConstr(constEqual, GRB.EQUAL, 0)

                if budget is not None:
                    for a in alpha:
                        const = LinExpr()
                        const.add(a - gamma, 1.0)
                        model.addLConstr(const, GRB.LESS_EQUAL, C)
                    for a in alpha_hat:
                        const = LinExpr()
                        const.add(a - gamma, 1.0)
                        model.addLConstr(const, GRB.LESS_EQUAL, C)

                model.optimize()

                if model.Status != GRB.OPTIMAL:
                    raise ValueError(
                        "optimal solution not found! " f"status={model.Status}"
                    )

                alpha_opt = np.array([a.x for a in alpha])
                alpha_hat_opt = np.array([a.x for a in alpha_hat])

                if budget is not None:
                    gamma_opt = gamma.x

                solution = (
                    (alpha_opt, alpha_hat_opt)
                    if budget is None
                    else (alpha_opt, alpha_hat_opt, gamma_opt)
                )

                return np.array(solution)

    def __repr__(self):
        return (
            f"GurobiSolver(time_limit={self.time_limit}, "
            + f"initial_values={self.initial_values})"
        )


class ReusableGurobiSolver(Solver):
    """Solver based on gurobi.

    Using this class requires that gurobi is installed and activated
    with a software key. The library is available at no cost for academic
    purposes (see
    https://www.gurobi.com/downloads/end-user-license-agreement-academic/).
    Alongside the library, also its interface to python should be installed,
    via the gurobipy package.
    """

    def __init__(self, problem="classification", time_limit=300, initial_values=None):
        """
        Build an object of type GurobiSolver.

        :param problem: Type of problem to be solved.
        :type problem: str ('classification' and 'regression' are allowed)
        :param time_limit: Maximum time (in seconds) before stopping iterative
          optimization, defaults to 10*60.
        :type time_limit: int
        :param initial_values: Initial values for variables of the optimization
          problem, defaults to None.
        :type initial_values: iterable of floats or None
        """
        super().__init__(problem)
        self.time_limit = time_limit
        self.initial_values = initial_values

        self.env = Env(empty=True)
        self.env.setParam("LogToConsole", 0)
        self.env.setParam("MIPGapAbs", 0.05)
        self.env.setParam("LogFile", "gurobi.log")

        self.model = None
        self.m = None

    def __init_base_model(self, X, y, C, kernel=GaussianKernel()):
        self.env.start()

        model = Model("svc", env=self.env)
        model.setParam("LogToConsole", 0)
        model.setParam("NonConvex", 2)
        model.setParam("TimeLimit", self.time_limit)

        model.addVars(list(range(self.m)), lb=0, ub=C, vtype=GRB.CONTINUOUS, name="alpha")

        model.update()

        vars = model.getVars()
        alpha = vars[:self.m]

        if self.initial_values is not None:
            for a, i in zip(alpha, self.initial_values[0]):
                a.start = i

            # if budget is not None:
            #    for g, i in zip(gamma, self.initial_values[0]):
            #        g.start = i

        obj = QuadExpr()

        obj.addTerms([1.0] * len(alpha), alpha)

        if kernel.precomputed:
            for i, j in it.product(range(self.m), range(self.m)):
                obj.add(
                    alpha[i] * alpha[j],
                    -0.5 * y[i] * y[j] * X[i][j],
                    )
        else:
            for i, j in it.product(range(self.m), range(self.m)):
                obj.add(
                    alpha[i] * alpha[j],
                    -0.5 * y[i] * y[j] * kernel.compute(X[i], X[j]),
                    )

        model.setObjective(obj, GRB.MAXIMIZE)

        constEqual = LinExpr()
        constEqual.add(quicksum(alpha * y), 1.0)

        model.addLConstr(constEqual, GRB.EQUAL, 0)

        #################################
        # budget part
        model.addVars(list(range(self.m)), vtype=GRB.BINARY, name="gamma")
        model.update()

        vars = model.getVars()

        alpha = vars[: self.m]
        gamma = vars[self.m:]

        for i, (a, g) in enumerate(zip(alpha, gamma)):
            const = QuadExpr()
            const.add(a, 1.0)
            model.addQConstr(const, GRB.LESS_EQUAL, C * g)

        # constraint actually limiting budget is set in __update_model_budget_constraint

        self.model = model

    def __update_model_budget_constraint(self, budget):
        assert self.model

        existing_constr = self.model.getConstrByName("budget_constraint")
        if existing_constr:
            existing_constr.setAttr(GRB.Attr.RHS, budget)
        else:
            vars = self.model.getVars()
            gamma = vars[self.m:]

            const = LinExpr()
            const.add(quicksum(gamma), 1.0)
            self.model.addLConstr(const, GRB.LESS_EQUAL, budget, name="budget_constraint")

    def solve_classification_problem(
            self, X, y, C=1, kernel=GaussianKernel(), budget=None
    ):
        """Optimize the classification-based optimization problem via gurobi.

        Build and solve the constrained optimization problem at the basis
        of SV classification, either classic or budgeted, using the gurobi
        API.

        :param X: Objects in training set.
        :type X: iterable
        :param y: Binary labels for the objects in `xs`.
        :type y: iterable
        :param C: constant managing the trade-off in joint complexity/error
                  optimization.
        :type C: float
        :param kernel: Kernel function to be used.
        :type kernel: :class:`mulearn.kernel.Kernel`
        :param budget: value of the budget (None in case of standard
                       classification).
        :type budget: float
        :raises: ValueError if C is non-positive, budget is specified as a
                 negative value or if X and y have different lengths.
        :returns: `list` -- optimal values for the independent variables
          of the problem."""

        m = len(X)
        assert m == len(y)
        assert C > 0

        if self.m is not None and self.m != m:
            # this happens when using cross validation. The dataset is the same, so we reuse the same solver object,
            # but fit is done on a different subset of points, so we rebuild the model.
            self.model = None

        self.m = m

        if not self.model:
            self.__init_base_model(X, y, C, kernel)

        if budget:
            self.__update_model_budget_constraint(budget)

        self.model.optimize()

        if self.model.Status != GRB.OPTIMAL:
            if self.model.Status != GRB.TIME_LIMIT or self.model.SolCount == 0:
                raise ValueError("no solution found! " f"status={self.model.Status}")

        self.obj_val = self.model.getObjective().getValue()

        vars = self.model.getVars()
        alpha = vars[: self.m]
        gamma = vars[self.m:]

        alpha_opt = np.array([a.x for a in alpha])
        if budget is not None:
            gamma_opt = np.array([g.x for g in gamma])

        solution = (alpha_opt, gamma_opt) if budget is not None else alpha_opt

        return np.array(solution), self.model.Status == GRB.OPTIMAL

    def solve_regression_problem(
            self, X, y, C=1, kernel=GaussianKernel(), epsilon=0.1, budget=None
    ):
        pass

    def __repr__(self):
        return (
                f"GurobiSolver(time_limit={self.time_limit}, "
                + f"initial_values={self.initial_values})"
        )
