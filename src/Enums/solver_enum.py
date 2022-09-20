"""
Author: Mélanie Gaillochet
Date: 2020-11-23
"""
import Methods.FullySupervised.solver
import Methods.SemiSupervised.solver


all_solvers = {
    "FullySupervised": Methods.FullySupervised.solver.Solver,
    "SemiSupervised": Methods.SemiSupervised.solver.Solver
}
