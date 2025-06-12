from servicemanager import startup_error

import module
from module import *
from docplex.cp.model import *
import docplex.cp.utils_visu as visu
import matplotlib.pyplot as plt
from pylab import rcParams
from ortools.sat.python import cp_model
import numpy as np
import copy

def cp_scheduling(_prob: Instance, time_limit=300, init_sol: Schedule = None):
    #TODO Subproblem에 이니셜 셋업 고려가 안되어 있음

    prob = copy.deepcopy(_prob)
    nbrOfJobs = prob.numJob
    jobs = [*range(0, nbrOfJobs)]
    nbrOfMachines = prob.numMch
    machines = [*range(0, nbrOfMachines)]
    processingTimes = prob.ptime
    setup_matrix = prob.setup

    mdl = CpoModel(name='cp_model')

    prob.job_list = sorted((job for job in prob.job_list), key=lambda j: j.ID)
    prob.machine_list = sorted((mch for mch in prob.machine_list), key=lambda m: m.ID)

    processing_itv_vars = [[mdl.interval_var(start=(prob.machine_list[m].available, INTERVAL_MAX), optional=True,
                                             size=processingTimes[m][j], name="interval_job{}_machine{}".format(j, m))
                            for m in machines] for j in jobs]

    for j in jobs:
        mdl.add(mdl.sum([mdl.presence_of(processing_itv_vars[j][m]) for m in machines]) == 1)

    sequence_vars = [mdl.sequence_var([processing_itv_vars[j][m] for j in jobs], types=[j for j in jobs],
                                      name="sequences_machine{}".format(m)) for m in machines]
    for m in machines:
        mdl.add(mdl.no_overlap(sequence_vars[m], setup_matrix[m], True))

    if module.OBJECTIVE_FUNCTION == 'T':
        objective = mdl.sum(
            [max(mdl.end_of(processing_itv_vars[j][m]) - prob.job_list[j].due, 0) for j in jobs for m in machines])
    elif module.OBJECTIVE_FUNCTION == 'C':
        objective = mdl.sum([mdl.end_of(processing_itv_vars[j][m]) for j in jobs for m in machines])
    else:
        objective = max([mdl.end_of(processing_itv_vars[j][m]) for j in jobs for m in machines])
    mdl.add(mdl.minimize(objective))

    if init_sol is not None:
        stp = mdl.create_empty_solution()  # add_interval_var_solution(var, presence=None, start=None, end=None, size=None, length=None)
        for bar in init_sol.bars:
            stp.add_interval_var_solution(processing_itv_vars[bar.job.ID][bar.machine], presence=True, start=bar.start, end= bar.end)
        mdl.set_starting_point(stp)

    msol = mdl.solve(TimeLimit=time_limit)  # log_output=True
    print("Solution: ")
    msol.print_solution()

    MA = {i: [] for i in machines}
    for i in jobs:
        for k in machines:
            if msol.get_var_solution(processing_itv_vars[i][k]).end != None:
                print('Job {0} on Machine {1} completed at {2}'.format(i, k, msol.get_var_solution(
                    processing_itv_vars[i][k]).end))
                job_i = prob.findJob(i)
                job_i.end = msol.get_var_solution(processing_itv_vars[i][k]).end
                MA[k].append(job_i)
    for k in machines:
        MA[k] = sorted((job for job in MA[k]), key=lambda m: m.end)
        machine = prob.findMch(k)
        for job in MA[k]:
            machine.process(job)

    obj = get_obj(prob)

    if msol.solution.objective_values[0] != obj:
        raise ValueError('Check Solution Result!')

    result = Schedule('CP_CPLEX', prob, obj=obj)
    result.print_schedule()
    result.comp_time = msol.process_infos['TotalSolveTime']
    result.status = msol.solve_status
    return result

def cp_scheduling_subprob(_prob: Instance, time_limit=300):
    #TODO Subproblem에 이니셜 셋업 고려가 안되어 있음

    prob = copy.deepcopy(_prob)
    nbrOfJobs = prob.numJob
    jobs = [*range(0, nbrOfJobs)]
    nbrOfMachines = prob.numMch
    machines = [*range(0, nbrOfMachines)]
    processingTimes = prob.ptime
    setup_matrix = prob.setup

    mdl = CpoModel(name='cp_model')

    prob.job_list = sorted((job for job in prob.job_list), key=lambda j: j.ID)
    prob.machine_list = sorted((mch for mch in prob.machine_list), key=lambda m: m.ID)

    processing_itv_vars = [[mdl.interval_var(optional=True, size=processingTimes[m][j], name="interval_job{}_machine{}".format(j, m)) for m in machines] for j in jobs]
    a = interval_var(length=10, start=5)
    a.size = 10
    a.start = 10

    for j in jobs:
        mdl.add(mdl.sum([mdl.presence_of(processing_itv_vars[j][m]) for m in machines]) == 1)

    sequence_vars = [mdl.sequence_var([processing_itv_vars[j][m] for j in jobs], types=[j for j in jobs],
                                      name="sequences_machine{}".format(m)) for m in machines]
    for m in machines:
        mdl.add(mdl.no_overlap(sequence_vars[m], setup_matrix[m], 1))

    for mch in prob.machine_list:
        if len(mch.assigned) != 0:
            for job in mch.assigned:
                mdl.add(mdl.presence_of(processing_itv_vars[job.ID][mch.ID]) == 1)
                idx = mch.assigned.index(job)
                if idx == 0:
                    mdl.add(first(sequence_vars[mch.ID], processing_itv_vars[job.ID][mch.ID]))

                if idx != (len(mch.assigned) - 1):
                    next = mch.assigned[idx+1]
                    mdl.add(previous(sequence_vars[mch.ID], processing_itv_vars[job.ID][mch.ID], processing_itv_vars[next.ID][mch.ID]))
                # elif idx == (len(mch.assigned) - 1) and len(mch.assigned) != 1: # Last One
                #     last = mch.assigned[idx]
                #     for j in jobs:
                #         if job.ID != j and j not in (assigned.ID for assigned in mch.assigned):
                #             mdl.add(previous(sequence_vars[mch.ID], processing_itv_vars[job.ID][mch.ID], processing_itv_vars[j][mch.ID]))

    if module.OBJECTIVE_FUNCTION == 'T':
        objective = mdl.sum(
            [max(mdl.end_of(processing_itv_vars[j][m]) - prob.job_list[j].due, 0) for j in jobs for m in machines])
    elif module.OBJECTIVE_FUNCTION == 'C':
        objective = mdl.sum([mdl.end_of(processing_itv_vars[j][m]) for j in jobs for m in machines])
    else:
        objective = max([mdl.end_of(processing_itv_vars[j][m]) for j in jobs for m in machines])

    mdl.add(mdl.minimize(objective))

    msol = mdl.solve(TimeLimit=time_limit)  # log_output=True
    print("Solution: ")
    msol.print_solution()
    if msol.solve_status != 'Optimal' and msol.solve_status != 'Feasible':
        print('check')

    MA = {i: [] for i in machines}
    for i in jobs:
        for k in machines:
            if prob.findJob(i).complete is False and msol.get_var_solution(processing_itv_vars[i][k]).end != None:
                print('Job {0} on Machine {1} completed at {2}'.format(i, k, msol.get_var_solution(
                    processing_itv_vars[i][k]).end))
                job_i = prob.findJob(i)
                job_i.end = msol.get_var_solution(processing_itv_vars[i][k]).end
                MA[k].append(job_i)
    for k in machines:
        MA[k] = sorted((job for job in MA[k]), key=lambda m: m.end)
        machine = prob.findMch(k)
        for job in MA[k]:
            if job not in machine.assigned:
                machine.process(job)

    obj = msol.solution.objective_values[0]
    obj = get_obj(prob)

    if msol.solution.objective_values[0] != obj:
        raise ValueError('Check Solution Result!')

    result = Schedule('CP_SUBPROB', prob, obj=obj)
    result.print_schedule()
    return result

def cp_scheduling_ortools(prob: Instance):
    jobs = [*range(0, prob.numJob)]
    machines = [*range(0, prob.numMch)]
    setup_matrix = prob.setup
    processingTimes = prob.ptime
    H = 100000000000000
    """ SJ = range(0, prob.numJob)
        max_s = np.array(setup_matrix).max()
        for i in SJ:
            M = M + max([row[i] for row in processingTimes])
            M = M + max_s
        H = M + max_s"""
    model = cp_model.CpModel()
    presence_vars = [[model.NewBoolVar(name="presence_machine{}_job{}".format(m, j)) for j in jobs] for m in machines]
    start_vars = [[model.NewIntVar(0, H, name="start_machine{}_job{}".format(m, j)) for j in jobs] for m in machines]
    end_vars = [[model.NewIntVar(0, H, name="end_machine{}_job{}".format(m, j)) for j in jobs] for m in machines]
    processing_itv_vars = [
        [model.NewOptionalIntervalVar(start=start_vars[m][j], end=end_vars[m][j], size=processingTimes[m][j],
                                      is_present=presence_vars[m][j], name="interval_machine{}_job{}".format(m, j))
         for j in jobs] for m in machines]
    for m in machines:
        model.AddNoOverlap(processing_itv_vars[m])

    presence_lit = [[[model.NewBoolVar('%i and %i in %i' % (j1, j2, m)) for j2 in jobs] for j1 in jobs] for m in
                    machines]
    precedence = [[[model.NewBoolVar('%i -> %i in %i' % (j1, j2, m)) for j2 in jobs] for j1 in jobs] for m in machines]
    for m in machines:
        for j1 in jobs:
            for j2 in jobs:
                if j1 != j2:
                    lit12 = precedence[m][j1][j2]
                    lit21 = precedence[m][j2][j1]
                    model.Add(start_vars[m][j2] >= end_vars[m][j1] + setup_matrix[m][j1][j2]).OnlyEnforceIf(lit12,
                                                                                                            presence_vars[
                                                                                                                m][j1],
                                                                                                            presence_vars[
                                                                                                                m][j2])
                    model.Add(start_vars[m][j1] >= end_vars[m][j2] + setup_matrix[m][j2][j1]).OnlyEnforceIf(lit21,
                                                                                                            presence_vars[
                                                                                                                m][j1],
                                                                                                            presence_vars[
                                                                                                                m][j2])
                    model.AddBoolOr(lit12, lit21, presence_vars[m][j1].Not(), presence_vars[m][j2].Not())
    for j in jobs:
        alt_intvs = []
        for m in machines:
            alt_intvs.append(presence_vars[m][j])
        model.Add(cp_model.LinearExpr.Sum(alt_intvs) == 1)

    objective = cp_model.LinearExpr.Sum([end_vars[m][j] for j in jobs for m in machines])
    model.Minimize(objective)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 300
    # solver.parameters.enumerate_all_solutions = True
    solver.parameters.log_search_progress = True
    status = solver.Solve(model)
    if status in [cp_model.OPTIMAL]:
        return [solver, "OPTIMAL"]
    elif status in [cp_model.FEASIBLE]:
        return [solver, "FEASIBLE"]
    else:
        return [solver, "no"]
