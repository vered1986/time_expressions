import json
import argparse
import gurobipy as gb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="en", type=str, required=False, help="Language code")
    parser.add_argument("--out_dir", default="output/lm_based/", type=str, required=False, help="Output directory")
    args = parser.parse_args()

    time_expressions = [line.strip().split("\t") for line in open(f"data/time_expressions/{args.lang}.txt")]

    with open(f"{args.out_dir}/{args.lang}_start_end.json") as f_in:
        grounding = json.load(f_in)

    labels = list(zip(*time_expressions))[0]
    labels = [l for l in labels if l != "before morning" and l in grounding.keys()]

    grounding = {exp: {edge: {int(hr): cnt for hr, cnt in per_edge.items()}
                       for edge, per_edge in per_exp.items()}
                       for exp, per_exp in grounding.items()
                 if exp != "before morning"}

    # Infer 24hr clock with ILP
    start_end = solve_ilp(grounding, labels)

    if start_end is not None:
        for exp in grounding.keys():
            grounding[exp].update(start_end[exp])

        with open(f"{args.out_dir}/{args.lang}_start_end.json", "w") as f_out:
            json.dump(grounding, f_out)


def solve_ilp(grounding, expressions):
    """
    Define and solve the ILP problem and determine the 24-hr clock time
    for each observation
    """
    model, start_variables, end_variables = create_ilp_model(grounding, expressions)
    model.optimize()

    if model.status == gb.GRB.INFEASIBLE:
        print("Model is infeasible")
        model.computeIIS()
        model.write("model_iis.ilp")
        return None
    else:
        start_end = {exp: {} for exp in grounding.keys()}
        for exp in grounding.keys():
            start_end[exp]["start"] = start_variables[exp][0].getAttr("x")
            start_end[exp]["end"] = end_variables[exp][0].getAttr("x")

    return start_end


def create_ilp_model(grounding, expressions):
    """
    Create the ILP model representing the binary AM/PM variable
    """
    # Create a new model
    model = gb.Model("24HrClock")

    # Create the model variables
    start_vars, end_vars = create_variables(grounding, model)

    # Create the model constraints
    create_constraints(model, start_vars, end_vars, expressions)

    # Create the objective function
    create_objective(model, grounding, start_vars, end_vars)
    return model, start_vars, end_vars


def create_objective(model, grounding, start_vars, end_vars):
    """
    Create the objective function: maximize score of start and end time
    """
    scores = [grounding[exp][edge][h] * curr_vars[exp][1][h]
              for edge, curr_vars in zip(["start", "end"], [start_vars, end_vars])
              for exp in grounding.keys() for h in range(24)]
    model.setObjective(gb.quicksum(scores), gb.GRB.MAXIMIZE)


def create_variables(grounding, model):
    """
    Create the model variables: start and end hour for each expression
    """
    start_vars = {exp: (model.addVar(vtype=gb.GRB.INTEGER, name=f"{exp}_start", lb=1, ub=24),
                        [model.addVar(vtype=gb.GRB.BINARY, name=f"{exp}_start_{h}") for h in range(1, 25)])
                        for exp in grounding.keys()}
    end_vars = {exp: (model.addVar(vtype=gb.GRB.INTEGER, name=f"{exp}_end", lb=1, ub=24),
                      [model.addVar(vtype=gb.GRB.BINARY, name=f"{exp}_end_{h}") for h in range(1, 25)])
                      for exp in grounding.keys()}
    model.update()
    return start_vars, end_vars


def create_constraints(model, start_vars, end_vars, expressions):
    """
    Create the model constraints
    """
    for exp in expressions:
        # During the day, start < end
        if exp != "night":
            model.addConstr(start_vars[exp][0] + 1 <= end_vars[exp][0], f"c_{exp}_min_duration")
        # At night, start > end
        else:
            model.addConstr(end_vars[exp][0] + 24 - start_vars[exp][0] >= 1, f"c_{exp}_min_duration")
            model.addConstr(start_vars["morning"][0] >= end_vars["night"][0], f"c_night_before_morning")

        # Helper variables
        for edge, curr_vars in zip(["start", "end"], [start_vars, end_vars]):
            for h in range(24):
                model.addConstr((curr_vars[exp][1][h] == 1) >> (curr_vars[exp][0] == h))

    # Sort expressions
    for i in range(len(expressions) - 1):
        model.addConstr(start_vars[expressions[i+1]][0] >= end_vars[expressions[i]][0],
                        f"c_{expressions[i]}_ends_before_{expressions[i+1]}_starts")


if __name__ == '__main__':
    main()
