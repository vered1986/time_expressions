import json
import argparse
import gurobipy as gb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="en", type=str, required=False, help="Language code")
    parser.add_argument("--out_dir", default="output/lm_based/regex", type=str, required=False, help="Output directory")
    args = parser.parse_args()

    time_expressions = [line.strip().split("\t") for line in open(f"data/time_expressions/{args.lang}.txt")]

    with open(f"{args.out_dir}/{args.lang}_24.json") as f_in:
        grounding = json.load(f_in)

    labels = list(zip(*time_expressions))[0]
    labels = [l for l in labels if l != "before morning" and l in grounding.keys()]

    grounding = {exp: {int(hr): cnt for hr, cnt in values.items()}
                 for exp, values in grounding.items()
                 if exp != "before morning"}

    # Infer 24hr clock with ILP
    start_end = solve_ilp(grounding, labels)

    if start_end is not None:
        for exp in grounding.keys():
            grounding[exp].update(start_end[exp])

        with open(f"{args.out_dir}/{args.lang}_24.json", "w") as f_out:
            json.dump(grounding, f_out)


def solve_ilp(grounding, expressions):
    """
    Define and solve the ILP problem and determine the 24-hr clock time
    for each observation
    """
    params = create_ilp_model(grounding, expressions)
    model, start_variables, end_variables, cnt_by_var = params
    model.optimize()

    if model.status == gb.GRB.INFEASIBLE:
        print("Model is infeasible")
        model.computeIIS()
        model.write("model_iis.ilp")
        return None
    else:
        start_end = {exp: {} for exp in grounding.keys()}
        for exp in grounding.keys():
            start_end[exp]["start"] = start_variables[exp].getAttr("x")
            start_end[exp]["end"] = end_variables[exp].getAttr("x")

    return start_end


def create_ilp_model(grounding, expressions):
    """
    Create the ILP model representing the binary AM/PM variable
    """
    # Create a new model
    model = gb.Model("24HrClock")

    # Create the model variables
    start_variables, end_variables, counted_variables = create_variables(grounding, model)

    # Create the model constraints
    create_constraints(model, start_variables, end_variables, counted_variables, expressions)

    # Create the objective function: maximize number of observations fit inside each range
    relative_importance = {(exp, h): cnt for exp, vals in grounding.items() for h, cnt in vals.items()}
    create_objective(model, counted_variables, relative_importance)
    return model, start_variables, end_variables, relative_importance


def create_objective(model, counted_vars, relative_importance):
    """
    Create the objective function: maximize number of observations fit inside each range
    """
    num_obs_within_range = [relative_importance[(exp, h)] * counted_vars[(exp, h)][0]
                            for exp, h in relative_importance.keys()]
    model.setObjective(gb.quicksum(num_obs_within_range), gb.GRB.MAXIMIZE)


def create_variables(grounding, model):
    """
    Create the model variables:
    1. start and end hour for each expression
    2. start before end? binary variables for each expression
    3. whether a given hour is within range of a given expression
    """
    start_vars = {exp: model.addVar(vtype=gb.GRB.INTEGER, name=f"{exp}_start", lb=1, ub=24) for exp in grounding.keys()}
    end_vars = {exp: model.addVar(vtype=gb.GRB.INTEGER, name=f"{exp}_end", lb=1, ub=24) for exp in grounding.keys()}
    model.update()

    counted_exp_hs = {(exp, h):
                          (model.addVar(vtype=gb.GRB.BINARY, name=f"h_{exp}_{h}_counted"),
                           model.addVar(vtype=gb.GRB.BINARY, name=f"h_{exp}_{h}_as"),
                           model.addVar(vtype=gb.GRB.BINARY, name=f"h_{exp}_{h}_be"))
                      for exp, exp_grounding in grounding.items()
                      for h in exp_grounding.keys()}

    model.update()
    return start_vars, end_vars, counted_exp_hs


def create_constraints(model, start_vars, end_vars, counted_vars, expressions):
    """
    Create the model constraints
    """
    # Set the within-range variables
    for exp in expressions:
        for h in range(1, 25):
            if (exp, h) in counted_vars.keys():
                counted, hr_after_start, hr_before_end = counted_vars[(exp, h)]
                model.addConstr((hr_after_start == 1) >> (start_vars[exp] <= h), f"c_{exp}_{h}_as")
                model.addConstr((hr_before_end == 1) >> (h <= end_vars[exp]), f"c_{exp}_{h}_be")

                # At night, start > end
                if exp != "night":
                    model.addConstr((counted == 1) >> (hr_after_start + hr_before_end == 2))
                else:
                    model.addConstr((counted == 1) >> (hr_after_start + hr_before_end == 1))

        # During the day, start < end
        if exp != "night":
            model.addConstr(start_vars[exp] + 1 <= end_vars[exp], f"c_{exp}_min_duration")
        # At night, start > end
        else:
            model.addConstr(end_vars[exp] + 24 - start_vars[exp] >= 1, f"c_{exp}_min_duration")
            model.addConstr(start_vars["morning"] >= end_vars["night"], f"c_night_before_morning")

    # Sort expressions
    for i in range(len(expressions) - 1):
        model.addConstr(start_vars[expressions[i+1]] >= end_vars[expressions[i]],
                        f"c_{expressions[i]}_ends_before_{expressions[i+1]}_starts")


if __name__ == '__main__':
    main()
