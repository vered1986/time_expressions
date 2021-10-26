import json
import argparse
import gurobipy as gb

from src.common import draw_violin


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="en", type=str, required=False, help="Language code")
    parser.add_argument("--out_dir", default="output", type=str, required=False, help="Output directory")
    args = parser.parse_args()

    time_expressions = [line.strip().split("\t") for line in open(f"data/time_expressions/{args.lang}.txt")]
    labels = list(zip(*time_expressions))[0]
    labels = [l for l in labels if l != "before morning"]

    with open(f"output/{args.lang}.json") as f_in:
        grounding = json.load(f_in)

    grounding = {exp: {int(hr): cnt for hr, cnt in values.items()}
                 for exp, values in grounding.items()
                 if exp != "before morning"}

    # Infer 24hr clock with ILP
    result = solve_ilp(grounding)

    if result is not None:
        new_grounding, start_and_end_times = result

        # Night: add 24 to the hours < 12
        start_and_end_times["night"] = (start_and_end_times["night"][0], start_and_end_times["night"][1] + 24)
        new_grounding["night"] = {h + 24 if h < 12 else h: vals for h, vals in new_grounding["night"].items()}

        title = f"Grounding of Time Expressions in {args.lang}"
        ax = draw_violin(new_grounding, labels, start_and_end_times)
        fig = ax.get_figure()
        fig.savefig(f"output/{args.lang}.png")
        fig.suptitle(title, fontsize=24)
        fig.show()


def solve_ilp(grounding):
    """
    Define and solve the ILP problem and determine the 24-hr clock time
    for each observation
    """
    params = create_ilp_model(grounding)
    model, hr_variables, start_variables, end_variables, cnt_by_var = params
    model.optimize()

    if model.status == gb.GRB.INFEASIBLE:
        print("Model is infeasible")
        model.computeIIS()
        model.write("model_iis.ilp")
        return None
    else:
        new_grounding = {exp: {int(get_value(var)): cnt_by_var[var] for var in curr_vars}
                         for exp, curr_vars in hr_variables.items()}
        start_and_end_times = {exp: (start_variables[exp].getAttr("x"), end_variables[exp].getAttr("x"))
                               for exp in hr_variables.keys()}

    return new_grounding, start_and_end_times


def create_ilp_model(grounding):
    """
    Create the ILP model representing the binary AM/PM variable
    """
    # Create a new model
    model = gb.Model("24HrClock")

    # Create the model variables
    variables = create_variables(grounding, model)
    hr_variables, start_variables, end_variables, counted_variables = variables

    # Create the model constraints
    create_constraints(model, hr_variables, start_variables, end_variables, counted_variables)

    # Create the objective function: maximize number of observations fit inside each range
    # Normalize count per expression
    per_exp_counts = {exp: sum(counts.values()) for exp, counts in grounding.items()}
    cnt_by_var = {var: grounding[exp][int(var.VarName.split("_")[-1])]
                  for exp, vars in hr_variables.items() for var in vars}
    create_objective(model, hr_variables, counted_variables, cnt_by_var)
    return model, hr_variables, start_variables, end_variables, cnt_by_var


def create_objective(model, hr_vars, counted_vars, relative_importance):
    """
    Create the objective function: maximize number of observations fit inside each range
    """
    num_obs_within_range = [relative_importance[var] * counted_vars[var][0]
                            for curr_hr_vars in hr_vars.values()
                            for var in curr_hr_vars]
    model.setObjective(gb.quicksum(num_obs_within_range), gb.GRB.MAXIMIZE)


def create_variables(grounding, model):
    """
    Create the model variables:
    1. the 24hr for each (12hr, exp) pair
    2. start and end hour for each expression
    3. start before end? binary variables for each expression
    4. whether each variable is within range
    """
    hr_vars_by_exp_and_h = {exp: {
        h: model.addVar(vtype=gb.GRB.BINARY, name=f"h_{exp}_{h}") for h in exp_grounding.keys()}
        for exp, exp_grounding in grounding.items()}
    hr_variables = {exp: exp_vars.values() for exp, exp_vars in hr_vars_by_exp_and_h.items()}

    start_vars = {exp: model.addVar(vtype=gb.GRB.INTEGER, name=f"{exp}_start", lb=1, ub=24) for exp in grounding.keys()}
    end_vars = {exp: model.addVar(vtype=gb.GRB.INTEGER, name=f"{exp}_end", lb=1, ub=24) for exp in grounding.keys()}
    model.update()

    counted_variables = {h_exp_var:
                             (model.addVar(vtype=gb.GRB.BINARY, name=f"h_{exp}_{h}_counted"),
                              model.addVar(vtype=gb.GRB.BINARY, name=f"h_{exp}_{h}_as"),
                              model.addVar(vtype=gb.GRB.BINARY, name=f"h_{exp}_{h}_be"))
                         for exp, exp_vars in hr_vars_by_exp_and_h.items()
                         for h, h_exp_var in exp_vars.items()}

    model.update()
    return hr_variables, start_vars, end_vars, counted_variables


def get_value(var):
    """
    Get current value of variable based on the 12-hour clock hour
    and its binary value
    """
    h = int(var.VarName.split("_")[-1])
    return h + 12 * var.getAttr("x")


def create_constraints(model, hr_vars, start_vars, end_vars, counted_vars):
    """
    Create the model constraints
    """
    # Set the within-range variables, and make sure that range < 12 hours.
    for exp, curr_vars in hr_vars.items():
        for hr_var in curr_vars:
            h = int(hr_var.VarName.split("_")[-1])
            counted, hr_after_start, hr_before_end = counted_vars[hr_var]
            model.addConstr((hr_after_start == 1) >> (start_vars[exp] <= h + 12 * hr_var), f"c_{exp}_{h}_as")
            model.addConstr((hr_before_end == 1) >> (h + 12 * hr_var <= end_vars[exp]), f"c_{exp}_{h}_be")

            # At night, start > end
            if exp != "night":
                model.addConstr((counted == 1) >> (hr_after_start + hr_before_end == 2))
            else:
                model.addConstr((counted == 1) >> (hr_after_start + hr_before_end == 1))

        # During the day, start < end
        if exp != "night":
            model.addConstr(start_vars[exp] + 1 <= end_vars[exp], f"c_{exp}_min_duration")
        # At night, start > end, between 8 and 16 hours
        else:
            model.addConstr(end_vars[exp] + 24 - start_vars[exp] >= 8, f"c_{exp}_min_duration")
            model.addConstr(end_vars[exp] + 24 - start_vars[exp] <= 16, f"c_{exp}_max_duration")
            model.addConstr(start_vars["morning"] >= end_vars["night"], f"c_night_before_morning")

    # Sort expressions
    expressions = ["morning", "noon", "afternoon", "evening", "night"]
    for i in range(len(expressions) - 1):
        model.addConstr(start_vars[expressions[i+1]] >= end_vars[expressions[i]],
                        f"c_{expressions[i]}_ends_before_{expressions[i+1]}_starts")


if __name__ == '__main__':
    main()
