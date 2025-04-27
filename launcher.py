from experiment_boston import run_boston
from experiment_bikeshare import run_bikeshare
from experiment_mushrooms import run_mushrooms
from experiment_concrete import run_concrete
from experiment_bikeshare_gumbel import run_bikeshare_comp

RUN_BOSTON = False
RUN_BIKESHARE = False
RUN_MUSHROOMS = False
RUN_CONCRETE = False
RUN_COMP = True

if __name__ == "__main__":
    if RUN_BOSTON:
        for i in range(40):
            run_boston()
    if RUN_BIKESHARE:
        for i in range(40):
            run_bikeshare()
    if RUN_MUSHROOMS:
        for i in range(40):
            run_mushrooms()
    if RUN_CONCRETE:
        for i in range(40):
            run_concrete()
    if RUN_COMP:
        for i in range(40):
            run_bikeshare_comp()
