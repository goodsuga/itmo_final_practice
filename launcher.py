from experiment_boston import run_boston


RUN_BOSTON = True

if __name__ == "__main__":
    if RUN_BOSTON:
        for i in range(40):
            run_boston()
