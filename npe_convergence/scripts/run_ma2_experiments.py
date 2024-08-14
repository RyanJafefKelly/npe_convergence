from npe_convergence.scripts.run_experiment import run_experiment
from npe_convergence.scripts.run_ma2_identifiable import run_ma2_identifiable

def run_ma2_experiments():
    kl_mat = run_experiment(run_ma2_identifiable)
    return None


if __name__ == "__main__":
    run_ma2_experiments()