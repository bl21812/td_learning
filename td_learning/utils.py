import argparse
import pickle

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

EXP_FILE = "experiments.pkl"
IMG_EXT = "png"


def export_pickle(experiments, out_dir):
    with open(out_dir / EXP_FILE, 'wb') as f:
        exps = [(name, RL.display_name, data) for name, _, RL, data in experiments]
        pickle.dump(exps, f)


def load_pickle(in_dir):
    with open(in_dir / EXP_FILE, 'rb') as f:
        experiments = pickle.load(f)
    return experiments


def moving(arr, window, fn=np.median):
    windowed_arr = np.lib.stride_tricks.sliding_window_view(arr, window_shape=window)
    return fn(windowed_arr, axis=-1)


def plot_experiments(experiments, key, title, ylab, mode, ax=None, window=None, window_fn=np.median, out_dir=None):
    """Generate plot of rewards for given experiments list and window controls the length
    of the window of previous episodes to use for average reward calculations.
    """
    if mode not in ['main', 'window', 'both']:
        print(f"Invalid mode: {mode}")

    if mode in ['window', 'both'] and window is None:
        print(f"WARNING: Window size {window} is not provided")

    colors = ['lightblue','lightcoral','lightgreen', 'darkgrey', 'magenta']
    dark_colors = ['blue', 'red', 'green', 'black', 'purple']

    if len(experiments) > 5:
        print("WARNING: MORE THAN 5 EXPERIMENTS WILL NOT BE PLOTTED")

    combine_plots = True
    if ax is None:
        combine_plots = False
        fig, ax = plt.subplots(1, 1)

    for exp, c, dc in zip(experiments, colors, dark_colors):
        if len(exp) == 4:
            (name, _, RL, data) = exp
            display_name = RL.display_name
        elif len(exp) == 3:
            (name, display_name, data) = exp
        else:
            raise ValueError("Invalid experiment format")
        
        num_episodes = data[key].shape[0]
        episodes = np.arange(num_episodes)

        if mode == 'main' or mode == 'both':
            line_main, = ax.plot(episodes, data[key], c=dc if mode == 'main' else c)

        if mode == 'window' or mode == 'both':
            if window is None:
                print(f"WARNING: Window size {window} is not provided")
                continue
            if num_episodes < window:
                print(f"WARNING: Window size {window} is larger than number of episodes {num_episodes}")
                continue

            window_arr = moving(data[key], window, window_fn)
            line_window, = ax.plot(episodes[-window_arr.shape[0]:], window_arr, c = dc)

        (line_main if mode == "main" else line_window).set_label(display_name)

        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Episode", fontsize=16)
        ax.set_ylabel(ylab, fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.legend()

    if not combine_plots:
        if out_dir is None:
            plt.show()
        else:
            file_name = out_dir / f"{title}.{IMG_EXT}"
            plt.savefig(file_name, format=IMG_EXT)


def window_fn_type(name):
    if name == "mean":
        return np.mean
    elif name == "var":
        return np.var
    elif name == "med":
        return np.median
    else:
        raise ValueError(f"Invalid window function: {name}")


def main():
    """
    Load experiments from a pickle file and generate plots.
    
    Args:
    input_dir (str): Path to the directory containing experiments.pkl
    output_dir (str, optional): Path to save the generated plots. If None, plots will be displayed.
    """

    parser = argparse.ArgumentParser(description="Generate plots from experiments.pkl")
    parser.add_argument("input_dir", help="Directory containing experiments.pkl")
    parser.add_argument("--output_dir", help="Directory to save generated plots")
    parser.add_argument("--window", type=int, default=250, help="Window size for moving average")
    parser.add_argument("--combine_plots", action="store_true", help="Plot things")
    parser.add_argument("--mode", type=str, default="both", choices=["both", "main", "window"], help="Plotting mode")
    parser.add_argument("--window_fn", type=window_fn_type, default="med", choices=[np.mean, np.var, np.median], help="Function to apply to window")
    args = parser.parse_args()
    
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = None

    experiments = load_pickle(Path(args.input_dir))

    if args.combine_plots:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(25, 5))
    else:
        ax1 = ax2 = ax3 = ax4 = None

    plot_experiments(experiments, "global_reward", "(a) Summed Rewards", "Reward",
                    args.mode, ax1, window=args.window, out_dir=out_dir, window_fn=args.window_fn)
    plot_experiments(experiments, 'ep_length', "(b) Total Path Length", "Length",
                    args.mode, ax2, window=args.window, out_dir=out_dir, window_fn=args.window_fn)
    plot_experiments(experiments, "pitfall", "(c) Number of Falls into Pits", "Number of Falls",
                    "window", ax3, window=args.window, out_dir=out_dir, window_fn=np.mean)
                    # "window", ax3, window=args.window, out_dir=out_dir, window_fn=args.window_fn)
    plot_experiments(experiments, 'wallbump', "(d) Number of Bumps into Walls", "Number of Bumps",
                    args.mode, ax4, window=args.window, out_dir=out_dir, window_fn=args.window_fn)
    

    if args.combine_plots:
        fig.suptitle("Figure 1: Task 1 Experiment Results, $\epsilon=0.1$, $\\alpha=0.1$, $\gamma=0.9$, Average over Window Size=25")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
