import matplotlib.pyplot as plt


IMG_EXT = "png"


def plot_rewards(experiments, window, out_dir=None):
    """Generate plot of rewards for given experiments list and window controls the length
    of the window of previous episodes to use for average reward calculations.
    """
    window_color_list=['blue','red','green','black','purple']
    color_list=['lightblue','lightcoral','lightgreen', 'darkgrey', 'magenta']
    label_list=[]

    for i, exp in enumerate(experiments):
        if len(exp) == 4:
            (name, _, RL, data) = exp
        elif len(exp) == 3:
            (name, RL, data) = exp
        else:
            raise ValueError("Invalid experiment format")

        x_values=range(len(data['global_reward']))
        label_list.append(RL.display_name)
        y_values=data['global_reward']
        plt.plot(x_values, y_values, c=color_list[i],label=label_list[-1])
        plt.legend(label_list)

    if len(x_values) >= window : 
        for i, exp in enumerate(experiments):
            if len(exp) == 4:
                (name, _, RL, data) = exp
            elif len(exp) == 3:
                (name, RL, data) = exp
            else:
                raise ValueError("Invalid experiment format")

            x_values=range(window, 
                    len(data['med_rew_window'])+window)
            y_values=data['med_rew_window']
            plt.plot(x_values, y_values,
                    c=window_color_list[i])

    plt.title("Summed Reward", fontsize=16)
    plt.xlabel("Episode", fontsize=16)
    plt.ylabel("Reward", fontsize=16)
    plt.tick_params(axis='both', which='major',
                    labelsize=14)

    if out_dir is None:
        plt.show()
    else:
        file_name = out_dir / f"rewards_plot.{IMG_EXT}"
        plt.savefig(file_name, format=IMG_EXT, dpi=300)


def plot_length(experiments, out_dir=None):
    color_list=['blue','green','red','black','magenta']
    label_list=[]

    for i, exp in enumerate(experiments):
        if len(exp) == 4:
            (name, _, RL, data) = exp
        elif len(exp) == 3:
            (name, RL, data) = exp
        else:
            raise ValueError("Invalid experiment format")

        x_values=range(len(data['ep_length']))
        label_list.append(RL.display_name)
        y_values=data['ep_length']
        plt.plot(x_values, y_values, c=color_list[i],label=label_list[-1])
        plt.legend(label_list)

    plt.title("Path Length", fontsize=16)
    plt.xlabel("Episode", fontsize=16)
    plt.ylabel("Length", fontsize=16)
    plt.tick_params(axis='both', which='major',
                    labelsize=14)

    if out_dir is None:
        plt.show()
    else:
        file_name = out_dir / f"length_plot.{IMG_EXT}"
        plt.savefig(file_name, format=IMG_EXT, dpi=300)
