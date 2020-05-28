import matplotlib.pyplot as plt

from CaImaging.CellReg import *
from CaImaging.util import bin_transients, filter_sessions, ScrollPlot
from SEFL_utils import batch_load, load_cellmap, project_dirs
from CaImaging.Assemblies import lapsed_activation
from CaImaging.util import nan_array, get_transient_timestamps

project_df, project_path = project_dirs()


class LapsedAssemblies:
    def __init__(self, mouse):
        self.mouse = mouse
        self.sessions = filter_sessions(project_df, 'Mouse',
                                        mouse)

    def get_lapsed_assemblies(self, template_session_type,
                              lapsed_session_types, nullhyp='circ',
                              n_shuffles=1000,
                              percentile=99, plot=False):
        """
        Plots activation of all assemblies across multiple sessions in a
        somewhat messy way.

        :param mouse:
        :param sessions:
        :return:
        """
        # Get the sessions of interest.
        template_session = filter_sessions(self.sessions, 'Session',
                                           template_session_type)
        lapsed_sessions = filter_sessions(self.sessions, 'Session',
                                          lapsed_session_types)
        sessions = template_session.append(lapsed_sessions)

        # Load sessions and cell map.
        spikes = batch_load(sessions, 'S')
        cell_map = load_cellmap(sessions, detected='everyday')
        self.spikes = rearrange_neurons(cell_map, spikes)

        # Get ensemble activation.
        self.activations, self.patterns, _, _, _ = \
            lapsed_activation(self.spikes[0], self.spikes[1:],
                              nullhyp=nullhyp, n_shuffles=n_shuffles,
                              percentile=percentile, plot=plot)

        self.assembly_sessions = sessions
        self.n_sessions = len(sessions)
        self.n_assemblies = self.patterns.shape[0]

    def organize_assemblies(self):
        """
        Make a list (each element corresponding to a day) of
        (pattern, neuron, time) arrays.

        :return:
        """
        sort_order = np.argsort(np.abs(self.patterns), axis=1)
        self.sorted_patterns = np.sort(np.abs(self.patterns), axis=1)

        self.sorted_spike_times = []
        self.sorted_spike_arrs = []
        for session_spikes in self.spikes:
            # Preallocate an array.
            sorted_spike_times = []
            sorted_spike_arrs = []

            # For each assembly, sort based on weight.
            for assembly, order in enumerate(sort_order):
                sorted_spikes = session_spikes[order]
                spiking, _, bool_arr = get_transient_timestamps(
                    sorted_spikes)
                sorted_spike_times.append(spiking)
                sorted_spike_arrs.append(bool_arr)

            # Append to list of sessions' activities.
            self.sorted_spike_times.append(sorted_spike_times)
            self.sorted_spike_arrs.append(sorted_spike_arrs)

    def plot_single_lapsed_assembly(self, assembly_number):
        fig, axs = plt.subplots(self.n_sessions, 1, sharey='col',
                                figsize=(12,18))

        # Iterate through sessions.
        for ax, activations, spikes, session \
                in zip(axs,
                       self.activations,
                       self.sorted_spike_times,
                       self.assembly_sessions.Session):
            # Color spikes according to contribution.
            color_array = np.ones((len(spikes[assembly_number]), 3))
            color_array *= np.tile(
                self.sorted_patterns[assembly_number], (3, 1)).T
            color_array /= np.max(color_array)
            color_array = 1 - color_array

            # Alternatively, use alpha.
            # color_array = np.zeros((len(spikes[assembly_number]), 4))
            # color_array[:,3] = self.sorted_patterns[assembly_number]

            ax2 = ax.twinx()
            ax2.eventplot(spikes[assembly_number], colors=color_array)
            ax.plot(activations[assembly_number], alpha=0.7,
                     color='k')
            ax.set_zorder(ax2.get_zorder() + 1)
            ax.patch.set_visible(False)
            ax.set_ylabel('Ensemble activation (z)')
            ax2.set_ylabel('Neuron #')
            ax.set_title(session + f' assembly #{assembly_number}')

        ax.set_xlabel('Time (frames)')
        plt.show()
        pass


if __name__ == '__main__':
    # session_types = ['TraumaEnd', 'TraumaPost']
    session_types = ['TraumaEnd', 'Baseline', 'TraumaPost']
    AssemblyObj = LapsedAssemblies('pp5')
    AssemblyObj.get_lapsed_assemblies(session_types[0],
                                      session_types[1:], plot=False)
    AssemblyObj.organize_assemblies()
    for i in range(AssemblyObj.n_assemblies):
        AssemblyObj.plot_single_lapsed_assembly(i)

    pass
