import matplotlib.pyplot as plt
import matplotlib.cm as cm
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

    def get_lapsed_assemblies(self, session_types, nullhyp='circ',
                              n_shuffles=1000, use_bool=True,
                              percentile=99, plot=False):
        """
        Plots activation of all assemblies across multiple sessions in a
        somewhat messy way.

        session_types: list of strs
            Sessions to analyze. The template session must be first.
        """
        # Get the sessions of interest.
        template_session = filter_sessions(self.sessions, 'Session',
                                           session_types[0])
        lapsed_sessions = filter_sessions(self.sessions, 'Session',
                                          session_types[1:])
        sessions = template_session.append(lapsed_sessions)

        # Load sessions and cell map.
        S = batch_load(sessions, 'S')
        cell_map = load_cellmap(sessions, detected='everyday')
        self.S = rearrange_neurons(cell_map, S)

        # Get ensemble activation.
        assemblies, spikes, = \
            lapsed_activation(self.S[0], self.S[1:],
                              nullhyp=nullhyp, n_shuffles=n_shuffles,
                              percentile=percentile, plot=plot,
                              use_bool=use_bool)

        self.activations = assemblies['activations']
        self.patterns = assemblies['patterns']
        self.S_normalized = spikes['S_normalized']
        self.bool_arr = spikes['bool_arr']
        self.spike_times = spikes['spike_times']
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
        self.sorted_bool_arrs = []
        self.sorted_S = []
        for spike_times, bool_arr, S in zip(self.spike_times,
                                            self.bool_arr,
                                            self.S_normalized):
            # Preallocate an array.
            sorted_spike_times = []
            sorted_bool_arrs = []
            sorted_S = []

            # For each assembly, sort based on weight.
            for order in sort_order:
                sorted_spike_times.append([spike_times[n] for n in order])
                sorted_bool_arrs.append(bool_arr[order])
                sorted_S.append(S[order])

            # Append to list of sessions' activities.
            self.sorted_spike_times.append(sorted_spike_times)
            self.sorted_bool_arrs.append(sorted_bool_arrs)
            self.sorted_S.append(sorted_S)

    def plot_single_lapsed_assembly(self, assembly_number,
                                    cmap='copper_r'):
        fig, axs = plt.subplots(self.n_sessions, 1, sharey='col',
                                figsize=(12,18))

        cmap_arr = cm.get_cmap(cmap)
        # Iterate through sessions.
        for ax, activations, spikes, session \
                in zip(axs,
                       self.activations,
                       self.sorted_spike_times,
                       self.assembly_sessions.Session):
            # Color S according to contribution.
            color_array = [cmap_arr(p)
                           for p in self.sorted_patterns[assembly_number]]
            color_array = np.asarray(color_array)

            # Alternatively, use alpha.
            # color_array = np.zeros((len(S[assembly_number]), 4))
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

        return fig, axs


    def plot_all_assemblies(self):
        figs = []
        for n in range(self.n_assemblies):
            fig = self.plot_single_lapsed_assembly(n)[0]
            figs.append(fig)


def get_assemblies_by_mouse(mice, sessions, use_bool=True,
                            replace_missing_data=False):
    all_mice = {mouse: LapsedAssemblies(mouse) for mouse in mice}

    for mouse_assemblies in all_mice.values():
        print(f'Analyzing {mouse_assemblies.mouse}...')
        if replace_missing_data:
            sessions_ = handle_missing_data(mouse_assemblies.mouse,
                                            sessions)
        else:
            sessions_ = sessions
        try:
            mouse_assemblies.get_lapsed_assemblies(sessions_,
                                                   plot=False,
                                                   use_bool=use_bool)

            mouse_assemblies.organize_assemblies()
        except:
            print(f'{mouse_assemblies.mouse} failed.')

    return all_mice


def plot_reactivations(all_mice_assemblies, session_numbers):
    # Initialize dictionary.
    activations = {'template': [],
                   'post': [],
                   'mouse': [],
                   'condition': []}

    # For all sessions, take the mean of ensemble activation.
    for assembly_class in all_mice_assemblies.values():
        try:
            for session_number, epoch in zip(session_numbers, ['template', 'post']):
                activity = np.mean(assembly_class.activations[session_number], axis=1)
                activations[epoch].extend(activity)

            # Get mouse names and conditions (trauma or control).
            mouse_name = np.repeat(assembly_class.mouse, assembly_class.n_assemblies)
            condition = np.repeat(assembly_class.sessions.Group.iloc[0], assembly_class.n_assemblies)

            activations['mouse'].extend(mouse_name)
            activations['condition'].extend(condition)

        except:
            pass

    # Get colors and also boolean condition array.
    activations['colors'] = [[1, 0, 0] if c == 'trauma' else [0, 0, 1]
                             for c in activations['condition']]
    activations['trauma'] = [1 if c == 'trauma' else 0
                             for c in activations['condition']]

    # Array-ify.
    activations = {key: np.asarray(item)
                   for key, item in activations.items()}

    # Plot control and trauma ensemble activations for template
    # compared to a lapsed session.
    fig, ax = plt.subplots()
    trauma_animals = activations['trauma'] == 1
    trauma = ax.scatter(activations['template'][trauma_animals],
                        activations['post'][trauma_animals],
                        c=activations['colors'][trauma_animals],
                        label='Trauma')
    control = ax.scatter(activations['template'][~trauma_animals],
                         activations['post'][~trauma_animals],
                         c=activations['colors'][~trauma_animals],
                         label='Control')

    # Label the animals.
    for i, (txt, x, y) in enumerate(zip(activations['mouse'],
                                        activations['template'],
                                        activations['post'])):
        ax.annotate(txt, (x, y))

    ax.set_xlabel('Template session activation')
    ax.set_ylabel('Post session activation')
    ax.axis('equal')
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]

    # now plot both limits against eachother
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.legend()

    return activations


def handle_missing_data(mouse, sessions):
    if mouse == 'pp2':
        if 'TraumaEnd' in sessions:
            if 'TraumaStart' not in sessions:
                print('Replaced TraumaEnd with TraumaStart for pp2')
                sessions = ['TraumaStart' if x == 'TraumaEnd'
                             else x
                             for x in sessions]
            else:
                sessions.remove('TraumaEnd')
                print('Removed TraumaEnd. '
                      'New template session for pp2 is ' + sessions[0])
    if mouse == 'pp7':
        if 'TraumaStart' in sessions:
            if 'TraumaEnd' not in sessions:
                print('Replaced TraumaStart with TraumaEnd for pp7')
                sessions = ['TraumaEnd' if x == 'TraumaStart'
                            else x
                            for x in sessions]
            else:
                sessions.remove('TraumaStart')
                print('Removed TraumaStart. '
                      'New template session for pp7 is ' + sessions[0])

    return sessions

if __name__ == '__main__':
    session_types = ['TraumaEnd', 'TraumaPost']
    AssemblyObj = LapsedAssemblies('pp7')
    AssemblyObj.get_lapsed_assemblies(session_types, plot=False,
                                      use_bool=True)
    AssemblyObj.organize_assemblies()
    for i in range(AssemblyObj.n_assemblies):
        AssemblyObj.plot_single_lapsed_assembly(i)

    mice = ['pp1',
            'pp2',
            'pp4',
            'pp5',
            'pp6',
            'pp7',
            'pp8']

    sessions = ['TraumaEnd', 'TraumaPost']
    assemblies = get_assemblies_by_mouse(mice,
                                         sessions = sessions)

    pass
