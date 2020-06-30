import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['text.usetex'] = False
plt.rcParams.update({'font.size': 12})
import matplotlib.cm as cm
from CaImaging.CellReg import *
from CaImaging.util import bin_transients, filter_sessions, ScrollPlot
from SEFL_utils import batch_load, load_cellmap, project_dirs
from CaImaging.Assemblies import lapsed_activation, \
    preprocess_multiple_sessions
from CaImaging.util import nan_array, get_transient_timestamps, sync_data
import pickle as pkl
from datetime import datetime

project_df, project_path = project_dirs()


class LapsedAssemblies:
    def __init__(self, mouse):
        self.mouse = mouse
        self.sessions = filter_sessions(project_df, 'Mouse',
                                        mouse)

    def get_lapsed_assemblies(self, session_types, nullhyp='circ',
                              n_shuffles=1000, use_bool=True,
                              smooth_factor=5, percentile=99,
                              z_method='global'):
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

        try:
            data = self.load_and_verify(sessions)
            for key in data:
                setattr(self, key, data[key])

            return
        except:
            pass

        # Load sessions and cell map.
        S = batch_load(sessions, 'S')
        cell_map = load_cellmap(sessions, detected='everyday')
        self.S = rearrange_neurons(cell_map, S)
        self.spikes = \
            preprocess_multiple_sessions(self.S,
                                         smooth_factor=smooth_factor,
                                         z_method=z_method,
                                         use_bool=use_bool)

        # Get ensemble activation.
        self.assemblies= \
            lapsed_activation(self.spikes['processed'], nullhyp=nullhyp,
                              n_shuffles=n_shuffles,
                              percentile=percentile)

        self.assembly_sessions = sessions
        self.n_sessions = len(sessions)
        self.n_assemblies = self.assemblies['patterns'].shape[0]

    def organize_assemblies(self):
        """
        Make a list (each element corresponding to a day) of
        (pattern, neuron, time) arrays. Sort by order of weight in
        the assembly models.

        :return:
        """
        sort_order = np.argsort(np.abs(self.assemblies['patterns']), axis=1)
        self.sorted_data = dict()

        self.sorted_data['patterns'] = np.sort(np.abs(self.assemblies['patterns']), axis=1)

        self.sorted_data['spike_times'] = []
        self.sorted_data['bool_arrs'] = []
        self.sorted_data['S'] = []
        for spike_times, bool_arr, S in zip(self.spikes['spike_times'],
                                            self.spikes['bool_arrs'],
                                            self.spikes['S']):
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
            self.sorted_data['spike_times'].append(sorted_spike_times)
            self.sorted_data['bool_arrs'].append(sorted_bool_arrs)
            self.sorted_data['S'].append(sorted_S)


    def plot_single_lapsed_assembly(self, assembly_number,
                                    cmap='copper_r', fps=15):
        fig, axs = plt.subplots(self.n_sessions, 1, sharey='col',
                                figsize=(12, 18))

        cmap_arr = cm.get_cmap(cmap)
        # Iterate through sessions.
        for ax, activations, spikes, session \
                in zip(axs,
                       self.assemblies['activations'],
                       self.sorted_data['spike_times'],
                       self.assembly_sessions.Session):
            # Color S according to contribution.
            color_array = [cmap_arr(p)
                           for p in
                           self.sorted_data['patterns'][assembly_number]]
            color_array = np.asarray(color_array)

            # Get time vector.
            n_samples = len(activations[assembly_number])
            t = range(n_samples)
            t_labels = np.linspace(0, np.round(n_samples, -2), 5)

            # Alternatively, use alpha.
            # color_array = np.zeros((len(S[assembly_number]), 4))
            # color_array[:,3] = self.sorted_patterns[assembly_number]

            ax2 = ax.twinx()
            ax2.eventplot(spikes[assembly_number], colors=color_array)
            ax.plot(t, activations[assembly_number], alpha=0.7,
                    color='k')
            ax.set_xticks(t_labels)
            ax.set_xticklabels(np.round(t_labels / fps))

            ax.set_zorder(ax2.get_zorder() + 1)
            ax.patch.set_visible(False)
            ax.set_ylabel('Ensemble activation (AU)')
            ax2.set_ylabel('Neuron #, sorted by ensemble membership weight')
            ax.set_title(session + f' assembly #{assembly_number}')

        # Where we have data, fill in regions where the mouse is
        # stationary.
        ymax = np.max([ax.get_ylim()[1] for ax in axs])
        for ax, behavior in zip(axs, self.behavior):
            try:
                n_samples = len(behavior)
                t = range(n_samples)

                freezing = behavior.Freezing
                ax.fill_between(t, 0, ymax, where=freezing>0,
                                color='r', alpha=0.5)
            except:
                pass

        ax.set_xlabel('Time (s)')
        plt.show()

        return fig, axs


    def plot_all_assemblies(self):
        figs = []
        for n in range(self.n_assemblies):
            fig = self.plot_single_lapsed_assembly(n)[0]
            figs.append(fig)

        return figs


    def align_to_behavior(self):
        sessions = self.assembly_sessions
        miniscope_cams = list(self.assembly_sessions['MiniscopeCam'])
        behavior_cams = list(self.assembly_sessions['BehaviorCam'])

        # Build lists of file names for alignment.
        get_behav = lambda path: glob.glob(os.path.join(path,
                                                        '*Output.csv'))
        behavior_csv_fnames = [get_behav(session)[0]
                               if get_behav(session)
                               else None
                               for session in sessions.Path]
        minian_fnames = [path for path in sessions.Path]
        timestamp_fnames = [os.path.join(session, 'timestamp.dat')
                            for session in sessions.Path]

        # Align.
        self.behavior = []
        for timestamp_fname, behavior_fname, minian_fname, \
            miniscope_cam, behavior_cam \
                in zip(timestamp_fnames, behavior_csv_fnames,
                       minian_fnames, miniscope_cams, behavior_cams):
            try:
                self.behavior.append(sync_data(behavior_fname,
                                               minian_fname,
                                               timestamp_fname,
                                               miniscope_cam=miniscope_cam,
                                               behav_cam=behavior_cam)[0])
            except:
                self.behavior.append(None)
                print(minian_fname + ' failed to sync behavior.')


    def load_and_verify(self, sessions):
        pickled_fnames = glob.glob(os.path.join(sessions.Path.iloc[0],
                                                'assembly_data*.pkl'))[::-1]
        for pickled_data in pickled_fnames:
            with open(pickled_data, 'rb') as file:
                data = pkl.load(file)

            if all(sessions['Path'].values == data['assembly_sessions']['Path'].values):
                print('Using ' + pickled_data + '.')
                return data
            else:
                print(pickled_data + ' did not match requested sessions.')

        return None



    def save_data(self, save_path=None):
        data_as_dict = vars(self)
        now = datetime.now()
        dt_string = now.strftime("%m_%d_%Y")

        if save_path is None:
            save_path = os.path.join(self.assembly_sessions.Path.iloc[0],
                                     'assembly_data_' +
                                     dt_string +
                                     '.pkl')

        with open(save_path, 'wb') as file:
            pkl.dump(data_as_dict, file)


def get_assemblies_by_mouse(mice, sessions, use_bool=True,
                            smooth_factor=5, z_method='global',
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
                                                   use_bool=use_bool,
                                                   smooth_factor=smooth_factor,
                                                   z_method=z_method)

            mouse_assemblies.organize_assemblies()
            mouse_assemblies.align_to_behavior()

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
                activity = np.mean(assembly_class.assemblies['activations'][session_number], axis=1)
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
    AssemblyObj = LapsedAssemblies('pp5')
    AssemblyObj.get_lapsed_assemblies(session_types,
                                      use_bool=True,
                                      smooth_factor=5,
                                      z_method='global')
    AssemblyObj.organize_assemblies()
    AssemblyObj.align_to_behavior()
    AssemblyObj.plot_all_assemblies()

    mice = ['pp1',
            'pp2',
            'pp4',
            'pp5',
            'pp6',
            'pp7',
            'pp8']

    sessions = ['TraumaStart', 'TraumaPost']
    assemblies = get_assemblies_by_mouse(mice,
                                         sessions = sessions)

    x = assemblies['trauma'] / 5
    y = assemblies['post']

    def rand_jitter(arr):
        stdev = .1 * (max(arr) - min(arr))
        return arr + np.random.randn(len(arr)) * stdev


    fig, ax = plt.subplots()
    ax.scatter(rand_jitter(x), y)
    ax.set_xticks([0, 0.2])
    ax.set_xticklabels(['Control', 'Trauma'])
    ax.set_ylabel('Mean reactivation rate')
    ax.axis('equal')

    pass
