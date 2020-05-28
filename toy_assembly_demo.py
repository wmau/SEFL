from sklearn.decomposition import PCA
from scipy import stats
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import warnings

def toyExample(assemblies, nneurons=10, nbins=1000, rate=1.):
    np.random.seed()

    actmat = np.random.poisson(rate, nneurons * nbins).reshape(nneurons, nbins)
    assemblies.actbins = [None] * len(assemblies.membership)
    for (ai, members) in enumerate(assemblies.membership):
        members = np.array(members)
        nact = int(nbins * assemblies.actrate[ai])
        actstrength_ = rate * assemblies.actstrength[ai]

        actbins = np.argsort(np.random.rand(nbins))[0:nact]

        actmat[members.reshape(-1, 1), actbins] = \
            np.ones((len(members), nact)) + actstrength_

        assemblies.actbins[ai] = np.sort(actbins)

    return actmat


class toyassemblies:

    def __init__(self, membership, actrate, actstrength):
        self.membership = membership
        self.actrate = actrate
        self.actstrength = actstrength


def marcenkopastur(significance):
    nbins = significance.nbins
    nneurons = significance.nneurons
    tracywidom = significance.tracywidom

    # calculates statistical threshold from Marcenko-Pastur distribution
    q = float(nbins) / float(nneurons)  # note that silent neurons are counted too
    lambdaMax = pow((1 + np.sqrt(1 / q)), 2)
    lambdaMax += tracywidom * pow(nneurons, -2. / 3)  # Tracy-Widom correction

    return lambdaMax


def getlambdacontrol(zactmat_):
    significance_ = PCA()
    significance_.fit(zactmat_.T)
    lambdamax_ = np.max(significance_.explained_variance_)

    return lambdamax_


def binshuffling(zactmat, significance):
    np.random.seed()

    lambdamax_ = np.zeros(significance.nshu)
    for shui in range(significance.nshu):
        zactmat_ = np.copy(zactmat)
        for (neuroni, activity) in enumerate(zactmat_):
            randomorder = np.argsort(np.random.rand(significance.nbins))
            zactmat_[neuroni, :] = activity[randomorder]
        lambdamax_[shui] = getlambdacontrol(zactmat_)

    lambdaMax = np.percentile(lambdamax_, significance.percentile)

    return lambdaMax


def circshuffling(zactmat, significance):
    np.random.seed()

    lambdamax_ = np.zeros(significance.nshu)
    for shui in range(significance.nshu):
        zactmat_ = np.copy(zactmat)
        for (neuroni, activity) in enumerate(zactmat_):
            cut = int(np.random.randint(significance.nbins * 2))
            zactmat_[neuroni, :] = np.roll(activity, cut)
        lambdamax_[shui] = getlambdacontrol(zactmat_)

    lambdaMax = np.percentile(lambdamax_, significance.percentile)

    return lambdaMax


def runSignificance(zactmat, significance):
    if significance.nullhyp == 'mp':
        lambdaMax = marcenkopastur(significance)
    elif significance.nullhyp == 'bin':
        lambdaMax = binshuffling(zactmat, significance)
    elif significance.nullhyp == 'circ':
        lambdaMax = circshuffling(zactmat, significance)
    else:
        print('ERROR !')
        print('    nyll hypothesis method ' + str(nullhyp) + ' not understood')
        significance.nassemblies = np.nan

    nassemblies = np.sum(significance.explained_variance_ > lambdaMax)
    significance.nassemblies = nassemblies

    return significance


def extractPatterns(actmat, significance, method):
    nassemblies = significance.nassemblies

    if method == 'pca':
        idxs = np.argsort(-significance.explained_variance_)[0:nassemblies]
        patterns = significance.components_[idxs, :]
    elif method == 'ica':
        from sklearn.decomposition import FastICA
        ica = FastICA(n_components=nassemblies)
        ica.fit(actmat.T)
        patterns = ica.components_
    else:
        print('ERROR !')
        print('    assembly extraction method ' + str(method) + ' not understood')
        patterns = np.nan

    if patterns is not np.nan:
        patterns = patterns.reshape(nassemblies, -1)

        # sets norm of assembly vectors to 1
        norms = np.linalg.norm(patterns, axis=1)
        patterns /= np.matlib.repmat(norms, np.size(patterns, 1), 1).T

    return patterns


def runPatterns(zactmat, method='ica', nullhyp='circ', nshu=1000,
                percentile=99, tracywidom=False):
    '''
    INPUTS

        zactmat:     activity matrix - numpy array (neurons, time bins)
                        should already be z-scored

        nullhyp:    defines how to generate statistical threshold for assembly detection.
                        'bin' - bin shuffling, will shuffle time bins of each neuron independently
                        'circ' - circular shuffling, will shift time bins of each neuron independently
                                                            obs: mantains (virtually) autocorrelations
                        'mp' - Marcenko-Pastur distribution - analytical threshold

        nshu:       defines how many shuffling controls will be done (n/a if nullhyp is 'mp')

        percentile: defines which percentile to be used use when shuffling methods are employed.
                                                                    (n/a if nullhyp is 'mp')

        tracywidow: determines if Tracy-Widom is used. See Peyrache et al 2010.
                                                (n/a if nullhyp is NOT 'mp')

    OUTPUTS

        patterns:     co-activation patterns (assemblies) - numpy array (assemblies, neurons)
        significance: object containing general information about significance tests
        zactmat:      returns zactmat

    '''

    nneurons = np.size(zactmat, 0)
    nbins = np.size(zactmat, 1)

    silentneurons = np.var(zactmat, axis=1) == 0
    if any(silentneurons):
        warnings.warn(f'Silent neurons detected: '
                      f'{np.where(silentneurons)[0].tolist()}')
    actmat_didspike = zactmat[~silentneurons, :]

    # # z-scoring activity matrix
    # actmat_ = stats.zscore(actmat_, axis=1)
    #
    # # Impute missing values.
    # imp = SimpleImputer(missing_values=np.nan, strategy='constant',
    #                     fill_value=0)
    # actmat_ = imp.fit_transform(actmat_.T).T

    # running significance (estimating number of assemblies)
    significance = PCA()
    significance.fit(actmat_didspike.T)
    significance.nneurons = nneurons
    significance.nbins = nbins
    significance.nshu = nshu
    significance.percentile = percentile
    significance.tracywidom = tracywidom
    significance.nullhyp = nullhyp
    significance = runSignificance(actmat_didspike, significance)
    if np.isnan(significance.nassemblies):
        return

    if significance.nassemblies < 1:
        print('WARNING !')
        print('    no assembly detected!')
        patterns = []
    else:
        # extracting co-activation patterns
        patterns_ = extractPatterns(actmat_didspike, significance, method)
        if patterns_ is np.nan:
            return

        # putting eventual silent neurons back (their assembly weights are defined as zero)
        patterns = np.zeros((np.size(patterns_, 0), nneurons))
        patterns[:, ~silentneurons] = patterns_
        # zactmat = np.copy(actmat)
        # zactmat[~silentneurons, :] = actmat_didspike


    return patterns, significance, zactmat


def computeAssemblyActivity(patterns, zactmat, zerodiag=True):
    nassemblies = len(patterns)
    nbins = np.size(zactmat, 1)

    assemblyAct = np.zeros((nassemblies, nbins))
    for (assemblyi, pattern) in enumerate(patterns):
        projMat = np.outer(pattern, pattern)
        projMat -= zerodiag * np.diag(np.diag(projMat))
        for bini in range(nbins):
            assemblyAct[assemblyi, bini] = \
                np.dot(np.dot(zactmat[:, bini], projMat), zactmat[:, bini])

    return assemblyAct


if __name__ == '__main__':
    toy = toyassemblies(membership=[[0, 1, 2, 3]],
                        actrate=[0.05],
                        actstrength=[10])
    act1 = stats.zscore(toyExample(toy, nbins=500), axis=1)

    toy = toyassemblies(membership=[[6, 7, 8, 9]],
                        actrate=[0.05],
                        actstrength=[10])
    act2 = stats.zscore(toyExample(toy, nbins=500), axis=1)
    acts = [act1, act2]

    toy = toyassemblies(membership=[[2, 3, 4, 5]],
                        actrate=[0.05],
                        actstrength=[10])
    act3 = stats.zscore(toyExample(toy, nbins=500), axis=1)
    acts = [act1, act2, act3]

    # Get patterns from first dataset.
    patterns = runPatterns(act1)[0]

    # Get activation strengths from all datasets.
    assemblyActs = []
    for act in acts:
        assemblyActs.append(computeAssemblyActivity(patterns, act))

    fig, axs = plt.subplots(len(acts), 2, sharey='col')
    for act, assemblyAct, ax in zip(acts,
                                    assemblyActs,
                                    axs):
        # Spikes and ensemble activation.
        ax[0].plot(assemblyAct.T, color='b', alpha=0.3)
        ax[0].set_ylabel('Activation strength')
        spike_ax = ax[0].twinx()
        spks = spike_ax.imshow(act, cmap='Reds')
        spike_ax.axis('tight')
        ax[0].set_zorder(spike_ax.get_zorder() + 1)
        ax[0].patch.set_visible(False)

        # Correlation matrix.
        r = ax[1].imshow(np.corrcoef(act))
        fig.colorbar(spks, ax=spike_ax)
        fig.colorbar(r, ax=ax[1])

    plt.tight_layout()
    axs[0, 1].set_title('Correlations')
    plt.show()