import pickle
import matplotlib.pyplot as plt

# Load data from spectral TE test run
path = '/home/patriciaw/repos/IDTxl/dev/test_spectral_te/'
with open(path + 'pvalues.txt', 'rb') as fp:
    pvalues = pickle.load(fp)
with open(path + 'sign.txt', 'rb') as fp:
    sign = pickle.load(fp)
with open(path + 'te_distr.txt', 'rb') as fp:
    te_distr = pickle.load(fp)
with open(path + 'reconstructed_surr.txt', 'rb') as fp:
    reconstructed = pickle.load(fp)
with open(path + 'te_orig.txt', 'rb') as fp:
    te_orig = pickle.load(fp)
with open(path + 'te_surr_temp.txt', 'rb') as fp:
    te_surr_temp = pickle.load(fp)

max_scale = len(reconstructed)
target_scale = 7  # freq band that carries info transfer

# Plot histograms of surrogate data for all scales
f, axes = plt.subplots(max_scale, sharex=True, sharey=True)
count = 0
for ax in axes[0:-1]:
    ax.hist(te_distr[count][0])
    ax.plot([te_orig, te_orig], [0, 4], 'r')
    ax.set_title('Scale {0}'.format(count + 1))
    count += 1
axes[-1].hist(te_surr_temp)
ax.plot([te_orig, te_orig], [0, 4], 'r')
ax.set_title('Temporal surrogates')

mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())
plt.show()

# Fine-tune figure; make subplots close to each other and hide x ticks for
# all but bottom plot.
f.subplots_adjust(hspace=0)
plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
