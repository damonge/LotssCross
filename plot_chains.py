import os
import re
import argparse
import matplotlib.pyplot as plt
from getdist import plots, loadMCSamples

# Parse chain folders
parser = argparse.ArgumentParser('Simple script to plot chains with GetDist.')
parser.add_argument('--chains', '-c', nargs='+', required=True, type=str,
                    help='List of chains to plot')
parser.add_argument('--ignore_rows', '-ir', default=0.2,  type=float,
                    help='Ignore rows (default: 0.2)')
parser.add_argument('--smooth_scale_1D', '-s1D', default=0.5,  type=float,
                    help='1D smooth scale (default: 0.5)')
parser.add_argument('--smooth_scale_2D', '-s2D', default=0.5,  type=float,
                    help='2D smooth scale (default: 0.5)')
parser.add_argument('--legends', '-l', nargs='+', default=[], type=str,
                    help='List of legends')
parser.add_argument('--output', '-o', default='', type=str,
                    help='Output file')
args = parser.parse_args()

# Load the chains
chains = []
for path in args.chains:
    try:
        prefix = [x for x in os.listdir(path) if re.match('.*__..txt', x)]
    except FileNotFoundError:
        prefix = []
        print('WARNING: chains not found at {}'.format(path))
        continue
    prefix.sort()
    chain = loadMCSamples(os.path.join(path, prefix[0][:-6]), no_cache=False,
                          settings={'ignore_rows': args.ignore_rows,
                                    'smooth_scale_1D': args.smooth_scale_1D,
                                    'smooth_scale_2D': args.smooth_scale_2D})
    chains.append(chain)

# Load legends
legends = []
if args.legends and len(args.legends) == len(chains):
    legends = args.legends
else:
    legends = [re.split('/', x.root)[-2] for x in chains]

# Load output file
if args.output:
    output_file = args.output
else:
    output_file = os.path.abspath('_vs_'.join(legends) + '.pdf')

# Do the plot
g = plots.getSubplotPlotter(subplot_size=4)
g.settings.alpha_filled_add = 0.4
g.triangle_plot(chains, filled=True, legend_labels=legends)
# Save the plot
plt.savefig(output_file)
print('Success!! Saved plot at {}'.format(output_file))
