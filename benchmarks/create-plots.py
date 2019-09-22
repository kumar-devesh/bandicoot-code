# create-plots.py
#
# Using the given results, plot a series of graphs for each task.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

filename = "results.csv"

# Sometimes, the benchmarking results show some strange outliers.  I'm not sure
# what the reason for this is, but we will simply filter those out when we
# detect them.
def filter_outliers(vec):
  # Anything more than 20% greater than the minimum time is what I consider an
  # outlier.
  m = min(vec)
  return [c for c in vec if c <= 1.2 * m]

results = pd.read_csv(filename)

# Assemble keys into one column.
results['key'] = results['device'] + '-' + results['backend'] + '-' + results['elem_type']

devices = pd.unique(results['device'])
styles = ['-', '--', '-.', ':']

device_style_dict = dict(zip(devices, styles[0:len(devices)]))

elem_types = pd.unique(results['elem_type'])
markers = ['x', 'o', '+', '.']

elem_type_marker_dict = dict(zip(elem_types, markers[0:len(elem_types)]))

backends = pd.unique(results['backend'])
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

backend_color_dict = dict(zip(backends, colors[0:len(backends)]))

for task in pd.unique(results['task']):
  # Collect each device/backend/element type combination (each will make up a line).
  z = results[results['task'] == task]

  # Clear the plot.
  plt.clf()
  gcf = plt.gcf()
  gcf.set_size_inches(14, 8)

  # Determine whether the x axis should be labeled 'elements' or 'rows'.
  x_label = 'rows'
  if len(pd.unique(z['cols'])) == 1:
    x_label = 'elements'

  plt.xlabel(x_label)
  plt.ylabel('runtime (s)')
  plt.grid()

  for key in pd.unique(z['key']):
    # Our x-axis will always be the number of rows.
    # We'll assemble the x/y values here.
    x_vals = pd.unique(z[z['key'] == key]['rows'])
    mean_vals = []
    std_vals = []
    for x in x_vals:
      # Compute the mean/std after filtering outliers.
      vals = filter_outliers(z[(z['key'] == key) & (z['rows'] == x)]['time'])
      mean_vals.append(np.mean(vals))
      std_vals.append(np.std(vals))

    # Now we need to figure out some details about how we will plot.
    one_row = z[z['key'] == key]
    one_row = one_row.iloc[0]
    color = backend_color_dict[one_row['backend']]
    style = device_style_dict[one_row['device']]
    marker = elem_type_marker_dict[one_row['elem_type']]

    # Everything is ready; now we can plot.
    plt.errorbar(x_vals, mean_vals, yerr=std_vals, color=color, fmt=(marker + style))

  # Add the legend and other parameters.
  plt.yscale('log')
  plt.xscale('log')
  plt.legend(pd.unique(z['key']), loc='best')
  plt.savefig(task + '.png')
