# Copyright 2021 The ParallelAccel Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""Classes for standard benchmarks and benchmark collections."""

from typing import (Text, List, Dict, Any,
                    Callable)
import abc
import functools as fct
import itertools
import inspect
import os
import time
import json
import sys
import pandas as pd
import pickle
import pprint
import traceback


import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from benchmarks.utils import mpl_style

PRINT_WIDTH = 80

class PlotNotImplementedError(NotImplementedError):
  pass


def print_center_pad(string, char, width=PRINT_WIDTH):
  padding_size = (width - len(string)) // 2
  print(char * padding_size + string + char * padding_size)


def collect_plots(dataframe, x_axis_name, y_axis_name, consolidation_parameter=None,
                  linestyle='solid', marker='o', label='', xscale='linear', yscale='linear',
                  colors=None):
  """
  Collect plots using data from `dataframe`.
  This function produces plots from the data in `dataframe`, where each plot
  plots column `x_axis_name` vs column `y_axis_name`. Values in
  `dataframe[y_axis_name]` can be scalar or a 1d array. If the values are 1d
  arrays, all values in the array will be plotted in the same style (i.e. same
  color, line-style, marker a.s.o.).

  Each single plot shows one or more lines (depending on if the values in
  `dataframe[y_axis_name]` are scalar of 1d arrays) for each distinct value
  in the column `dataframe[consolidation_parameter]`. If additinal independent
  parameters apart from `consolidation_parameter` or `x_axis_name` are present
  in the data, then a new plot will be produced for each distinct combination of
  values that these additinal parameters can take. For example, using the
  following dataframe

  'car-brand'   'number of seats' 'gas-usage (liter/100kms)'  'production year'
     VW               5                  40                       1980
     VW               5                  20                       1990
     VW               5                  10                       2010
     VW               2                  60                       1980
     VW               2                  40                       1990
     VW               2                  20                       2010
     Audi             5                  32                       1980
     Audi             5                  34                       1990
     Audi             5                  8                        2010
     Audi             2                  62                       1980
     Audi             2                  41                       1990
     Audi             2                  30                       2010
     BMW              5                  31                       1980
     BMW              5                  32                       1990
     BMW              5                  10                       2010
     BMW              2                  58                       1980
     BMW              2                  48                       1990
     BMW              2                  32                       2010


  with
  `x_axis_name = 'production year'`
  `y_axis_name = 'gas-usage (liter/100kms)'`
  `consolidation_parameter = 'number of seats'`

  this function will produce one figure for each value for 'car-brand',
  where each figure plots 'production year' vs 'gas-usage (liter/100kms)'
  with one line for each value of `numer of seats`.
  If there are more parameters in addition to 'car-brand',the function will
  produce one plot per combination of all these parameters.

  Args:
    dataframe: A pandas.DataFrame objet with the data.
    x_axis_name: The parameter to be plotted on the x-axis of the plots
    y_axis_name: The benchmark result to be plotted on the y-axis of the
      plots.
    consolidation_parameter: The parameter that should be collected into a
      single plot. For example, for `x_axis_name='num_discretes'`,
      `y_axis_name='walltime'` and `consolidation_param='num_subgraphs'`, each
      plot will show 'num_discretes' vs 'walltime' for all 'num_subgraphs'.
      Can be `None`, in which case one plot per combination of parameters
      `list(set(dataframe.columns.values) - set([consolidation_parameter,
                                                 x_axis_name, y_axis_name]))`
      will be produced.
    linestyle: A pyplot-linestyle, either `str` or tuples of the form
      `(int, (int, int))`
    marker: A pyplot-marker.
    label: A main label for the plots. Additional information will be appended
      to it.
    xscale: X-scale type for plotting, can be in
      {"linear", "log", "symlog", "logit"}
    yscale: Y-scale type for plotting, can be in
      {"linear", "log", "symlog", "logit"}
    colors: An iterable of strings, denoting the color-codes to be
      used for each distinct plot.

  Returns:
    List[Tuple[plt.Figure, plt.Axes]]: The plots.
  """
  parameter_names = list(set(dataframe.columns.values) - set(
      [consolidation_parameter, x_axis_name, y_axis_name]))
  plots = []
  if len(parameter_names) > 0:
    grouped = dataframe.groupby(parameter_names, as_index=False)
  else:
    grouped =[('', dataframe)]
  for data in grouped:
    param_values = data[0]
    # pd.DataFrame.groupby returns inconsistent types :(
    if not hasattr(param_values,'__iter__'):
      param_values = [param_values]

    subdf = data[1]
    fig, ax = plt.subplots()
    title = ', '.join([
      f"{name} = {value}"
      for name, value in zip(parameter_names, param_values)
    ])
    if consolidation_parameter is not None:
      grouped2 = subdf.groupby(consolidation_parameter, as_index=False)
    else:
      grouped2 = [('', subdf)]
    if colors is None:
      color_codes = map('C{}'.format, itertools.cycle(range(10)))
    else:
      color_codes = itertools.cycle(colors)
    for data2 in grouped2:
      tmpdf = data2[1]
      x = tmpdf[x_axis_name].to_numpy()
      y = tmpdf[y_axis_name].to_numpy()
      color = next(color_codes)
      if consolidation_parameter is not None:
        plot_label = f"{label}, {consolidation_parameter}={data2[0]}"
      else:
        plot_label = label
      y = np.stack(y, axis=0)
      ax.plot(x,
              y,
              color=color,
              marker=marker,
              linestyle=linestyle,
              label=plot_label)
      ax.set_xlabel(x_axis_name)
      ax.set_ylabel(y_axis_name)
      ax.set_xscale(xscale)
      ax.set_yscale(yscale)
    ax.set_title(title)
    ax.legend()
    plt.close(fig)
    plots.append((fig, ax))
  return plots

def add_lines(ax1, ax2, xlabel, ylabel):
  """
  Add lines from pyplot Axes object `ax2` to
  the pyplot Axes object `ax1`.

  Args:
    ax1, ax2: Two plt.Axes objects.
    xlabel: The x-label for ax1
    ylabel: The y-label for ax1
  """
  for line in ax2.lines:
    ax1.plot(line.get_data()[0],
             line.get_data()[1],
             color=line.get_color(),
             marker=line.get_marker(),
             linestyle=line.get_linestyle(),
             label=line.get_label())
    ax1.legend()
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)

def merge_plot_axes(axes, xlabel, ylabel, title='', xscale='linear', yscale='linear'):
  """
  Merge the plots from ax1 and ax2 into a single plot.

  Args:
    axes: A list of plt.Axes objects to be merged.
    xlabel: The x-label of the resulting plot.
    ylabel: The y-label of the resulting plot.
    title: A title for the new plot.
    xscale: X-scale type for plotting, can be in
      {"linear", "log", "symlog", "logit"}
    yscale: Y-scale type for plotting, can be in
      {"linear", "log", "symlog", "logit"}

  """
  f, a = plt.subplots()
  plt.close(f)

  for ax in axes:
    add_lines(a, ax, xlabel, ylabel)
  a.set_xscale(xscale)
  a.set_yscale(yscale)
  a.set_title(f'{title}')
  a.set_xlabel(xlabel)
  a.set_ylabel(ylabel)

  return f, a

def to_subplots(figures_and_axes, xlabel, ylabel,figsize=None):
  """
  Produce a single plot with subplots from a list of figures and axes.

  Args:
    figures_and_axes: A dict mapping consolidated parameter-names to a list of
      pyplot figures and axes.
    figsize: Tuple if int, the size of the figure.
    xlabel: String, the x-label.
    ylabel: String, the y-label.
  """
  num_plots = sum([len(plots) for plots in figures_and_axes.values()])
  fig, axes = plt.subplots(nrows=num_plots, figsize=figsize)
  if num_plots == 1:
    axes = [axes]

  n = 0
  for params, plots in figures_and_axes.items():
    for plot in plots:
      _, a = plot
      add_lines(axes[n], a, xlabel, ylabel)
      titletext=a.get_title().split('\n')[1:]
      axes[n].set_title('\n'.join(titletext))
      n += 1
  suptitle = a.get_title().split('\n')[0]
  if num_plots == 1:
    axes = axes[0]
  fig.suptitle(suptitle)
  plt.close(fig)
  return fig, axes

class Benchmark(metaclass=abc.ABCMeta):
  """Base class for all benchmarks.

  This class is intended to designate a type of benchmark, specifying
  the context and execution to be used by the benchmark.

  This class is intended to be instantiated with a set of context parameters,
  creating an object that represents the benchmark to be run over the specific
  set of parameters.

  This class should be subclassed to create a new type of benchmark.

  A complete benchmark must implement the following elements:
  * name - the name of the benchmark.
  * context_fn - a function which takes the items of the context_param_dict
    as input kwargs and generates a context for the benchmark.
  * benchmark_fn - a function which takes the generated context, runs the
    benchmark actions and returns a dict of results.

  To support plotting the benchmark must implement the get_plot() method.
  This is expected to take no required arguments and return a matplotlib compatible
  figure. The method can access results through `self`, and can either
  implement its own plotting logic, or use self.plot_helper

  run_configs is a list of run configurations of the form:
  [{'context_params': params,
    'results': {key: [values_each_repetition]}
  }]
  """

  def __init__(
      self,
      context_param_dict,
      results={},
  ):
    """Instantiate a benchmark object with a specific set of context parameters
    defined in context_param_dict.
    """
    self.context_param_dict = context_param_dict

    # Check that context_param_dict contains expected args
    context_fn_args = set(
        k for k, v in inspect.signature(self.context_fn).parameters.items()
        if v.default == inspect.Parameter.empty)
    context_param_dict_keys = set(context_param_dict.keys())

    if not context_fn_args.issubset(context_param_dict_keys):
      raise ValueError(
          f'context_fn expects the following arguments: {context_fn_args}.\n'
          f'context_param_dict provided: {context_param_dict_keys}')
    keys, values = zip(*self.context_param_dict.items())
    self.permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    self.finished_runs = set()
    self.results=results

  @staticmethod
  @abc.abstractmethod
  def context_fn(*args, **kwargs)->Any:
    """
    Generate a `context` for `benchmark_fn`.
    """
    pass

  @staticmethod
  @abc.abstractmethod
  def benchmark_fn(context)->Dict:
    """
    Benchmark function. The input `context` is the output
    of `context_fn`.
    """
    pass

  @property
  @abc.abstractmethod
  def name(self):
    pass

  @property
  def completed(self):
    return {tuple(v.values()) for v in self.permutations} == self.finished_runs

  @property
  def parameters(self):
    return self.context_param_dict

  @property
  def result_keys(self):
    """
    Return the keys of the output-dictionary of `benchmark_fn`.
    """
    keys = set()
    if self.completed:
      for run in self.results['run_configs']:
        keys |= set(run['results'].keys())
    return keys

  def results_to_dataframe(self):
    records = []
    for run in self.results['run_configs']:
      params  = dict(run['context_params'])
      params.update(run['results'])
      records.append(params)
    return pd.DataFrame.from_records(records)

  def plot_helper(self,
                  x_axis_name: Text,
                  y_axis_name: Text,
                  dataframe=None,
                  reduction_function: Callable = None,
                  label: Text = None,
                  consolidation_params: List[Text] = None,
                  filter: Dict[Text, List[Any]] = {},
                  xscale: Text = 'linear',
                  yscale: Text = 'linear',
                  markers: List[Text] = None,
                  linestyles: List = None,
                  colors: List[Text] = None,
                  average_over=None, figsize=None):
    """
    Generate plots from the benchmark data.
    Each benchmark defines a `self.parameters` dictionary
    which determines the parameter values for which to run the benchmark.
    This function lets the user generate different plots from the
    benchmark data using the parameter names defined in
    `self.parameters.keys()`.


    Args:
      x_axis_name: The parameter to be plotted on the x-axis of the plots
      y_axis_name: The benchmark result to be plotted on the y-axis of the
        plots.
      dataframe: An optional dataframe to use instead of using
        self.results_to_dataframe()
      reduction_function: A reduction function to be applied to the results
       `y_axis_name` of the benchmark, e.g. `lambda x: np.median(x)
        for a sequence-like result (e.g. walltimes of several repetitions).
        The output of `reduction_function` has to be a scalar.
      label: A main label for the plots. Additional information will be appended
        to it.
      consolidation_params: A list of parameters that should be collected into a
        single plot. For example, for `x_axis_name='num_discretes'`,
        `y_axis_name='walltime'` and `consolidation_params=['num_subgraphs']`, each
        plot will show 'num_discretes' vs 'walltime' for all 'num_subgraphs'.
        If `None`, defaults to
        ```
        consolidation_params = set(self.parameters.keys()) - set([x_axis_name])
        ```
        ie all independent parameters except `x_axis_name`.
        If empty or `[None]`, no consolidation will happen, i.e. each plot will
        contain a single line.
      filter: A dictionary whichs maps parameter-names to a list of desired values
        for this parameter. The results will be filtered for these parameter-value
        tuples.
      xscale: X-scale type for plotting, can be in
        {"linear", "log", "symlog", "logit"}
      yscale: Y-scale type for plotting, can be in
        {"linear", "log", "symlog", "logit"}
      markers: A list of pyplot-markers, one for each value in
        `consolidation_params`. Defaults to mps_style.MARKERS. Markers will be
        cyclicly repeated.
      linestyles: A list of pyplot-linestyles, either `str` or tuples of the
        form `(int, (int, int))`, one for each value in `consolidation_params`.
        Defaults to mps_style.LINESTYLES. Line styles will be cyclicly repeated.
      colors: An iterable of strings, denoting the color-codes to be
        used for each distinct plot. Colors are cyclicly repeated.
      average_over: List of parameter-names to average results over. For example.
        if the benchmark was run for several values of a seed, then `average_over=['seed']`
        will average results over the seeds.

    Returns:
      plt.Figure: The final figure.
      plt.Axes: The final axes.
      Dict[Text, List[Tuple[plt.Figure, plt.Axes]]]: The figures and axes for each
        value in `consolidation_params`.

    """
    if dataframe is None:
      df = self.results_to_dataframe()
    else:
      df = dataframe
    for param_name, filter_values in filter.items():
      df = df[df[param_name].isin(filter_values)]

    if reduction_function is not None:
      df[y_axis_name] = df[y_axis_name].apply(reduction_function)

    # remove all other results
    df = df.drop(
        columns=[name for name in self.result_keys if name != y_axis_name])

    if average_over is not None:
      group_names = list(
          set(df.columns.values) - set(average_over) - set([y_axis_name]))
      df = df.groupby(group_names, as_index=False).mean()
      df = df.drop(columns=average_over)
    if consolidation_params is None:
      consolidation_params = set(df.columns.values) - set([x_axis_name
                                                          ]) - self.result_keys
    if len(consolidation_params) == 0:
      #each plot has a single line
      consolidation_params = [None]

    figures_and_axes = {}
    if markers is None:
      markers = mpl_style.MARKERS

    if linestyles is None:
      linestyles = mpl_style.LINESTYLES
    markers = itertools.cycle(markers)
    linestyles = itertools.cycle(linestyles)
    if colors is None:
      color_codes = map('C{}'.format, itertools.cycle(range(10)))
    else:
      color_codes = itertools.cycle(colors)
    for param in consolidation_params:
      marker = next(markers)
      linestyle = next(linestyles)
      plots = collect_plots(df,
                            x_axis_name,
                            y_axis_name,
                            consolidation_parameter=param,
                            linestyle=linestyle,
                            marker=marker,
                            label=label,
                            colors=color_codes,
                            xscale=xscale,
                            yscale=yscale)
      for plot in plots:
        ax = plot[1]
        ax.set_title(f"{self.name}\n{ax.get_title()}")

      figures_and_axes[param] = plots

    # consolidate all axes into a single plot
    fig, axes = to_subplots(figures_and_axes, x_axis_name, y_axis_name,
                            figsize)
    return fig, axes, figures_and_axes


  def get_plot(self, **kwargs):
    """
    Produce a plot from the benchmark results.
    This function should take no required arguments, and return
    a single pyplot.Figure object.
    """
    raise PlotNotImplementedError('The get_plot method has not been implemented.')

  def cleanup_run_config(self):
    pass

  def save(self, filename=None):
    """Save Benchmark to a file."""
    if filename is None:
      name = '_'.join(self.name.lower().split(' '))
      filename = f'benchmark_{name}_{time.time()}'
    with open(filename, 'wb') as f:
      pickle.dump(self, f)

  @staticmethod
  def load(filename):
    """Load a Benchmark from a file."""
    with open(filename, 'rb') as f:
      return pickle.load(f)


  def run(self, repetitions=1):
    """Run the specified benchmark.

    Args:
      repetitions: number of times to run each run configuration.

    Returns:
      A dict representing the benchmark containing context parameters and
      benchmark_fn output for each run config. The dict is formatted as follows:

        {'benchmark_name': name,
         'run_configs': [{'context_params': params,
                          'results': {key: [values_each_repetition]}
                          }]
        }
    """
    if self.completed:
      print(f'benchmark {self.name} already ran')
      return self.results
    benchmarkname = '_'.join(f'{type(self).__name__}.{self.name.lower()}'.split(' '))
    self.results = {'benchmark_name': benchmarkname}

    print()
    print_center_pad(f' Running benchmark:  {benchmarkname}', '#')
    checkpointname = f"checkpoint_{benchmarkname}.pickle"

    t0 = time.time()
    for perm in self.permutations:
      print()
      print_center_pad(' Running with parameter configuration ', '-')
      print(f"\n".join([f"    {name}={val}" for name, val in perm.items()]))
      print_center_pad('', '-')
      param_values = tuple(perm.values())
      if param_values in self.finished_runs:
        continue

      context = self.context_fn(**perm)
      run_config = {'context_params': perm}
      try:
        results = [self.benchmark_fn(context) for _ in range(repetitions)]
      except Exception as e:
        print(f'Exception in benchmark {self.name}')
        traceback.print_exc()
        sys.stdout.flush()
        print('Continuing...')
        run_config['fail'] = repr(e)
        results = None

      # Consolidate keys [{k:v1}, {k:v2}] -> {k: [v1, v2]}
      if results:
        inv_results = {}
        for result in results:
          for k, v in result.items():
            inv_results[k] = inv_results.get(k, []) + [v]
        run_config['results'] = inv_results
      else:
        run_config['results'] = {}

      self.results['run_configs'] = self.results.get('run_configs',
                                                     []) + [run_config]

      self.finished_runs |= {param_values}
      self.cleanup_run_config()
      print(f'checkpointing {checkpointname}')
      self.save(checkpointname)

    print(
        f'Finished running benchmark: {benchmarkname} in {time.time() - t0} seconds'
    )
    # remove the checkpoint file
    if os.path.exists(checkpointname):
      print(f'clearing checkpoint {checkpointname}')
      os.remove(checkpointname)
    return self.results


class BenchmarkSuite:
  """Class defining a collection of benchmarks and functions to run and
  organize results from benchmark runs.

  This class is intented to be used by instantiating a new BenchmarkSuite object
  and then adding benchmark objects to be run.
  """

  def __init__(self, name, benchmarks=[]):
    """Initialize the benchmark suite.

    Args:
      benchmarks: An optional list of benchmarks.
    """
    self._benchmarks = list(benchmarks)
    self.name = name

  def append(self, benchmark):
    """Append a benchmark to the suite."""
    if not isinstance(benchmark, Benchmark):
      raise ValueError(
          f'benchmark must be of type Benchmark, not {type(benchmark).__name__}')
    self._benchmarks.append(benchmark)

  def extend(self, collection):
    """Extend the suite from a collection of benchmarks.
    This can be a list of benchmarks or another BenchmarkSuite.
    """
    if isinstance(collection, BenchmarkSuite):
      collection = collection._benchmarks
    for benchmark in collection:
      self.append(benchmark)

  @property
  def completed(self):
    return all(benchmark.completed for benchmark in self._benchmarks)

  @property
  def benchmarks(self):
    return self._benchmarks

  def run(self, iterations=1):
    """Run all benchmarks.

    Args:
      iterations: number of runs for each benchmark parameter set.
    """
    name = '_'.join(self.name.lower().split(' '))
    print_center_pad(f' Running Benchmark Suite: {self.name} ', '#')
    t0 = time.time()
    try:
      for benchmark in self._benchmarks:
        benchmark.run(iterations)
        name = '_'.join(self.name.lower().split(' '))
        self.save(f'results_dump_{name}_{time.time()}')

    except KeyboardInterrupt:
      print('KeyboardInterrupt, dumping results')
      name = '_'.join(self.name.lower().split(' '))
      self.save(f'results_dump_{name}_{time.time()}')
    finally:
      print( f'Finished running benchmark suite: {self.name} in '
              f'{time.time() - t0} seconds')

  def save(self, filename=None):
    """Save BenchmarkSuite to a file."""
    if filename is None:
      name = '_'.join(self.name.lower().split(' '))
      filename = f'benchmark_suite_{name}_{time.time()}'
    with open(filename, 'wb') as f:
      pickle.dump(self, f)

  @staticmethod
  def load(filename):
    """Load BenchmarksSuite from a file."""
    with open(filename, 'rb') as f:
      return pickle.load(f)

  def save_plots(self, filename=None):
    """Generate plots from results and save to file."""
    if filename is None:
      name = '_'.join(self.name.lower().split(' '))
      filename = f'plot_{name}_{time.time()}.pdf'
    with PdfPages(filename) as pdf:
      for benchmark in self._benchmarks:
        if benchmark.completed:
          try:
            figure = benchmark.get_plot()
            pdf.savefig(figure)
          except PlotNotImplementedError:
            print(f'plot not implemented for {type(benchmark).__name__}, skipping...')
        else:
          print(f'in save_plots: skipping uncompleted benchmark {type(benchmark).__name__}.{benchmark.name}')

  def get_results(self):
    """Return json results string."""
    output = {'suite_name': self.name,'benchmarks': []}
    for benchmark in self._benchmarks:
      if benchmark.completed:
        output['benchmarks'].append( benchmark.results)
    return output

  def print_results(self):
    """Pretty print results string."""
    output = {'suite_name': self.name,'benchmarks': []}
    for benchmark in self._benchmarks:
      if benchmark.completed:
        output['benchmarks'].append(benchmark.results_to_dataframe())
    pprint.pprint(output)
