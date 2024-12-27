'''

NOTE: Modification for magnitude handling.

Original code from. 

Yu, L. et al. (2019). Identifying Exoplanets with Deep Learning III: Automated Triage and Vetting of TESS Candidates. *The Astronomical Journal*, 158(1), 25.

See also the original Shallue & Vanderburg paper:

Shallue, C. J., & Vanderburg, A. (2018). Identifying Exoplanets with Deep
Learning: A Five-planet Resonant Chain around Kepler-80 and an Eighth Planet
around Kepler-90. *The Astronomical Journal*, 155(2), 94.

Full text available at [*The Astronomical Journal*](http://iopscience.iop.org/article/10.3847/1538-3881/aa9e09/meta).

---------------------------------------------------------------------------------------------------------------------
global_view and local_view are modified to adapt BRITE data.

'''
import numpy as np

class SparseLightCurveError(Exception):
    """Indicates light curve with too few points in chosen time range."""
    pass


def median_filter(x, y, num_bins, bin_width=None, x_min=None, x_max=None):
  """Computes the median y-value in uniform intervals (bins) along the x-axis.

  The interval [x_min, x_max) is divided into num_bins uniformly spaced
  intervals of width bin_width. The value computed for each bin is the median
  of all y-values whose corresponding x-value is in the interval. Bins are overlapping if bin_width > bin_spacing.

  NOTE: x must be sorted in ascending order or the results will be incorrect.

  Args:
    x: 1D array of x-coordinates sorted in ascending order. Must have at least 2
        elements, and all elements cannot be the same value.
    y: 1D array of y-coordinates with the same size as x.
    num_bins: The number of intervals to divide the x-axis into. Must be at
        least 2.
    bin_width: The width of each bin on the x-axis. Must be positive, and less
        than x_max - x_min. Defaults to (x_max - x_min) / num_bins.
    x_min: The inclusive leftmost value to consider on the x-axis. Must be less
        than or equal to the largest value of x. Defaults to min(x).
    x_max: The exclusive rightmost value to consider on the x-axis. Must be
        greater than x_min. Defaults to max(x).

  Returns:
    1D NumPy array of size num_bins containing the median y-values of uniformly
    spaced bins on the x-axis.

  Raises:
    ValueError: If an argument has an inappropriate value.
    SparseLightCurveError: If light curve has too few points within given window.
  """
  if num_bins < 2:
    raise ValueError("num_bins must be at least 2. Got: %d" % num_bins)

  # Validate the lengths of x and y.
  x_len = len(x)
  if x_len < 2:
    raise SparseLightCurveError("len(x) must be at least 2. Got: %s" % x_len)
  if x_len != len(y):
    raise ValueError("len(x) (got: %d) must equal len(y) (got: %d)" % (x_len,
                                                                       len(y)))

  # Validate x_min and x_max.
  x_min = x_min if x_min is not None else x[0]
  x_max = x_max if x_max is not None else x[-1]
  if x_min >= x_max:
    raise ValueError("x_min (got: %d) must be less than x_max (got: %d)" %
                     (x_min, x_max))

  # This is unhelpful for sparse light curves. Use more specific error below
  # if x_min > x[-1]:
  #   raise ValueError(
  #       "x_min (got: %d) must be less than or equal to the largest value of x "
  #       "(got: %d)" % (x_min, x[-1]))

  # Drop light curves with no/few points in time range considered, or too little coverage in time
  in_range = np.where((x >= x_min) & (x <= x_max))[0]
  if (len(in_range) < 5) or (x[-1] - x[0] < (x_max - x_min) / 2 ):
    raise SparseLightCurveError('Too few points near transit')

  # Validate bin_width.
  bin_width = bin_width if bin_width is not None else (x_max - x_min) / num_bins
  if bin_width <= 0:
    raise ValueError("bin_width must be positive. Got: %d" % bin_width)
  if bin_width >= x_max - x_min:
    raise ValueError(
        "bin_width (got: %d) must be less than x_max - x_min (got: %d)" %
        (bin_width, x_max - x_min))

  bin_spacing = (x_max - x_min - bin_width) / (num_bins - 1)

  # # Bins with no y-values will fall back to the global median. - Don't do this for sparse light curves
  # result = np.repeat(np.median(y), num_bins)
  result = np.repeat(np.nan, num_bins)
  # For sparse light curves, fill empty bins with NaN to be interpolated over later.

  # Find the first element of x >= x_min. This loop is guaranteed to produce
  # a valid index because we know that x_min <= x[-1].
  x_start = 0
  while x[x_start] < x_min:
    x_start += 1

  # The bin at index i is the median of all elements y[j] such that
  # bin_min <= x[j] < bin_max, where bin_min and bin_max are the endpoints of
  # bin i.
  bin_min = x_min  # Left endpoint of the current bin.
  bin_max = x_min + bin_width  # Right endpoint of the current bin.
  j_start = x_start  # Inclusive left index of the current bin.
  j_end = x_start  # Exclusive end index of the current bin.

  for i in range(num_bins):
    # Move j_start to the first index of x >= bin_min.
    while j_start < x_len and x[j_start] < bin_min:
      j_start += 1

    # Move j_end to the first index of x >= bin_max (exclusive end index).
    while j_end < x_len and x[j_end] < bin_max:
      j_end += 1

    if j_end > j_start:
      # Compute and insert the median bin value.
      result[i] = np.median(y[j_start:j_end])

    # Advance the bin.
    bin_min += bin_spacing
    bin_max += bin_spacing

  result = fill_empty_bin(result)
  return result


def fill_empty_bin(y):
  """Fill empty bins by interpolating between adjacent bins.

  :param y: 1D array of y-coordinates with the same size as x. Empty bins should have NaN values.
  :return: same as y, but with NaNs replaced with interpolated values.
  """

  i = 0
  while i < len(y):
    if np.isnan(y[i]):
      left = i-1
      right = i+1
      # Find nearest non-NaN values on both sides
      while left >= 0 and np.isnan(y[left]):
        left -= 1
      while right < len(y) and np.isnan(y[right]):
        right += 1
      if left >= 0 and right < len(y):
        slope = (y[right] - y[left]) / (right - left)
        for j in range(left + 1, right):
          y[j] = y[left] + slope*(j - left)
      elif left < 0 and right < len(y):
        y[:right] = y[right]
      elif left >= 0 and right == len(y):
        y[left+1:] = y[left]
      else:
        raise ValueError('Light curve consists only of invalid values')
    i += 1
  return y

def phase_fold_time(time, period, t0):
  """Creates a phase-folded time vector.
  Original code: Astronet-Triage.

  result[i] is the unique number in [-period / 2, period / 2)
  such that result[i] = time[i] - t0 + k_i * period, for some integer k_i.

  Args:
    time: 1D numpy array of time values.
    period: A positive real scalar; the period to fold over.
    t0: The center of the resulting folded vector; this value is mapped to 0.

  Returns:
    A 1D numpy array.
  """
  half_period = period / 2
  result = np.mod(time + (half_period - t0), period)
  result -= half_period
  return result

def phase_fold_and_sort_light_curve(time, flux, period, t0):
  """Phase folds a light curve and sorts by ascending time.
  Original code: Astronet-Triage.
  Args:
    time: 1D NumPy array of time values.
    flux: 1D NumPy array of flux values.
    period: A positive real scalar; the period to fold over.
    t0: The center of the resulting folded vector; this value is mapped to 0.

  Returns:
    folded_time: 1D NumPy array of phase folded time values in
        [-period / 2, period / 2), where 0 corresponds to t0 in the original
        time array. Values are sorted in ascending order.
    folded_flux: 1D NumPy array. Values are the same as the original input
        array, but sorted by folded_time.
  """
  # Phase fold time.
  time = phase_fold_time(time, period, t0)

  # Sort by ascending time.
  sorted_i = np.argsort(time)
  time = time[sorted_i]
  flux = flux[sorted_i]

  return time, flux


def generate_view(time, flux, num_bins, bin_width, t_min, t_max,
                  normalize=True):
  """Generates a view of a phase-folded light curve using a median filter.
  Original code: Astronet-Triage.
  Args:
    time: 1D array of time values, phase folded and sorted in ascending order.
    flux: 1D array of flux values.
    num_bins: The number of intervals to divide the time axis into.
    bin_width: The width of each bin on the time axis.
    t_min: The inclusive leftmost value to consider on the time axis.
    t_max: The exclusive rightmost value to consider on the time axis.
    normalize: Whether to center the median at 0 and minimum value at -1.

  Returns:
    1D NumPy array of size num_bins containing the median flux values of
    uniformly spaced bins on the phase-folded time axis.
  """
  view = median_filter(time, flux, num_bins, bin_width, t_min,
                                     t_max)
  if normalize:
    #view -= np.median(view)
    #view /= np.abs(np.min(view))  # In pathological cases, min(view) is zero...
    
    # # modification to work with magnitude
    view -= np.median(view)
    view /= np.abs(np.max(view))  # In pathological cases, min(view) is zero...

  return (-1)*view # modification to match original AstroNet range


def global_view(time, flux, period, num_bins=201, 
                bin_width_factor=1.2/201
               ):
  """Generates a 'global view' of a phase folded light curve.
  Original code: Astronet-Triage.
  See Section 3.3 of Shallue & Vanderburg, 2018, The Astronomical Journal.
  http://iopscience.iop.org/article/10.3847/1538-3881/aa9e09/meta

  Args:
    time: 1D array of time values, sorted in ascending order.
    flux: 1D array of flux values.
    period: The period of the event (in days).
    num_bins: The number of intervals to divide the time axis into.
    bin_width_factor: Width of the bins, as a fraction of period.

  Returns:
    1D NumPy array of size num_bins containing the median flux values of
    uniformly spaced bins on the phase-folded time axis.
  """
  return generate_view(
      time,
      flux,
      num_bins=num_bins,
      bin_width=period * bin_width_factor,
      t_min=-period / 2,
      t_max=period / 2)


def twice_global_view(time, flux, period, num_bins=402, bin_width_factor=1.2 / 402):
  """Generates a 'global view' of a phase folded light curve at 2x the BLS period.
  Original code: Astronet-Triage.
  See Section 3.3 of Shallue & Vanderburg, 2018, The Astronomical Journal.
  http://iopscience.iop.org/article/10.3847/1538-3881/aa9e09/meta
  If single transit, this is pretty much identical to global_view.

  Args:
    time: 1D array of time values, sorted in ascending order, phase-folded at 2x period.
    flux: 1D array of flux values.
    period: The period of the event (in days).
    num_bins: The number of intervals to divide the time axis into.
    bin_width_factor: Width of the bins, as a fraction of period.

  Returns:
    1D NumPy array of size num_bins containing the median flux values of
    uniformly spaced bins on the phase-folded time axis.
  """
  return generate_view(
      time,
      flux,
      num_bins=num_bins,
      bin_width=period * bin_width_factor,
      t_min=-period,
      t_max=period)


def local_view(time,
               flux,
               period,
               duration,
               num_bins=61,
               bin_width_factor=0.16,
               num_durations=2):
  """Generates a 'local view' of a phase folded light curve.
  Original code: Astronet-Triage.
  See Section 3.3 of Shallue & Vanderburg, 2018, The Astronomical Journal.
  http://iopscience.iop.org/article/10.3847/1538-3881/aa9e09/meta
  Args:
    time: 1D array of time values, sorted in ascending order.
    flux: 1D array of flux values.
    period: The period of the event (in days).
    duration: The duration of the event (in days).
    num_bins: The number of intervals to divide the time axis into.
    bin_width_factor: Width of the bins, as a fraction of duration.
    num_durations: The number of durations to consider on either side of 0 (the
        event is assumed to be centered at 0).
  Returns:
    1D NumPy array of size num_bins containing the median flux values of
    uniformly spaced bins on the phase-folded time axis.
  """
  return generate_view(
      time,
      flux,
      num_bins=num_bins,
      bin_width=duration * bin_width_factor,
      t_min=max(-period / 2, -duration * num_durations),
      t_max=min(period / 2, duration * num_durations))