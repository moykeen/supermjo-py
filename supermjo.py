# coding=utf-8
"""
interface to SuperMjograph
(C) Makoto Tanahashi, 2017/08/23

    dependent modules: py-applescript, pyobjc, numpy, pandas, inspect
"""

import numpy as np
import pandas as pd
import applescript
import inspect

#%%

# to deal with huge array efficiently
class _MxCodecs (applescript.Codecs):

    def __init__(self):
        super().__init__()

        # append a custom pack method
        self.encoders[np.ndarray] = self.packnumpy

    def packnumpy(self, val):
        return self._packbytes(applescript.kae.typeData, val.data.tobytes())

class _MxAppleScript (applescript.AppleScript):
    # replace codecs by mine
    _codecs = _MxCodecs()


_import_autox_cmd = """
on run {activation, data_ptr, label_lists}
tell application "SuperMjograph"
if activation then
    activate
end if
set importer to first task
set fmid to frontmost of importer
if fmid is -1 then
    figure importer
end if
importWithAutoX importer to data_ptr label label_lists
end tell
end run
"""

_import_vsfirst_cmd = """
on run {activation, data_ptr, label_lists}
tell application "SuperMjograph"
if activation then
    activate
end if
set importer to first task
set fmid to frontmost of importer
if fmid is -1 then
    figure importer
end if
importVSFirst importer to data_ptr label label_lists
end tell
end run
"""

_import_asmulticol_cmd = """
on run {activation, data_ptr, arg_ncol, arg_label}
tell application "SuperMjograph"
if activation then
    activate
end if
set importer to first task
set fmid to frontmost of importer
if fmid is -1 then
    figure importer
end if
importAsMultiColumn importer to data_ptr ncol arg_ncol label arg_label
end tell
end run
"""

_clear_cmd = """
on run
tell application "SuperMjograph"
set importer to first task
clear importer
end tell
end run
"""

_figure_cmd = """
on run {fig_no}
tell application "SuperMjograph"
set importer to first task
figure importer of fig_no
end tell
end run
"""

_close_cmd = """
on run {fig_no}
tell application "SuperMjograph"
set importer to first task
close importer of fig_no
end tell
end run
"""

# default col check threshold
_col_check_default_th = 10

#%% functions

# check if column is accidently huge
def _col_check(n_col, too_many_col_check):
    if too_many_col_check and (n_col >= too_many_col_check):
        raise ValueError("""
Your array has %d columns. Haven't you mistaken row and colum?
If this is as you intended, invoke with too_many_col_check=False""" % (n_col))


def _plot_np(x, **param_dict):
    assert type(x) == np.ndarray, "only support Numpy ndarray"
    assert x.ndim <= 2, "high dimensional array is not supported"

    # additional parameter check
    activation = param_dict.get("activation", False)

    # data must be float
    if x.dtype != np.float:
        x = x.astype(np.float)

    # check size
    if x.ndim == 1:
        x = x[:, np.newaxis]
    n_row, n_col = x.shape
    _col_check(n_col, param_dict.get("too_many_col_check", _col_check_default_th))

    # make labels for legend
    if "labels" in param_dict:
        labels = param_dict["labels"]
        assert len(labels) == n_col, "labels do not match to series' count"

    else:
        # hack to retrive the argment variable name
        # https://stackoverflow.com/questions/2749796/how-to-get-the-original-variable-name-of-variable-passed-to-a-function
        frame = inspect.currentframe()
        frame = inspect.getouterframes(frame)[2]
        arg_string = inspect.getframeinfo(frame[0]).code_context[0].strip()
        args = arg_string[arg_string.find('(') + 1:-1].split(',')
        base_label = args[0]

        if n_col == 1:
            labels = [base_label]
        else:
            labels = [base_label + ", %d-th col" % j for j in range(n_col)]
        # print(labels)

    # finally, invoke the import script
    # (the array must be transposed to conform to mjograph's format)
    _MxAppleScript(_import_autox_cmd).run(activation, x.T, labels)

def _plot_np_twoarg(x, y, **param_dict):
    assert type(x) == np.ndarray, "only support Numpy ndarray"
    assert type(y) == np.ndarray, "only support Numpy ndarray"
    assert x.ndim <= 2, "high dimensional array is not supported"
    assert y.ndim <= 2, "high dimensional array is not supported"

    # additional parameter check
    activation = param_dict.get("activation", False)

    # data must be float
    if x.dtype != np.float:
        x = x.astype(np.float)
    if y.dtype != np.float:
        y = y.astype(np.float)

    # check size
    n_row_ind = len(x)

    # n_row, n_col = x.shape
    if y.ndim == 1:
        y = y[:, np.newaxis]
    n_row, n_col = y.shape
    # assert n_col == 1, "index must be a vector"
    assert n_row == n_row_ind, "x and y has different length"
    _col_check(n_col, param_dict.get("too_many_col_check", _col_check_default_th))

    assert x.squeeze().ndim == 1, "index must be a vector"
    if x.ndim == 1:
        x = x[:, np.newaxis]
    x = np.concatenate((x, y), axis=1)
    # print(x.shape)

    # make labels for legend
    if "labels" in param_dict:
        labels = param_dict["labels"]
        assert len(labels) == n_col, "labels do not match to series' count"

    else:
        frame = inspect.currentframe()
        frame = inspect.getouterframes(frame)[2]
        arg_string = inspect.getframeinfo(frame[0]).code_context[0].strip()
        args = arg_string[arg_string.find('(') + 1:-1].split(',')
        base_label = args[0] + " vs. " + args[1]

        if n_col == 1:
            labels = [base_label]
        else:
            labels = [base_label + ", %d-th col" % j for j in range(n_col)]
        # print(labels)

    # finally, invoke the import script
    # (the array must be transposed to conform to mjograph's format)
    _MxAppleScript(_import_vsfirst_cmd).run(activation, x.T, labels)

def _plot_np_asmulti(x, **param_dict):
    assert type(x) == np.ndarray, "only support Numpy ndarray"

    # additional parameter check
    activation = param_dict.get("activation", False)

    # data must be float
    if x.dtype != np.float:
        x = x.astype(np.float)

    # check size
    if x.ndim == 1:
        x = x[:, np.newaxis]
    n_row, n_col = x.shape
    _col_check(n_col, param_dict.get("too_many_col_check", _col_check_default_th))

    # make labels for legend
    if "labels" in param_dict:
        labels = param_dict["labels"]
        assert len(labels) == 1, "labels do not match to series' count"
        label = labels[0]

    else:
        frame = inspect.currentframe()
        frame = inspect.getouterframes(frame)[2]
        arg_string = inspect.getframeinfo(frame[0]).code_context[0].strip()
        args = arg_string[arg_string.find('(') + 1:-1].split(',')
        label = args[0]

    # finally, invoke the import script
    _MxAppleScript(_import_asmulticol_cmd).run(activation, x.T, n_col, label)


def _plot_pd(x, **param_dict):
    assert type(x) == pd.DataFrame, "support only Pandas DataFrame"

    # additional parameter check
    activation = param_dict.get("activation", False)

    # make equivalent numpy array
    if type(x.index) == pd.DatetimeIndex:
        y = np.concatenate((np.array(x.index.astype(np.int64) // 10**9)
                [:, np.newaxis], x.values), axis=1)

    else:
        # x.index.astype(np.int64)
        y = np.concatenate((np.array(x.index)[:, np.newaxis], x.values), axis=1)

    # data must be float
    if y.dtype != np.float:
        y = y.astype(np.float)

    # check size
    n_row, n_col = x.shape
    _col_check(n_col, param_dict.get("too_many_col_check", _col_check_default_th))

    # make labels for legend
    if "labels" in param_dict:
        labels = param_dict["labels"]
        assert len(labels) == n_col, "labels do not match to series' count"

    else:
        frame = inspect.currentframe()
        frame = inspect.getouterframes(frame)[2]
        arg_string = inspect.getframeinfo(frame[0]).code_context[0].strip()
        args = arg_string[arg_string.find('(') + 1:-1].split(',')
        base_label = args[0]
        labels = [base_label + ", " + str(col) for col in x.columns]

    # finally, invoke the import script
    _MxAppleScript(_import_vsfirst_cmd).run(activation, y.T, labels)


def _plot_pd_asmulti(x, **param_dict):
    assert type(x) == pd.DataFrame, "support only Pandas DataFrame"

    # additional parameter check
    activation = param_dict.get("activation", False)

    # make equivalent numpy array
    if type(x.index) == pd.DatetimeIndex:
        y = np.concatenate((np.array(x.index.astype(np.int64) // 10**9)
                [:, np.newaxis], x.values), axis=1)

    else:
        # x.index.astype(np.int64)
        y = np.concatenate((np.array(x.index)[:, np.newaxis], x.values), axis=1)

    # data must be float
    if y.dtype != np.float:
        y = y.astype(np.float)

    # check size
    n_row, n_col = x.shape
    _col_check(n_col, param_dict.get("too_many_col_check", _col_check_default_th))

    # make labels for legend
    if "labels" in param_dict:
        labels = param_dict["labels"]
        assert len(labels) == 1, "labels do not match to series' count"
        label = labels[0]

    else:
        frame = inspect.currentframe()
        frame = inspect.getouterframes(frame)[2]
        arg_string = inspect.getframeinfo(frame[0]).code_context[0].strip()
        args = arg_string[arg_string.find('(') + 1:-1].split(',')
        label = args[0]

    # finally, invoke the import script
    _MxAppleScript(_import_asmulticol_cmd).run(activation, y.T, n_col+1, label)


def clear():
    _MxAppleScript(_clear_cmd).run()

def figure(fig_id=0):
    if type(fig_id) != int:
        raise ValueError("figure id must be integer")
    _MxAppleScript(_figure_cmd).run(fig_id)

def close(fig_id=0):
    if type(fig_id) != int:
        raise ValueError("figure id must be integer")
    _MxAppleScript(_close_cmd).run(fig_id)


def plot(x, y=None, single_series=False, **param_dict):

    # preprocessing to unify types
    if type(x) == pd.Series:
        x = x.to_frame()

    if type(x) == list or type(x) == tuple:
        x = np.array(x)

    if y is not None and (type(y) == list or type(y) == tuple):
        y = np.array(y)

    # treat as a single multi-column series
    if single_series:
        assert y == None
        if type(x) == np.ndarray:
            _plot_np_asmulti(x, **param_dict)
        elif type(x) == pd.DataFrame:
            _plot_pd_asmulti(x, **param_dict)

        return

    if type(x) == np.ndarray:
        # if y is given, y is plotted vs. x, else x is plotted vs. the array's indices
        if y is not None and type(y) == np.ndarray:
            _plot_np_twoarg(x, y, **param_dict)
        else:
            _plot_np(x, **param_dict)

        return

    elif type(x) == pd.DataFrame:
        # plot contents vs. the dataframe's index
        _plot_pd(x, **param_dict)

        return


    raise ValueError("Could not plot your data")



#%% test code


if False:
# if __name__ == "main":

    # x1 = np.tile(np.arange(100), (4, 2)).T
    # figure(1)
    # plot(x1, activation=True, too_many_col_check=False)
    #
    # df1 = pd.DataFrame(data=np.random.randn(10, 3))
    # df1.columns = ["abc", "def", "xyz"]
    # df1.set_index(pd.date_range("2017-08-30", periods=len(df1)), inplace=True)
    # figure(2)
    # plot(df1)



    n = 5
    x1, y1 = np.meshgrid(np.arange(n), np.arange(n))
    x1 = x1.ravel()
    y1 = y1.ravel()
    # strength = np.random.randint(0, 100, len(x1))
    strength = np.zeros((n, n))
    for i, j in itertools.product(range(n), range(n)):
        strength[i, j] = i - j
    strength = strength.ravel()

    bubble = np.zeros((n, n))
    for i, j in itertools.product(range(n), range(n)):
        # bubble[i, j] = i * j
        bubble[i, j] = np.random.randn()
    bubble = bubble.ravel()

    plot(np.stack((x1, y1, bubble, strength)).T, is_single_series=True)
    # plot(x1, np.stack((x1, y1, strength)).T)
    # plot(np., is_single_series=True)


#%% test2
#
# if True:
#
#     # the raw data
#     # s_data = pd.io.pytables.read_hdf("big-data/s_full_to20170324.h5", "s_data")
