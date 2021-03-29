""" Nested list operation while preserving the original structure. """
from copy import deepcopy
from typing import List
from itertools import chain

__all__ = "ListKeeper"


def flatten_list(nested_list: List):
    """ Flatten nested list. """

    def _explore(_list):
        if type(_list) is list:
            new_list = []
            for i in _list:
                new_list.append(_explore(i))
            if all(type(i) is not list for i in new_list):
                return new_list
            else:
                return list(chain(*new_list))
        else:
            return _list

    return _explore(deepcopy(nested_list))


def restore_list(original_list: List, flatten_value: List):
    """ Restore nested structure with new values. """
    flatten_value = flatten_value.copy()

    def _restore(_list):
        if type(_list) is list:
            new_list = []
            for i in _list:
                new_list.append(_restore(i))
            return new_list
        else:
            return flatten_value.pop(0)

    return _restore(deepcopy(original_list))


class ListKeeper:
    """ Nested list operation while preserving the original structure. """

    def __init__(self, _list: List):
        self.original_list = _list
        self.flatten_list = flatten_list(_list)

    def restore_structure(self, values):
        assert len(values) == len(self.flatten_list), 'inconsistent length: {} != {}'.format(
            len(values), len(self.flatten_list))
        return restore_list(self.original_list, values)

    def __len__(self):
        return len(self.flatten_list)
