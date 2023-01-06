from typing import List

from entity.filter import Filter


class Task(object):

    def __init__(self, name, func=None, conditions: dict = None):
        self.name = name
        self.func = func
        self.filter_list: List[Filter] = []
        if conditions:
            for key, val in conditions.items():
                self.filter_list.append(Filter.create(key, val))
