import csv
import os
import threading
from queue import Queue
from typing import List
from concurrent.futures import ThreadPoolExecutor

from entity.instance import Instance
from entity.schema import Schema
from entity.task import Task

DATASET_DIR_PATH = "E:/code/dataset"

pool = ThreadPoolExecutor(10)


class Dataset(object):
    """
    数据集
    对数据源有以下限制：
    1. 主键字段必须要有，默认名称为id，可以自定义名称；
    2. 主键必须位于第一列，标签必须位于第二列，主要是为了方便后面的计算；
    """

    def __init__(self, name, id_name="id", label_name=None):
        self.name = name
        self._path = f"{DATASET_DIR_PATH}/dataset_{name}.csv"
        self._tasks: List[Task] = []
        self._make_if_not_exists()
        self.schema = Schema.from_columns(id_name=id_name,
                                          label_name=label_name,
                                          columns=self._get_columns())

    def _make_if_not_exists(self):
        if not os.path.exists(self._path):
            with open(self._path, "w+") as f:
                pass

    def _get_columns(self):
        with open(self._path, "r+") as f:
            line = f.readline().strip()
        columns = line.split(",") if line else []
        return columns

    def set_columns(self, id_name, with_label, label_name, features: list):
        if not isinstance(features, list):
            raise TypeError(
                f"features should be type of list, but {type(features)} found")
        schema = Schema(id_name=id_name,
                        with_label=with_label,
                        label_name=label_name,
                        features=features)
        self.schema = schema

    def map_features(self, func):
        """逐个对所有特征数据进行计算，不包括id 和 label"""
        task = Task(name="map_features", func=func)
        self._tasks.append(task)
        return self

    def map_rows(self, func):
        """
        对所有行进行计算，不包括id 和 label
        该方法的func，入参是特征列表，返回值也是列表，但是返回的列表长度可能与原始长度不同，因此该方法处理后，特征的列名将变为无效，需要用户自己重新设置列名
        """
        task = Task(name="map_rows", func=func)
        self._tasks.append(task)
        return self

    def map_columns(self, func):
        """对所有列进行计算，不包括id 和 label"""
        task = Task(name="map_columns", func=func)
        self._tasks.append(task)
        return self

    def map_id(self, func):
        """逐个对所有的id进行处理"""
        task = Task(name="map_id", func=func)
        self._tasks.append(task)
        return self

    def map_ids(self, func):
        """对所有的id进行处理，func的入参是id列表"""
        task = Task(name="map_ids", func=func)
        self._tasks.append(task)
        return self

    def map_label(self, func):
        """逐个对所有的label 进行处理"""
        task = Task(name="map_label", func=func)
        self._tasks.append(task)
        return self

    def map_labels(self, func):
        """对所有的label进行处理，func的入参是label列表"""
        task = Task(name="map_labels", func=func)
        self._tasks.append(task)
        return self

    def filter(self, **conditions):
        """
        过滤器
        根据字段值过滤: age__gte=10, age__in=[10, 20]
        根据行号过滤: _dataset_index__gte=10
        """
        task = Task(name="filter", conditions=conditions)
        self._tasks.append(task)
        return self

    def merge(self, dataset: "Dataset"):
        """合并(左右合并)"""
        pass

    def concat(self, dataset: "Dataset"):
        """合并(上下合并)"""
        pass

    def _write_columns(self):
        columns = self.schema.get_columns()
        with open(self._path, "w+", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(columns)

    def _write_rows(self, rows: list):
        if not rows:
            return
        with open(self._path, "a+", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerows(rows)

    def _run_tasks(self, q: Queue):
        with open(self._path, "r") as f:
            counter = 0
            while True:
                counter += 1
                line = f.readline().strip()
                if not line:
                    break
                if counter == 1:
                    continue
                inst = Instance.from_line(line=line,
                                          with_label=self.schema.with_label)
                pool.submit(Dataset.run_tasks, inst, self._tasks, q)

    def _save_to_file(self, name: str, schema: Schema, q: Queue):
        rows = []
        new_dataset = Dataset(name,
                              id_name=self.schema.id_name,
                              label_name=self.schema.label_name)
        new_dataset.schema = schema
        new_dataset._write_columns()
        counter = 0
        while True:
            counter += 1
            data = q.get(block=True)
            if not data:
                break
            inst, filter_status = data[0], data[1]
            rows.append(inst.to_csv())
            if len(rows) >= 10000:
                print(f"write rows, i={counter}")
                new_dataset._write_rows(rows)
                rows = []
        new_dataset._write_rows(rows)

    def save_in_parallel(self, name, id_name, with_label, label_name,
                         features: list):
        schema = Schema(id_name=id_name,
                        with_label=with_label,
                        label_name=label_name,
                        features=features)
        q = Queue(maxsize=1000000)
        t1 = threading.Thread(target=self._run_tasks, args=(q, ))
        t2 = threading.Thread(target=self._save_to_file,
                              args=(name, schema, q))
        t1.start()
        t2.start()
        t1.join()
        q.put([])
        t2.join()

    def save(self, name, id_name, with_label, label_name, features: list):
        """将计算后的结果保存为新的数据集"""
        new_dataset = Dataset(name,
                              id_name=self.schema.id_name,
                              label_name=self.schema.label_name)
        new_dataset.set_columns(id_name=id_name,
                                with_label=with_label,
                                label_name=label_name,
                                features=features)
        new_dataset._write_columns()
        rows = []
        with open(self._path, "r") as f:
            counter = 0
            while True:
                counter += 1
                line = f.readline().strip()
                if not line:
                    break
                if counter == 1:
                    continue
                inst = Instance.from_line(line=line,
                                          with_label=self.schema.with_label)
                inst, filter_status = Dataset.run_tasks(
                    inst, self._tasks, None)
                if not filter_status:
                    continue
                rows.append(inst.to_csv())
                if len(rows) >= 10000:
                    print(f"write rows, i={counter}")
                    new_dataset._write_rows(rows)
                    rows = []
            new_dataset._write_rows(rows)

    @classmethod
    def run_tasks(cls, inst: Instance, tasks: List[Task], q: Queue):
        filter_status = True
        for task in tasks:
            if task.name == "map_features":
                inst.features = [task.func(val) for val in inst.features]
            elif task.name == "map_rows":
                new_features = task.func(inst.features)
                if not isinstance(new_features, list):
                    raise TypeError(
                        f"The func for map_rows which named '{task.func.__name__}' \
                            should return type of list, but type '{type(new_features)}' found"
                    )
                inst.features = new_features
            elif task.name == "map_id":
                inst.pk = task.func(inst.pk)
            elif task.name == "map_label":
                inst.label = task.func(inst.label)
            elif task.name == "filter":
                filter_status = inst.match_filter(task.filter_list)
                if not filter_status:
                    break
        if q:
            q.put((inst, filter_status))
        else:
            return inst, filter_status
