import time
import hashlib
from dataset import Dataset


def md5(s):
    return hashlib.md5(s.encode("utf8")).hexdigest()


if __name__ == "__main__":
    t1 = time.time()
    dataset = Dataset(name="100w_50x", id_name="id", label_name="y")
    dataset.map_id(lambda x: md5(x)).save(name="100w_50x_new",
                                          id_name=dataset.schema.id_name,
                                          with_label=dataset.schema.with_label,
                                          label_name=dataset.schema.label_name,
                                          features=dataset.schema.features)
    t2 = time.time()
    print(t2 - t1)
