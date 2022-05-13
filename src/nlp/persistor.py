import os
from typing import Text, Optional


def get_persistor(name: Text) -> Optional["Persistor"]:
    """Returns an instance of the requested persistor.

    Currently, `aws`, `gcs`, `azure` and providing
    module paths are supported remote storages.
    """
    if name == "qcloud":
        return QcloudPersistor(
            os.environ.get("BUCKET_NAME"), os.environ.get("PATH")
        )
    return None


class Persistor(object):
    pass


class QcloudPersistor(Persistor):
    pass
