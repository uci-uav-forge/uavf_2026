from typing import TypeVar


_ItemType = TypeVar("_ItemType")


def loop(items: list[_ItemType]):
    while True:
        for item in items:
            yield item
