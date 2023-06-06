from typing import List, Generator, Any


def batch_list(L: list, N: int) -> Generator[list, None, None]:
    """batches list L into N size chunks. Returns a generator"""
    for i in range(0, len(L), N):
        yield L[i : i + N]


def flatten_list(L: List[List[Any]]) -> List[Any]:
    return [x for l in L for x in l]


def parse_str_to_list(s: str) -> List[str]:
    return [
        x.strip().strip("'").strip('"')
        for x in s.strip().strip("[").strip("]").split(",")
    ]
