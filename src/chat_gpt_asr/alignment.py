import jiwer
import numpy as np
import pandas as pd


def _cer(ref: str, hyp: str) -> float:
    """Return character error rate between the input strings.

    More flexible than jiwer.cer, which doesn't allow an empty string in the
    reference. NB: Only supports single strings, not lists of strings."""
    if ref == hyp == "":
        return 0
    if ref == "" or hyp == "":
        return 1
    return jiwer.cer(ref, hyp)


def align3(a: str, b: str, c: str) -> tuple[list[str], list[str], list[str]]:
    """Align three word sequences using DTW based on character edit distance."""
    a, b, c = a.split(), b.split(), c.split()
    m = np.zeros((len(a) + 1, len(b) + 1, len(c) + 1))  # DTW matrix
    bp = np.zeros((len(a) + 1, len(b) + 1, len(c) + 1)).tolist()  # Back pointers
    bp[0][0][0] = (0, 0, 0)
    moves = [
        (-1, -1, -1),
        (0, -1, -1),
        (-1, 0, -1),
        (-1, -1, 0),
        (0, 0, -1),
        (0, -1, 0),
        (-1, 0, 0),
    ]

    def _is_valid(move: tuple[int, int, int], i: int, j: int, k: int) -> bool:
        """Return whether the given move is valid for the given current indices."""
        return all(x + y >= 0 for x, y in zip(move, [i, j, k], strict=False))

    def _mean_cer(move: tuple[int, int, int], i: int, j: int, k: int) -> float:
        di, dj, dk = move
        word_a = a[i + di] if di == -1 else ""
        word_b = b[j + dj] if dj == -1 else ""
        word_c = c[k + dk] if dk == -1 else ""
        return np.mean(
            [_cer(word_a, word_b), _cer(word_a, word_c), _cer(word_b, word_c)]
        )

    for i in range(len(a) + 1):
        for j in range(len(b) + 1):
            for k in range(len(c) + 1):
                if i == j == k == 0:
                    continue
                options = [
                    m[i + di, j + dj, k + dk] + _mean_cer((di, dj, dk), i, j, k)
                    if _is_valid((di, dj, dk), i, j, k)
                    else np.inf
                    for di, dj, dk in moves
                ]
                min_option = np.argmin(options)
                m[i, j, k] = options[min_option]
                bp[i][j][k] = moves[min_option]

    # Backtrack to find best path
    i, j, k = np.array(m.shape) - 1
    out = []
    while i > 0 or j > 0 or k > 0:
        di, dj, dk = bp[i][j][k]
        i, j, k = i + di, j + dj, k + dk
        out.append(
            (
                a[i] if di == -1 else "",
                b[j] if dj == -1 else "",
                c[k] if dk == -1 else "",
            )
        )
    out.reverse()
    return tuple(map(list, zip(*out, strict=False)))


def print_alignment(alignment: tuple[list[str], ...]) -> None:
    print(pd.DataFrame(alignment))
