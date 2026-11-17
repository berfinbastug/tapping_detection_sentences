"""Put events (words, noise bursts) on an onset grid."""
import numpy as np


def draw_token_order(n, n_tokens, rng):
    """n token indices (0 ... n_tokens-1) drawn at random, never the same token twice in a row."""
    if n_tokens < 2 and n > 1:
        raise ValueError('need at least 2 tokens to avoid repeats')
    order, last = [], None
    for _ in range(n):
        k = int(rng.choice([j for j in range(n_tokens) if j != last]))
        order.append(k)
        last = k
    return order


def build_sequence(events, fs, *, soa_ms, shifts_ms, end, edge_fade_ms):
    """Place events so that event k's onset falls on k * soa_ms (+ shifts_ms[k]).

    events      : list of dicts with y (signal), on, off (onset / offset sample inside y);
                  any other keys (word, kind, block, ...) are copied into the returned table.
    fs          : sampling rate of all events.
    soa_ms      : onset-to-onset interval of the grid.
    shifts_ms   : None, or one value per event (ms added to its grid position; + = later).
    end         : 'offset' = the file ends at the last event's offset; 'slot' = after a full soa_ms slot.
    edge_fade_ms: tiny cosine fade at the start and end of the file.

    The first event's onset is at t = 0. Raises ValueError if events overlap or the result clips.
    Returns (y, table): y float32; table = one dict per event with position, grid_ms, shift_ms,
    onset_ms, offset_ms (+ the copied keys).
    """
    soa = soa_ms / 1000 * fs
    shifts = np.zeros(len(events)) if shifts_ms is None else np.asarray(shifts_ms, dtype=float)
    if len(shifts) != len(events):
        raise ValueError('shifts_ms needs one value per event')
    targets = [int(round(k * soa + s / 1000 * fs)) for k, s in enumerate(shifts)]
    if targets[0] < 0:
        raise ValueError('the first event cannot be shifted earlier than t = 0')

    if end == 'offset':
        length = targets[-1] + events[-1]['off'] - events[-1]['on'] + 1
    elif end == 'slot':
        length = int(round(len(events) * soa))
    else:
        raise ValueError("end must be 'offset' or 'slot'")

    out = np.zeros(length, dtype=np.float64)
    table, prev_off = [], None
    for k, (ev, t) in enumerate(zip(events, targets)):
        start = t - ev['on']
        a, b = max(start, 0), min(start + len(ev['y']), length)
        out[a:b] += np.asarray(ev['y'], dtype=float)[a - start:b - start]
        off = t + ev['off'] - ev['on']
        if prev_off is not None and t <= prev_off:
            raise ValueError(f'event {k + 1} starts before event {k} ends')
        prev_off = off
        row = {key: val for key, val in ev.items() if key not in ('y', 'on', 'off')}
        row.update(position=k + 1, grid_ms=round(k * soa_ms, 3), shift_ms=float(shifts[k]),
                   onset_ms=round(t / fs * 1000, 3), offset_ms=round(off / fs * 1000, 3))
        table.append(row)

    n = int(round(edge_fade_ms / 1000 * fs))
    if n > 0:
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n))
        out[:n] *= ramp
        out[-n:] *= ramp[::-1]
    peak = np.abs(out).max()
    if peak > 1:
        raise ValueError(f'the sequence peaks at {peak:.2f} > 1 (clipping): lower the level of words and noise')
    return out.astype(np.float32), table
