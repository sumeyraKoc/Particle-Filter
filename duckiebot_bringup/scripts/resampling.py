import copy
import random


def normalize_weights(particles):
    """Normalize weights in-place. Returns False if the weight sum is unusable."""
    if not particles:
        return False

    total = sum(max(0.0, p.weight) for p in particles)
    if total <= 0.0:
        uniform = 1.0 / len(particles)
        for p in particles:
            p.weight = uniform
        return False

    for p in particles:
        p.weight = max(0.0, p.weight) / total
    return True


def effective_sample_size(particles):
    """ESS = 1 / sum(w_i^2). Low ESS means particle degeneracy."""
    normalize_weights(particles)
    return 1.0 / sum(p.weight * p.weight for p in particles)


def systematic_resample(particles):
    """Systematic resampling. Returns deep-copied particles with uniform weights."""
    n = len(particles)
    if n == 0:
        return []

    normalize_weights(particles)

    start = random.random() / n
    positions = [start + i / n for i in range(n)]

    cumulative = []
    csum = 0.0
    for p in particles:
        csum += p.weight
        cumulative.append(csum)
    cumulative[-1] = 1.0

    indexes = []
    i = 0
    j = 0
    while i < n:
        if positions[i] <= cumulative[j]:
            indexes.append(j)
            i += 1
        else:
            j += 1
            if j >= n:
                indexes.append(n - 1)
                i += 1

    uniform = 1.0 / n
    new_particles = []
    for idx in indexes:
        p = copy.deepcopy(particles[idx])
        p.weight = uniform
        new_particles.append(p)

    return new_particles
