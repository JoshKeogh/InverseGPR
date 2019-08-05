import re
import numpy as np
from numpy import linalg


def format_CH4_energies(in_path, out_path):
    data = []
    with open(in_path, "r") as file:
        for i, line in enumerate(file):
            row = re.findall(r'-*\d.*?\d+', line)
            if len(row) == 16:
                data.extend(row)

        data = np.unique(np.array(data, float).reshape(-1, 16), axis=1)
        x, y = data[:, :15], data[:, 15]
        np.savez(out_path, x, y)
        print("Success")


#format_CH4_energies("data/CH4-Energies.txt", "data/CH4-Energies.npz")


def format_positions(x_path, y_path, out_path):
    x_data = []
    y_data = []
    with open(x_path, "r") as x:
        with open(y_path, "r") as y:
            for n, y_line in enumerate(y):
                y_match = re.search(r"-*\d*.?\d+", y_line)
                if y_match:
                    y_data.append(y_match.group())
                    row = []
                    while True:
                        x_line = x.readline()
                        if '----' in x_line:
                            while True:
                                x_line = x.readline()
                                x_match = re.findall(r'-*\d.*?\d+', x_line)
                                if len(x_match) > 2:
                                    row.extend((x_match[0], x_match[1], x_match[2]))
                                else:
                                    x_data.append(row)
                                    break
                            break

    np.savez(out_path, np.array(x_data, float), np.array(y_data, float))
    print("Success")


#format_positions("data/AllPositions.txt", "data/AllEnergies.txt", "data/Positions.npz")


def CoulumbEigenvalues(xyz, Z):
    num_atoms = len(xyz[0])
    num_samples = len(xyz)
    Coulomb = np.empty([num_samples, num_atoms, num_atoms])
    for n in range(num_samples):
        for a in range(num_atoms):
            for b in range(num_atoms):
                if a == b:
                    Coulomb[n][a][b] = 0.5*Z[a]**2.4
                else:
                    diff = linalg.norm(xyz[n][a, :] - xyz[n][b, :])
                    Coulomb[n][a][b] = Z[a]*Z[b]/diff

        # Get the eigenvalues
        Coulomb[n], v = linalg.eig(Coulomb[n])
        # Sort the numbers from smallest to largest
        Coulomb[n] = np.sort(Coulomb[n])

    return Coulomb[:, 0]


def format_CH4_Coulomb(in_path, out_path, Z=[0.77, 0.32, 0.32, 0.32, 0.32]):
    data = []
    with open(in_path, "r") as file:
        for i, line in enumerate(file):
            row = re.findall(r'-*\d.*?\d+', line)
            if len(row) == 16:
                data.extend(row)

    data = np.unique(np.array(data, float).reshape(-1, 16), axis=1)
    x, y = data[:, :15].reshape(-1, 5, 3), data[:, 15]

    eigs = CoulumbEigenvalues(x, Z)
    np.savez(out_path, eigs, y)
    print("Success")


#format_CH4_Coulomb("data/CH4-Energies.txt", "data/CH4-Coulomb.npz")
