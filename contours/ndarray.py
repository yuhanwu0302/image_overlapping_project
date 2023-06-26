import numpy as np 

a = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9]
])

a.cumsum(axis=0)

import csv
import matplotlib.pyplot as plt
data = []

with open('contours_1.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        row = row[0].strip('[]')
        row = row.split()
        row = [int(element) for element in row]
        data.append(row)

for row in data:
    print(row)

x_coords = [coord[0] for coord in data]
y_coords = [coord[1] for coord in data]
plt.plot(x_coords, y_coords)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Dynamic Plot')
plt.show()



import csv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

data = []

with open('contours_1.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        row = row[0].strip('[]')
        row = row.split()
        row = [int(element) for element in row]
        data.append(row)

fig, ax = plt.subplots()
line, = ax.plot([], [], color='black')
fill = ax.fill([], [], color='gray')

def update(frame):
    x_coords = [coord[0] for coord in data[:frame+1]]
    y_coords = [coord[1] for coord in data[:frame+1]]
    x_coords.append(x_coords[0])
    y_coords.append(y_coords[0])

    line.set_data(x_coords, y_coords)
    fill[0].remove()
    fill[0] = ax.fill(x_coords, y_coords, color='gray')[0]

    return line, fill

def init():
    ax.set_xlim(0, 100)  # Adjust the x-axis limits as needed
    ax.set_ylim(0, 100)  # Adjust the y-axis limits as needed

ani = FuncAnimation(fig, update, frames=len(data), init_func=init, blit=True)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Dynamic Image')
plt.show()
