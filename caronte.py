import numpy as np
from scipy.spatial.distance import pdist, squareform
import plotly.express as px
import plotly.graph_objects as go

class Persona:
    def __init__(self,
                 pos = [],
                 vel = [],
                 estado = [],
                 t_rec = []):
        self.pos = pos
        self.vel = vel
        self.estado = estado
        self.t_rec = t_rec
        self.c_time = 0

    def bordes(self, lim):
        out = np.array([(self.pos[0] - (lim[0] + r_pers)),
                        (-self.pos[0] + (lim[1] - r_pers)),
                        (self.pos[1] - (lim[2] + r_pers)),
                        (-self.pos[1] + (lim[3] - r_pers))])
        outbound = out < 0

        if outbound[0]: self.pos[0] = lim[0] + r_pers;
        if outbound[1]: self.pos[0] = lim[1] - r_pers;
        if outbound[2]: self.pos[1] = lim[2] + r_pers;
        if outbound[3]: self.pos[1] = lim[3] - r_pers;

        if outbound[0] | outbound[1]: self.vel[0] *= -1
        if outbound[2] | outbound[3]: self.vel[1] *= -1

        out = np.abs(out)

        ft = np.zeros((4, 2))
        ft[0,0] = (1/out[0]**2) * wall_force * vel
        ft[1,0] = -(1/out[1]**2) * wall_force * vel
        ft[2,1] = (1/out[2]**2) * wall_force * vel
        ft[3,1] = -(1/out[3]**2) * wall_force * vel
        self.add_force(ft)

    def recover(self):
        self.c_time += 1
        if self.c_time > self.t_rec:
            self.estado = 2
            if np.random.uniform() <= let_ratio:
                self.estado = 3
                self.vel = [0,0]

    def step(self):
        self.pos += self.vel
        self.bordes(limites)
        if self.estado == 1:
            self.recover()

    def contagio(self):
        if np.random.random() < prob_cont and self.estado == 0:
            self.estado = 1

    def add_force(self, f):
        for i in f:
            self.vel += i
        self.vel = (self.vel / np.linalg.norm(self.vel)) * vel

class Mundo:
    def __init__(self,
                 pos = [],
                 vel = [],
                 estado = [],
                 t_rec = []):
        self.pos = pos
        self.vel = vel
        self.estado = estado
        self.dist = np.zeros(len(estado))
        self.p = []
        self.t = 0
        for i in range(len(estado)):
            self.p.append(Persona(pos=pos[i], vel=vel[i], estado=estado[i], t_rec=t_rec[i]))

    def update_child(self):
        for i in range(len(self.p)):
            self.p[i].pos = self.pos[i]
            self.p[i].vel = self.vel[i]
            self.p[i].estado = self.estado[i]

    def update_master(self):
        for i in range(len(self.p)):
            self.pos[i] = self.p[i].pos
            self.vel[i] = self.p[i].vel
            self.estado[i] = self.p[i].estado

    def get_dist(self):
        self.dist = squareform(pdist(self.pos))
        a1, a2 = np.where(self.dist < r_avoid)
        c1, c2 = np.where(self.dist < r_cont)
        unique = (a1 < a2); a1 = a1[unique]; a2 = a2[unique]
        unique = (c1 < c2); c1 = c1[unique]; c2 = c2[unique]
        #if a1.size > 0: print("A", a1, a2, "t =", self.t)
        #if c1.size > 0: print("C", c1, c2, "t =", self.t)
        return (a1, a2, c1, c2)

    def step(self, tl):
        a1, a2, c1, c2 = self.get_dist()

        for i in range(len(self.p)):
            if i in c1 and self.p[i].estado == 1:
                contactos = np.where(c1 == i)[0]
                for j in contactos:
                    if self.p[c2[j]].estado == 0:
                        self.p[c2[j]].contagio()
            if i in c2 and self.p[i].estado == 1:
                contactos = np.where(c2 == i)[0]
                for j in contactos:
                    if self.p[c1[j]].estado == 0:
                        self.p[c1[j]].contagio()
            if i in a1 and self.p[i].estado != 3:
                vecinos = np.where(a1 == i)[0]
                f = []
                for j in vecinos:
                    dist = self.dist[i, a2[j]]
                    dir = (self.p[i].pos - self.p[a2[j]].pos) / dist
                    f.append((1 / dist ** soc_dist_factor) * soc_dist_atenuation * vel * dir)
                self.p[i].add_force(f)
            self.p[i].step()

        self.update_master()
        tl[self.t, :, 0:2] = self.pos
        tl[self.t, :, 2] = self.estado
        self.t += 1

p_num = 100
p_mov_ratio = 1
vel = 0.05
r_pers = 0.05
r_avoid = 0.5
soc_dist_factor = 3
soc_dist_atenuation = 1 #1 - Nearly total avoidance, 0.001 - Fluid avoidance
wall_force = 10000 #10000 for 1, 0.001 for 0.001
r_cont = 0.2
prob_cont = 0.2
t_rec_medio = 50
var_rec = 10
let_ratio = 0.02
limites = [-2,2,-2,2]
final = False
m = []
c = np.zeros(4)
iter = 100
timeline = np.zeros((iter, p_num, 3))

def begin():
    global m

    p_mov = int(p_num * p_mov_ratio)
    t_rec = np.ones(p_num) * np.rint(t_rec_medio + (np.random.random(p_num) - 0.5) * var_rec)

    pos_i = (np.random.random((p_num, 2)) - 0.5) * 4

    dir = np.random.random((p_mov)) * 2 * np.pi
    vel_i = np.zeros((p_num, 2))
    vel_i[0:p_mov, 0] = np.cos(dir) * vel
    vel_i[0:p_mov, 1] = np.sin(dir) * vel

    est_i = np.zeros(p_num)
    est_i[0] = 1

    m = Mundo(pos=pos_i, vel=vel_i, estado=est_i, t_rec=t_rec)

begin()

cscale = [[0, "dodgerblue"], [0.25, "dodgerblue"],
          [0.25, "tomato"], [0.5, "tomato"],
          [0.5, "gold"], [0.75, "gold"],
          [0.75, "grey"], [1, "grey"]]

fig_dict = {
    "data": [],
    "layout": {"height": 800,
               "width": 800,
               "autosize": False,
               "margin":dict(l=70, r=70, b=50, t=50, pad=4)},
    "frames": []}

fig_dict["layout"]["xaxis"] = {"range": limites[0:2]}
fig_dict["layout"]["yaxis"] = {"range": limites[2:4]}
fig_dict["layout"]["hovermode"] = "closest"
fig_dict["layout"]["sliders"] = {
    "args": [
        "transition", {
            "duration": 100,
            "easing": "linear"
        }
    ],
    "initialValue": "0",
    "plotlycommand": "animate",
    "values": np.linspace(0,iter,1),
    "visible": True
}
fig_dict["layout"]["updatemenus"] = [
    {
        "buttons": [
            {
                "args": [None, {"frame": {"duration": 100, "redraw": False},
                                "fromcurrent": True, "transition": {"duration": 100,
                                                                    "easing": "linear"}}],
                "label": "Play",
                "method": "animate"
            },
            {
                "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                  "mode": "immediate",
                                  "transition": {"duration": 0}}],
                "label": "Pause",
                "method": "animate"
            }
        ],
        "direction": "left",
        "pad": {"r": 10, "t": 87},
        "showactive": False,
        "type": "buttons",
        "x": 0.1,
        "xanchor": "right",
        "y": 0,
        "yanchor": "top"
    }
]

sliders_dict = {
    "active": 0,
    "yanchor": "top",
    "xanchor": "left",
    "currentvalue": {
        "font": {"size": 20},
        "prefix": "Iteration:",
        "visible": True,
        "xanchor": "right"
    },
    "transition": {"duration": 100, "easing": "linear"},
    "pad": {"b": 10, "t": 50},
    "len": 0.9,
    "x": 0.1,
    "y": 0,
    "steps": []
}

data_dict = {
    "x": m.pos[:,0],
    "y": m.pos[:,1],
    "mode": "markers",
    "marker":{"color": "dodgerblue",
              "size": 10,
              "sizemode": "diameter"}}
fig_dict["data"].append(data_dict)

for i in range(iter):
    m.step(timeline)
    frame = {"data": [], "name": i}
    data_dict = {
        "x": np.append(timeline[i,:,0], (-3, -3)),
        "y": np.append(timeline[i,:,1], (-3, -3)),
        "mode": "markers",
        "marker":{"color": np.append(timeline[i,:,2], (0, 3)),
                  "colorscale": cscale,
                  "size": 10,
                  "sizemode": "diameter"}}
    frame["data"].append(data_dict)
    fig_dict["frames"].append(frame)

    slider_step = {"args":[[i],
        {"frame": {"duration": 300, "redraw": False},
         "mode": "immediate",
         "transition": {"duration": 300}}],
         "label": i,
         "method": "animate"}
    sliders_dict["steps"].append(slider_step)

fig_dict["layout"]["sliders"] = [sliders_dict]

fig = go.Figure(fig_dict)

fig.show()
