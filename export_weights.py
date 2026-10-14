"""
Convert the trained model (connect4_ann.pt) into weights.js for the web game.

Run this in the same folder as connect4_ann.pt:
    python export_weights.py
Then put the generated weights.js next to index.html.
"""
import json
import torch

sd = torch.load("connect4_ann.pt", map_location="cpu")


def grab(key):
    return sd[key].numpy().round(5).tolist()


weights = {
    "w0": grab("net.0.weight"), "b0": grab("net.0.bias"),
    "w1": grab("net.2.weight"), "b1": grab("net.2.bias"),
    "w2": grab("net.4.weight"), "b2": grab("net.4.bias"),
}

with open("weights.js", "w") as f:
    f.write("const WEIGHTS = " + json.dumps(weights, separators=(",", ":")) + ";\n")

print("Wrote weights.js")
